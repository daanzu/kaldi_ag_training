#!/bin/bash

# Adapted from egs/mini_librispeech/s5/local/chain/tuning/run_tdnn_1h.sh

# Required Inputs:
#   conf/{mfcc,mfcc_hires,online_cmvn}.conf
#   data/train/{text,wav.scp,utt2spk}
#   data/lang/?????
# Writes To:
#   exp/?????

# Set -e here so that we catch if any executable fails immediately
set -euo pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
nj=8
train_set=train
test_sets=test
gmm=tri3b
nnet3_affix=

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=1h   # affix for the TDNN directory name
tree_affix=
train_stage=-10
get_egs_stage=-10
decode_iter=

# training options
chunk_width=140,100,160,50  # Standard default is 140,100,160 but try 140,100,160,50 for training with utterances of short commands
num_utts_subset=300  # default 300; you may want many more for short-utterance datasets <----------------
dropout_schedule='0,0@0.20,0.3@0.50,0'
common_egs_dir=
xent_regularize=0.1

# training options
srand=0
# remove_egs=true
remove_egs=false
reporting_email=

#decode options
test_online_decoding=true  # if true, it will run the last decoding stage.

# daanzu options
tdnnf_dim=1024  # Must be one of: 738,1024,1536
ivector_dim=100 # dimension of the extracted i-vector
respect_speaker_info=true
initial_lrate=
final_lrate=
num_epochs=5
num_gpus=1
spot=
decode_name=


# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

num_gpu_jobs=$num_gpus

function write_params() {
  for v in $*; do
    echo "${v}=${!v}" >> params.txt
  done
}

if ! cuda-compiled; then
  # cat <<EOF && exit 1
  cat <<EOF
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

if [ $stage -le 0 ]; then
  # Clear for run_personal_nnet3_ivector_common.sh below
  rm -rf data/train_sp data/train_sp_hires
fi

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 11" if you have already
# run those things.
bash run_personal_nnet3_ivector_common.sh --stage $stage \
                                  --nj $nj \
                                  --train-set $train_set \
                                  --gmm $gmm \
                                  --respect-speaker-info $respect_speaker_info \
                                  --ivector-dim $ivector_dim \
                                  --nnet3-affix "$nnet3_affix" || exit 1;

# Problem: We have removed the "train_" prefix of our training set in
# the alignment directory names! Bad!
gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp
tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
lang=data/lang_chain
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain${nnet3_affix}/tdnn${affix}_sp
train_data_dir=data/${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires

for f in $gmm_dir/final.mdl $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 10 ]; then
  echo "$0: creating lang directory $lang with chain-type topology"
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  if [ -d $lang ]; then
    if [ $lang/L.fst -nt data/lang/L.fst ]; then
      echo "$0: $lang already exists, not overwriting it; continuing"
    else
      echo "$0: $lang already exists and seems to be older than data/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r data/lang $lang
    silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
  fi
fi

if [ $stage -le 11 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 12 ]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.  The num-leaves is always somewhat less than the num-leaves from
  # the GMM baseline.
  # if [ -f $tree_dir/final.mdl ]; then
  #    echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
  #    exit 1;
  # fi
  steps/nnet3/chain/build_tree.sh \
    --frame-subsampling-factor 3 \
    --context-opts "--context-width=2 --central-position=1" \
    --cmd "$train_cmd" 3500 ${lores_train_data_dir} \
    $lang $ali_dir $tree_dir
fi

if [ $tdnnf_dim -eq 738 ]; then
  if [ $stage -le 13 ]; then
    # from mini_librispeech tdnn_1h
    mkdir -p $dir
    echo "$0: creating neural net configs using the xconfig parser";

    num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
    learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

    tdnn_opts="l2-regularize=0.03 dropout-proportion=0.0 dropout-per-dim-continuous=true"
    tdnnf_opts="l2-regularize=0.03 dropout-proportion=0.0 bypass-scale=0.66"
    linear_opts="l2-regularize=0.03 orthonormal-constraint=-1.0"
    prefinal_opts="l2-regularize=0.03"
    output_opts="l2-regularize=0.015"

    write_params num_targets learning_rate_factor tdnn_opts tdnnf_opts linear_opts prefinal_opts output_opts

    mkdir -p $dir/configs
    cat <<EOF > $dir/configs/network.xconfig
    input dim=$ivector_dim name=ivector
    input dim=40 name=input

    delta-layer name=delta
    no-op-component name=input2 input=Append(delta, Scale(1.0, ReplaceIndex(ivector, t, 0)))

    # the first splicing is moved before the lda layer, so no splicing here
    relu-batchnorm-dropout-layer name=tdnn1 $tdnn_opts dim=768
    tdnnf-layer name=tdnnf2 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=1
    tdnnf-layer name=tdnnf3 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=1
    tdnnf-layer name=tdnnf4 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=1
    tdnnf-layer name=tdnnf5 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=0
    tdnnf-layer name=tdnnf6 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
    tdnnf-layer name=tdnnf7 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
    tdnnf-layer name=tdnnf8 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
    tdnnf-layer name=tdnnf9 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
    tdnnf-layer name=tdnnf10 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
    tdnnf-layer name=tdnnf11 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
    tdnnf-layer name=tdnnf12 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
    tdnnf-layer name=tdnnf13 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
    linear-component name=prefinal-l dim=192 $linear_opts

    ## adding the layers for chain branch
    prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts small-dim=192 big-dim=768
    output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

    # adding the layers for xent branch
    prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts small-dim=192 big-dim=768
    output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
    steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
  fi
  initial_lrate=${initial_lrate:-0.002}
  final_lrate=${final_lrate:-0.0002}

elif [ $tdnnf_dim -eq 1024 ]; then
  if [ $stage -le 13 ]; then
    # from wsj tdnn_1g
    mkdir -p $dir
    echo "$0: creating neural net configs using the xconfig parser";

    num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
    learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

    tdnn_opts="l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true"
    tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
    linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
    prefinal_opts="l2-regularize=0.01"
    output_opts="l2-regularize=0.005"

    write_params num_targets learning_rate_factor tdnn_opts tdnnf_opts linear_opts prefinal_opts output_opts

    mkdir -p $dir/configs
    cat <<EOF > $dir/configs/network.xconfig
    input dim=$ivector_dim name=ivector
    input dim=40 name=input

    delta-layer name=delta
    no-op-component name=input2 input=Append(delta, Scale(1.0, ReplaceIndex(ivector, t, 0)))
    # no-op-component name=input2 input=delta

    relu-batchnorm-dropout-layer name=tdnn1 $tdnn_opts dim=1024 input=input2
    tdnnf-layer name=tdnnf2 $tdnnf_opts dim=1024 bottleneck-dim=96 time-stride=1
    tdnnf-layer name=tdnnf3 $tdnnf_opts dim=1024 bottleneck-dim=96 time-stride=1
    tdnnf-layer name=tdnnf4 $tdnnf_opts dim=1024 bottleneck-dim=96 time-stride=1
    tdnnf-layer name=tdnnf5 $tdnnf_opts dim=1024 bottleneck-dim=96 time-stride=0
    tdnnf-layer name=tdnnf6 $tdnnf_opts dim=1024 bottleneck-dim=96 time-stride=3
    tdnnf-layer name=tdnnf7 $tdnnf_opts dim=1024 bottleneck-dim=96 time-stride=3
    tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1024 bottleneck-dim=96 time-stride=3
    tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1024 bottleneck-dim=96 time-stride=3
    tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1024 bottleneck-dim=96 time-stride=3
    tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1024 bottleneck-dim=96 time-stride=3
    tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1024 bottleneck-dim=96 time-stride=3
    tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1024 bottleneck-dim=96 time-stride=3
    linear-component name=prefinal-l dim=192 $linear_opts

    ## adding the layers for chain branch
    prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts small-dim=192 big-dim=1024
    output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

    # adding the layers for xent branch
    prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts small-dim=192 big-dim=1024
    output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
    steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
  fi
  initial_lrate=${initial_lrate:-0.001}
  final_lrate=${final_lrate:-0.0001}
  # initial_lrate=0.001
  # final_lrate=0.00005

elif [ $tdnnf_dim -eq 1536 ]; then
  if [ $stage -le 13 ]; then
    # from chime5 tdnn_1b
    mkdir -p $dir
    echo "$0: creating neural net configs using the xconfig parser";

    num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
    learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

    tdnn_opts="l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim=true dropout-per-dim-continuous=true"
    tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
    linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
    prefinal_opts="l2-regularize=0.01"
    output_opts="l2-regularize=0.002"

    write_params num_targets learning_rate_factor tdnn_opts tdnnf_opts linear_opts prefinal_opts output_opts

    mkdir -p $dir/configs
    cat <<EOF > $dir/configs/network.xconfig
    input dim=$ivector_dim name=ivector
    input dim=40 name=input

    delta-layer name=delta
    no-op-component name=input2 input=Append(delta, Scale(1.0, ReplaceIndex(ivector, t, 0)))

    relu-batchnorm-dropout-layer name=tdnn1 $tdnn_opts dim=1536 input=input2
    tdnnf-layer name=tdnnf2 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
    tdnnf-layer name=tdnnf3 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
    tdnnf-layer name=tdnnf4 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
    tdnnf-layer name=tdnnf5 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=0
    tdnnf-layer name=tdnnf6 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
    tdnnf-layer name=tdnnf7 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
    tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
    tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
    tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
    tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
    tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
    tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
    tdnnf-layer name=tdnnf14 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
    tdnnf-layer name=tdnnf15 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
    linear-component name=prefinal-l dim=256 $linear_opts

    prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
    output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

    prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
    output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
    steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
  fi
  initial_lrate=${initial_lrate:-0.001}
  final_lrate=${final_lrate:-0.0001}

else
  echo "$0: ERROR: invalid tdnnf_dim: ${tdnnf_dim}"; exit 1
fi


if [ $stage -le 14 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/mini_librispeech-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi
  [ $num_gpu_jobs -gt 1 ] && sudo nvidia-smi -c 3
  write_params chunk_width dropout_schedule xent_regularize initial_lrate final_lrate num_epochs num_gpu_jobs stage train_stage num_utts_subset
  # --num-valid-egs-combine --num-train-egs-combine --num-egs-diagnostic ??? see steps/nnet3/chain/get_egs.sh

  steps/nnet3/chain/train.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir=$train_ivector_dir \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.0 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=$num_epochs \
    --trainer.frames-per-iter=3000000 \
    --trainer.optimization.num-jobs-initial=1 \
    --trainer.optimization.num-jobs-final=$num_gpu_jobs \
    --trainer.optimization.initial-effective-lrate=$initial_lrate \
    --trainer.optimization.final-effective-lrate=$final_lrate \
    --trainer.num-chunk-per-minibatch=128,64 \
    --egs.chunk-width=$chunk_width \
    --egs.dir="$common_egs_dir" \
    --egs.opts="--frames-overlap-per-eg 0 --num-utts-subset $num_utts_subset" \
    --cleanup.remove-egs=$remove_egs \
    --cleanup.preserve-model-interval=1000 \
    --use-gpu=true \
    --reporting.email="$reporting_email" \
    --feat-dir=$train_data_dir \
    --tree-dir=$tree_dir \
    --lat-dir=$lat_dir \
    --dir=$dir  || exit 1;
fi

steps/online/nnet3/prepare_online_decoding.sh \
  --mfcc-config conf/mfcc_hires.conf \
  $lang exp/nnet3${nnet3_affix}/extractor ${dir} ${dir}_online

exit 0;

if [ $stage -le 15 ]; then
  # Note: it's not important to give mkgraph.sh the lang directory with the
  # matched topology (since it gets the topology file from the model).
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang${decode_name:+_$decode_name} \
    $tree_dir $tree_dir/graph_${decode_name} || exit 1;
fi

if [ $stage -le 16 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
          $tree_dir/graph_${decode_name} data/${data}_hires ${dir}/decode_${decode_name}_${data} || exit 1
      # steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      #   data/lang_test_{tgsmall,tglarge} \
      #  data/${data}_hires ${dir}/decode_{tgsmall,tglarge}_${data} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

# Not testing the 'looped' decoding separately, because for
# TDNN systems it would give exactly the same results as the
# normal decoding.

if $test_online_decoding && [ $stage -le 17 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
    --mfcc-config conf/mfcc_hires.conf \
    $lang exp/nnet3${nnet3_affix}/extractor ${dir} ${dir}_online

  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      # note: we just give it "data/${data}" as it only uses the wav.scp, the
      # feature type does not matter.
      steps/online/nnet3/decode.sh \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --nj $nspk --cmd "$decode_cmd" \
        $tree_dir/graph_${decode_name} data/${data} ${dir}_online/decode_${decode_name}_${data} || exit 1
      # steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      #   data/lang_test_{tgsmall,tglarge} \
      #  data/${data}_hires ${dir}_online/decode_{tgsmall,tglarge}_${data} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

if [ $stage -le 18 ]; then
  export_dir=export/$(basename ${dir}_online)
  [ -d $export_dir ] && echo "$0: backing up $export_dir" && mv $export_dir ${export_dir}.$(date +"%Y%m%d_%H%M%S")
  echo "$0: exporting to $export_dir";
  mkdir -p $export_dir
  files="conf/  final.mdl  frame_subsampling_factor  ivector_extractor/  phones.txt  tree"
  for file in $files; do
    cp -rp ${dir}_online/$file $export_dir
  done
  cp -rp data/lang${decode_name:+_$decode_name} $export_dir
  mkdir -p $export_dir/local
  cp -rp data/local/dict $export_dir/local
  cp -rp data/local/lang $export_dir/local
  cp -rp $tree_dir/graph_${decode_name} $export_dir/graph
  # cp -rp $tree_dir/1.mdl $export_dir
  cp -rp local/chain_run_tdnn_1h.sh $export_dir
  cp -rp $dir/accuracy.report $export_dir
  cp -rp $0 $export_dir
fi

exit 0;


export affix=1h
export nnet3_affix=.1h
export dir=exp/chain${nnet3_affix}/tdnn${affix}_sp
export tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
export data=test
steps/online/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --nj 64 --cmd "run.pl" $tree_dir/graph_${decode_name} data/${data} ${dir}_online/decode_${decode_name}_${data}
steps/online/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --nj 64 --cmd "run.pl" --iter ../tdnn1h_sp/60-final $tree_dir/graph_${decode_name} data/${data} ${dir}_online/decode_${decode_name}_${data}
for x in exp/*/decode*; do [ -d $x ] && [[ $x =~ "$1" ]] && grep WER $x/wer_* | utils/best_wer.sh; done
for x in exp/chain/*/decode*; do [ -d $x ] && [[ $x =~ "$1" ]] && grep WER $x/wer_* | utils/best_wer.sh; done
