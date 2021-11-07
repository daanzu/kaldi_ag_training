# docker run -it --rm -v $(pwd):/mnt/input -v $(pwd)/work:/mnt/work -w /mnt/work --user "$(id -u):$(id -g)" daanzu/kaldi_ag_training:2020-11-28 bash run.finetune.sh models/kaldi_model_daanzu_20200905_1ep-mediumlm data/standard2train --num-epochs 5 --train-stage -10 --stage 1
# docker run -it --rm -v $(pwd):/mnt/input -v $(pwd)/work:/mnt/work -w /mnt/work --user "$(id -u):$(id -g)" --runtime=nvidia daanzu/kaldi_ag_training_gpu:2020-11-28 bash run.finetune.sh models/kaldi_model_daanzu_20200905_1ep-mediumlm data/standard2train --num-epochs 5 --train-stage -10 --stage 1

set -euxo pipefail

nice_cmd="nice ionice -c idle"

[[ $# -ge 2 ]] || exit 1

model=/mnt/input/$1; shift
dataset=/mnt/input/$1; shift

[[ -d $model ]] || exit 1
[[ -d $dataset ]] || exit 1

echo "base_model=${model#/mnt/input/}" >> params.txt
echo "train_dataset=${dataset#/mnt/input/}" >> params.txt

cat <<\EOF > cmd.sh
export train_cmd="utils/run.pl"
export decode_cmd="utils/run.pl"
export cuda_cmd="utils/run.pl"
# export cuda_cmd="utils/run.pl -l gpu=1"
EOF
cat <<\EOF > path.sh
export KALDI_ROOT=/opt/kaldi
export LD_LIBRARY_PATH="$KALDI_ROOT/tools/openfst/lib:$KALDI_ROOT/tools/openfst/lib/fst:$KALDI_ROOT/src/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH=$KALDI_ROOT/src/lmbin/:$KALDI_ROOT/../kaldi_lm/:$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/ivectorbin/:$KALDI_ROOT/src/kwsbin:$KALDI_ROOT/src/nnet3bin:$KALDI_ROOT/src/chainbin:$KALDI_ROOT/src/rnnlmbin:$PWD${PATH:+:$PATH}
export LC_ALL=C
EOF
ln -sf /opt/kaldi/egs/wsj/s5/steps
ln -sf /opt/kaldi/egs/wsj/s5/utils

extractor_dir=exp/nnet3_chain/extractor
tree_dir=exp/nnet3_chain/tree_sp

mkdir -p conf data/{lang/phones,finetune} exp exp/model_src_dir $extractor_dir
cp $model/conf/{mfcc,mfcc_hires}.conf conf/
cp $model/conf/online_cmvn.conf conf/  # Only needed if/for finetune_ivector_extractor
cp $model/conf/online_cmvn.conf $extractor_dir/
cp $model/conf/online_cmvn_iextractor $extractor_dir/ 2>/dev/null || true
cp $model/conf/splice.conf $extractor_dir/splice_opts
cp $model/{words,phones}.txt data/lang/
cp $model/disambig.int data/lang/phones/
cp $model/wdisambig_{words,phones}.int data/lang/phones/  # Only needed if/for mkgraph.sh
cp $model/{phones.txt,final.mdl,tree} exp/model_src_dir/
cp $model/cmvn_opts exp/model_src_dir/ 2>/dev/null || true

[[ -f $model/oov.int ]] && cp $model/oov.int data/lang/oov.int || echo "18" > data/lang/oov.int
[[ -f $model/frame_subsampling_factor ]] && cp $model/frame_subsampling_factor exp/model_src_dir/frame_subsampling_factor || echo "3" > exp/model_src_dir/frame_subsampling_factor

if [[ ! "$*" =~ .*"--finetune-ivector-extractor true".* ]]; then
    # Careful not to overwrite finetuned ivector_extractor, if we are finetuning it!
    cp $model/ivector_extractor/final.{ie,dubm,mat} $extractor_dir/
    cp $model/ivector_extractor/global_cmvn.stats $extractor_dir/
fi

echo "1:2:3:4:5:6:7:8:9:10:11:12:13:14:15" > data/lang/phones/context_indep.csl
echo "1:2:3:4:5:6:7:8:9:10:11:12:13:14:15" > data/lang/phones/silence.csl

. path.sh

# rm $tree_dir/ 2> /dev/null || true
mkdir -p $tree_dir
cp $model/{phones.txt,final.mdl,tree} $tree_dir/
cp -r $model/dict data/  # Only needed if/for finetune_tree
# cp $model/tree_stuff/topo data/lang/  # Only needed if/for finetune_tree
# cp $model/tree_stuff/sets.int data/lang/phones/  # Only needed if/for finetune_tree
mkdir -p exp/nnet3_chain/finetune/

if [[ ! "$*" =~ .*"--finetune-phonelm true".* ]]; then
    # Careful not to overwrite finetuned phonelm, if we are finetuning it!

    # Skip train.py::create_phone_lm()
    touch $tree_dir/{ali.1.gz,tree,final.mdl}  # Fake empty, to pacify the training script later

    # Skip train.py::create_denominator_fst()
    copy-transition-model $model/final.mdl exp/nnet3_chain/finetune/0.trans_mdl 2> /dev/null
    cp $model/tree $model/tree_stuff/{den,normalization}.fst exp/nnet3_chain/finetune/
fi

perl -ane '@A=split(" ",$_); $w = shift @A; $p = shift @A; @A>0||die;
    if(@A==1) { print "$w $p $A[0]_S\n"; } else { print "$w $p $A[0]_B ";
    for($n=1;$n<@A-1;$n++) { print "$A[$n]_I "; } print "$A[$n]_E\n"; } ' \
    < $model/lexiconp.txt > data/lang/lexiconp_pdp.txt || exit 1;
utils/lang/make_lexicon_fst.py --sil-prob=0.5 --sil-phone=SIL data/lang/lexiconp_pdp.txt | \
    fstcompile --isymbols=$model/phones.txt --osymbols=$model/words.txt --keep_isymbols=false --keep_osymbols=false | \
    fstarcsort --sort_type=olabel > data/lang/L.fst || exit 1

cp -r $dataset/{text,wav.scp,utt2spk} data/finetune
utils/fix_data_dir.sh data/finetune || exit 1
# ln -sfT /mnt/input/audio_data audio_data

$nice_cmd bash run_finetune_tdnn_1a_daanzu.sh --src-dir exp/model_src_dir --nj $(nproc) $*

# > cp -r work.test.per/data/lang/phones/* work.test.fin/data/lang/phones/
# > cp -r work.test.per/data/lang_chain/topo work.test.fin/data/lang/
