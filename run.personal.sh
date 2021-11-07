# docker run -it --rm -v $(pwd):/mnt/input -v $(pwd)/work:/mnt/work -w /mnt/work --user "$(id -u):$(id -g)" daanzu/kaldi_ag_training:2020-11-28 bash run.personal.sh models/kaldi_model_daanzu_20200905_1ep-mediumlm data/standard2train --num-epochs 5 --stage -10
# docker run -it --rm -v $(pwd):/mnt/input -v $(pwd)/work:/mnt/work -w /mnt/work --user "$(id -u):$(id -g)" --runtime=nvidia daanzu/kaldi_ag_training_gpu:2020-11-28 bash run.personal.sh models/kaldi_model_daanzu_20200905_1ep-mediumlm data/standard2train --num-epochs 5 --stage -10

set -euxo pipefail

nice_cmd="nice ionice -c idle"
stage=-10
gmm_stage=0  # always stage+10

# Scan through arguments, checking for stage argument, which if included we need to use to set the gmm_stage
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    --stage)
    stage="$2"
    gmm_stage=$((stage+10))
    POSITIONAL+=("$1" "$2") # save it in an array for later
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

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

mkdir -p data/train data/dict conf exp
cp $model/conf/{mfcc,mfcc_hires,online_cmvn}.conf conf
cp $model/dict/{extra_questions.txt,lexiconp.txt,lexicon.txt,nonsilence_phones.txt,optional_silence.txt,silence_phones.txt} data/dict

[[ $stage -gt -10 ]] || rm -rf data/train/*
cp $dataset/{text,wav.scp,utt2spk} data/train
utils/fix_data_dir.sh data/train || exit 1
# ln -sfT /mnt/input/audio_data audio_data
# ln -sfT /mnt/input/audio_data/daanzu wav

# utils/fix_data_dir.sh data/train
$nice_cmd bash run_personal_gmm.sh --nj $(nproc) --stage $gmm_stage
$nice_cmd bash run_personal_chain_tdnn_1h.sh --nj $(nproc) $*
