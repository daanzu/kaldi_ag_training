docker run -it --rm -v $(pwd):/mnt/input -w /mnt/input --user "$(id -u):$(id -g)" \
    --runtime=nvidia daanzu/kaldi_ag_training_gpu:2020-11-28 \
    $*
