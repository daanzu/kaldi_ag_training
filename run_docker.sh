docker run -it --rm -v $(pwd):/mnt/input -w /mnt/input --user "$(id -u):$(id -g)" \
    daanzu/kaldi_ag_training:2020-11-28 \
    $*
