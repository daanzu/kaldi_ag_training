# Kaldi AG Training Setup

[![Donate](https://img.shields.io/badge/donate-GitHub-pink.svg)](https://github.com/sponsors/daanzu)
[![Donate](https://img.shields.io/badge/donate-Patreon-orange.svg)](https://www.patreon.com/daanzu)
[![Donate](https://img.shields.io/badge/donate-PayPal-green.svg)](https://paypal.me/daanzu)

Docker image and scripts for training finetuned or completely personal Kaldi speech models. Particularly for use with [kaldi-active-grammar](https://github.com/daanzu/kaldi-active-grammar).

## Usage

All commands are run in the Docker container as follows. Training on the CPU should work, just much more slowly. To do so, remove the `--runtime=nvidia` and use the image `daanzu/kaldi_ag_training:2020-11-28` instead the GPU image. You can run Docker directly with the following parameter structure, or as a shortcut, use the `run_docker.sh` script (and edit it to suit your needs and configuration).

```bash
docker run -it --rm -v $(pwd):/mnt/input -w /mnt/input --user "$(id -u):$(id -g)" \
    --runtime=nvidia daanzu/kaldi_ag_training_gpu:2020-11-28 \
    [command and args...]
```

Example commands:

```bash
# Download and prepare base model (needed for either finetuning or personal model training)
wget https://github.com/daanzu/kaldi_ag_training/releases/download/v0.1.0/kaldi_model_daanzu_20200905_1ep-mediumlm-base.zip
unzip kaldi_model_daanzu_20200905_1ep-mediumlm-base.zip

# Prepare training dataset files
python3 convert_tsv_to_scp.py yourdata.tsv [optional output directory]

# Pick only one of the following:
# Run finetune training, with default settings
bash run_docker.sh bash run.finetune.sh kaldi_model_daanzu_20200905_1ep-mediumlm-base dataset
# Run completely personal training, with default settings
bash run_docker.sh bash run.personal.sh kaldi_model_daanzu_20200905_1ep-mediumlm-base dataset

# When training completes, export trained model
python3 export_trained_model.py {finetune,personal} [optional output directory]
# Finally run the following in your kaldi-active-grammar python environment (will take as much as an hour and several GB of RAM)
python3 -m kaldi_active_grammar compile_agf_dictation_graph -v -m [model_dir]

# Test a new or old model
python3 test_model.py testdata.tsv [model_dir]
```

### Notes

* To run either training, you must have a base model to use as a template. (For finetuning this is also the starting point of the model; for personal it is only a source of basic info.) You can use [this base model](https://github.com/daanzu/kaldi_ag_training/releases/download/v0.1.0/kaldi_model_daanzu_20200905_1ep-mediumlm-base.zip) from this project's release page. Download the zip file and extract it to the root directory of this repo, so the directory `kaldi_model_daanzu_20200905_1ep-mediumlm-base` is here.

* Kaldi requires the training data metadata to be in the SCP format, which is an annoying multi-file format. To convert the standard KaldiAG TSV format to SCP, you can run `python3 convert_tsv_to_scp.py yourdata.tsv dataset` to output SCP format in a new directory `dataset`. You can run these commands within the Docker container, or directly using your own python environment.
    * Even better, run `python3 convert_tsv_to_scp.py -l kaldi_model_daanzu_20200905_1ep-mediumlm-base/dict/lexicon.txt yourdata.tsv dataset` to filter out utterances containing out-of-vocabulary words. OOV words are not currently well supported by these training scripts.

* The audio data should be 16-bit Signed Integer PCM 1-channel 16kHz WAV files. Note that it needs to be accessible within the Docker container, so it can't be behind a symlink that points outside this repo directory, which is shared with the Docker container.

* There are some directory names you should avoid using in this repo directory, because the scripts will create & use them during training. Avoid: `conf`, `data`, `exp`, `extractor`, `mfcc`, `steps`, `tree_sp`, `utils`.

* Training may use a lot of storage. You may want to locate this directory somewhere with ample room available.

* The training commands (`run.*.sh`) accept many optional parameters. More info later.

    * `--stage n` : Skip to given stage.
    * `--num-utts-subset 3000` : You may need this parameter to prevent an error at the beginning of nnet training if your training data contains many short (command-like) utterances. (3000 is a perhaps overly careful suggestion; 300 is the default value.)

* I decided to try to treat the docker image as evergreen, and keep the things liable to change a lot like scripts in the git repo instead.

* The format of the training dataset input `.tsv` file is of tab-separated-values fields as follows: `wav_filename ignored ignored ignored text_transcript`

## Related Repositories

* [daanzu/speech-training-recorder](https://github.com/daanzu/speech-training-recorder): Simple GUI application to help record audio dictated from given text prompts, for use with training speech recognition or speech synthesis.
* [daanzu/kaldi-active-grammar](https://github.com/daanzu/kaldi-active-grammar): Python Kaldi speech recognition with grammars that can be set active/inactive dynamically at decode-time.

## License

This project is licensed under the GNU Affero General Public License v3 (AGPL-3.0-or-later). See the [LICENSE file](LICENSE) for details. If this license is problematic for you, please contact me.
