#!/usr/bin/env python3

import argparse, os, re

parser = argparse.ArgumentParser(description='Convert a TSV file to Kaldi SCP files.')
parser.add_argument('filename', help='Dataset TSV file to convert.')
parser.add_argument('output_dir', nargs='?', default='dataset', help='Directory to save the output files.')
parser.add_argument('-l', '--lexicon_file', default='kaldi_model_daanzu_20200905_1ep-mediumlm-base/dict/lexicon.txt', help='Filename of the lexicon file, for filtering out out-of-vocabulary utterances.')
parser.add_argument('--no_lexicon', action='store_true', help='Do not filter utterances based on lexicon to remove ones containing out-of-vocabulary wordstt.')
parser.add_argument('--no_sanitize', action='store_true', help='Do not sanitize the input text (lower casing, and removing punctuation).')
args = parser.parse_args()

if not os.path.exists(args.filename):
    raise Exception('File does not exist: %s' % args.filename)
os.makedirs(args.output_dir, exist_ok=True)
lexicon = set()
if args.lexicon_file:
    with open(args.lexicon_file, 'r') as f:
        for line in f:
            word = line.strip().split(None, 1)[0]
            lexicon.add(word)
else:
    print("WARNING: No lexicon file specified.")

utt2spk_dict, wav_dict, text_dict = {}, {}, {}
num_entries, num_dropped_lexicon, num_dropped_missing_wav = 0, 0, 0
with open(args.filename, 'r') as f:
    for line in f:
        num_entries += 1
        fields = line.strip().split('\t')
        text = fields[4]
        wav_path = fields[0]
        utt_id = os.path.splitext(os.path.basename(wav_path))[0]
        if not args.no_sanitize:
            text = text.lower()
            text = re.sub(r'[\-]', ' ', text)
            text = re.sub(r'[^a-z\']', '', text)
        if lexicon and any([word not in lexicon for word in text.split()]):
            num_dropped_lexicon += 1
            continue
        if not os.path.exists(wav_path):
            num_dropped_missing_wav += 1
            continue
        utt2spk_dict[utt_id] = utt_id
        wav_dict[utt_id] = wav_path
        text_dict[utt_id] = text

with open(os.path.join(args.output_dir, 'utt2spk'), 'w') as f:
    for (key, val) in utt2spk_dict.items():
        f.write('%s %s\n' % (key, val))
with open(os.path.join(args.output_dir, 'wav.scp'), 'w') as f:
    for (key, val) in wav_dict.items():
        f.write('%s %s\n' % (key, val))
with open(os.path.join(args.output_dir, 'text'), 'w') as f:
    for (key, val) in text_dict.items():
        f.write('%s %s\n' % (key, val))

if num_dropped_lexicon:
    print(f"{num_dropped_lexicon} ({num_dropped_lexicon / num_entries * 100:.1f}%) utterances dropped because they contained out-of-lexicon words.")
if num_dropped_missing_wav:
    print(f"{num_dropped_missing_wav} ({num_dropped_missing_wav / num_entries * 100:.1f}%) utterances dropped because couldn't find wav file at given path.")
if not len(text_dict):
    raise Exception("No utterances remaining! Failure!")
print(f"Wrote training dataset ({len(text_dict)} utterances) to: {args.output_dir}")
