#!/usr/bin/env python

import argparse, os

parser = argparse.ArgumentParser(description='Convert a TSV file to Kaldi SCP files.')
parser.add_argument('filename', help='The TSV file to convert.')
parser.add_argument('output_dir', default='dataset', help='The directory to save the output files.')
parser.add_argument('-l', '--lexicon_file', help='The name of the lexicon file, for filtering out out-of-vocabulary utterances.')
args = parser.parse_args()

if not os.path.exists(args.filename):
    raise Exception('File does not exist: %s' % args.filename)
os.makedirs(args.output_dir, exist_ok=True)
lexicon = set()
if args.lexicon_file:
    with open(args.lexicon_file, 'r') as f:
        for line in f:
            word, num = line.strip().split(None, 1)
            lexicon.add(word)

utt2spk_dict, wav_dict, text_dict = {}, {}, {}
with open(args.filename, 'r') as f:
    for line in f:
        fields = line.strip().split('\t')
        text = fields[4]
        wav_path = fields[0]
        utt_id = os.path.splitext(os.path.basename(wav_path))[0]
        if lexicon and any([word not in lexicon for word in text.split()]):
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

print(f"Wrote training dataset to {args.output_dir}")
