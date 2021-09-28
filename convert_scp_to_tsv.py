#!/usr/bin/env python

import argparse, glob, os, re

parser = argparse.ArgumentParser(description='Convert Kaldi SCP files to TSV file.')
parser.add_argument('input_dir', help='Dataset SCP directory to convert.')
parser.add_argument('filename', nargs='?', default='', help='TSV file to save the output.')
args = parser.parse_args()

if not args.filename:
    args.filename = args.input_dir.rstrip('/') + '.tsv'
if not os.path.exists(args.input_dir):
    raise ValueError('Input directory does not exist.')
if os.path.exists(args.filename):
    raise ValueError('Output file already exists.')

wav_dict, text_dict = {}, {}
with open(os.path.join(args.input_dir, 'wav.scp'), 'r') as f:
    for line in f:
        fields = line.rstrip('\n').split()
        wav_dict[fields[0]] = fields[1]
with open(os.path.join(args.input_dir, 'text'), 'r') as f:
    for line in f:
        fields = line.rstrip('\n').split(None, 1)
        text_dict[fields[0]] = fields[1] if len(fields) > 1 else ''

assert len(wav_dict) == len(text_dict), "Error: wav, text dicts are of different length!"

with open(args.filename, 'w') as f:
    for key in sorted(wav_dict.keys()):
        f.write('{}\t\t\t\t{}\n'.format(wav_dict[key], text_dict[key]))

print(f"Wrote TSV of {len(wav_dict)} utterances to {args.filename}")
