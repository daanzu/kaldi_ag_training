#!/usr/bin/env python3

import argparse, os, re

parser = argparse.ArgumentParser(description="Rebase a TSV file's wav files.")
parser.add_argument('filename', help='Dataset TSV file to rebase.')
parser.add_argument('new_wav_path', help='Path to directory containing the wav files.')
args = parser.parse_args()

if not os.path.exists(args.filename):
    raise Exception('File does not exist: %s' % args.filename)
if not os.path.exists(args.new_wav_path):
    raise Exception('Path does not exist: %s' % args.new_wav_path)

lines = []
with open(args.filename, 'r') as f:
    for line in f:
        fields = line.strip().split('\t')
        wav_path = fields[0]
        wav_path = re.sub(r'\\', '/', wav_path)
        wav_path = re.sub(r'^.*/', '', wav_path)
        wav_path = os.path.join(args.new_wav_path, wav_path)
        fields[0] = wav_path
        lines.append(fields)

with open(args.filename, 'w') as f:
    for line in lines:
        f.write('\t'.join(line) + '\n')
