#!/usr/bin/env python3

import argparse, os, re

parser = argparse.ArgumentParser(description="Normalize a TSV file's transcripts.")
parser.add_argument('filename', help='Dataset TSV file to normalize.')
args = parser.parse_args()

if not os.path.exists(args.filename):
    raise Exception('File does not exist: %s' % args.filename)

def normalize_script(script):
    script = re.sub(r'[\-]', ' ', script)
    script = re.sub(r'[,.?!:;"]', '', script)
    return script.strip().lower()

lines = []
with open(args.filename, 'r') as f:
    for line in f:
        fields = line.rstrip('\n').split('\t')
        text = fields[4]
        text = normalize_script(text)
        fields[4] = text
        lines.append(fields)

with open(args.filename, 'w') as f:
    for line in lines:
        f.write('\t'.join(line) + '\n')
