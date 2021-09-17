#!/usr/bin/env python3

import argparse, os, shutil

parser = argparse.ArgumentParser(description='Export trained model.')
parser.add_argument('type', choices=('personal', 'finetune'), help='Type of trained model.')
parser.add_argument('output_dir', nargs='?', default='exported_model', help='Directory to save the output model.')
parser.add_argument('-b', '--base_model_dir', default='kaldi_model_daanzu_20200905_1ep-mediumlm-base', help='Directory of model to copy base files from.')
parser.add_argument('-f', '--force', action='store_true', help='Force output: remove any existing output directory.')
args = parser.parse_args()

if not os.path.exists(args.base_model_dir):
    raise Exception('Base model directory does not exist.')
if os.path.exists(args.output_dir):
    if args.force:
        shutil.rmtree(args.output_dir)
    else:
        raise Exception('Output directory already exists.')
shutil.copytree(args.base_model_dir, args.output_dir, ignore=shutil.ignore_patterns('dict', 'tree_stuff'))
os.makedirs(os.path.join(args.output_dir, 'training'), exist_ok=True)

if args.type == 'personal':
    for name in 'final.mdl tree'.split():
        shutil.copy2(os.path.join('exp/chain/tdnn1h_sp_online', name), args.output_dir)
    for name in 'final.dubm final.ie final.mat global_cmvn.stats'.split():
        shutil.copy2(os.path.join('exp/chain/tdnn1h_sp_online', 'ivector_extractor', name), os.path.join(args.output_dir, 'ivector_extractor'))
    shutil.copy2('exp/chain/tdnn1h_sp/accuracy.report', os.path.join(args.output_dir, 'training'))
    shutil.copy2('params.txt', os.path.join(args.output_dir, 'training'))

elif args.type == 'finetune':
    for name in 'final.mdl'.split():
        shutil.copy2(os.path.join('exp/nnet3_chain/finetune', name), args.output_dir)
    shutil.copy2('exp/nnet3_chain/finetune/accuracy.report', os.path.join(args.output_dir, 'training'))

print(f"Wrote exported {args.type} model to {args.output_dir}")
print("NOTE: You still must run the following in your kaldi-active-grammar python environment:")
print("python -m kaldi_active_grammar compile_agf_dictation_graph -v -m [model_dir] G.fst")
