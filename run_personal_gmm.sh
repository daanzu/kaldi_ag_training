#!/bin/bash

# Adapted from egs/rm/s5/run.sh

# Usage: run_personal_start.sh arpa_file openwebtext.arpa.gz --stage 0

# Required Inputs:
#   conf/mfcc.conf
#   data/train/{text,wav.scp,utt2spk}
#   dict/{extra_questions.txt,lexiconp.txt,lexicon.txt,nonsilence_phones.txt,optional_silence.txt,silence_phones.txt}
# Writes To:
#   data/local/dict/
#   data/lang/
#   exp/{make_mfcc,mono,mono_ali,tri1,tri1_ali,tri2b,tri2b_ali,tri3b}

set -e

stage=0
endstage=99
nj=16
test_decoding=false
dataset=train
dict_dir=data/dict
oov_word="<unk>"
arpa_file=
tree_num_leaves=2000
num_gauss=
. utils/parse_options.sh  # e.g. this parses the --stage option if supplied.

num_gauss=${num_gauss:-$(($tree_num_leaves * 5))}

. ./path.sh
. ./cmd.sh

if [ $stage -le 0 ] && [ $endstage -ge 0 ]; then
    # data preparation.
    rm -f data/train/{spk2utt,feats.scp,cmvn.scp}
    rm -rf data/local data/lang

    utils/fix_data_dir.sh data/$dataset || exit 1;
    # utils/utt2spk_to_spk2utt.pl data/$dataset/utt2spk > data/$dataset/spk2utt
    featdir=mfcc
    steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" data/$dataset exp/make_mfcc/$dataset $featdir
    # steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" --allow-downsample true data/$dataset exp/make_mfcc/$dataset $featdir
    steps/compute_cmvn_stats.sh data/$dataset exp/make_mfcc/$dataset $featdir
    utils/validate_data_dir.sh data/$dataset

    mkdir -p data/local/dict
    cp $dict_dir/* data/local/dict/
    utils/prepare_lang.sh data/local/dict "$oov_word" data/local/lang data/lang
    utils/validate_lang.pl data/lang

    # if [ -z "$arpa_file" ]; then
    #     mkdir data/local/tmp
    #     ngram-count -order 3 -write-vocab data/local/tmp/vocab-full.txt -wbdiscount -text data/local/dict/corpus.txt -lm data/local/tmp/lm.arpa
    #     arpa2fst --disambig-symbol=#0 --read-symbol-table=data/lang/words.txt data/local/tmp/lm.arpa data/lang/G.fst
    #     # ../kenlm/lmplz --text dict/corpus.txt --arpa data/local/tmp/lm.arpa -S 50% -o 3
    # else
    #     zcat -f "$arpa_file" | arpa2fst --disambig-symbol=#0 --read-symbol-table=data/lang/words.txt - data/lang/G.fst
    # fi
fi

if [ $stage -le 1 ] && [ $endstage -ge 1 ]; then
    # monophone
    steps/train_mono.sh --nj $nj --cmd "$train_cmd" data/train data/lang exp/mono
    # utils/mkgraph.sh data/lang exp/mono exp/mono/graph
fi

if [ $stage -le 2 ] && [ $endstage -ge 2 ]; then
    # tri1 [first triphone pass]
    steps/align_si.sh --nj $nj --cmd "$train_cmd" data/train data/lang exp/mono exp/mono_ali
    steps/train_deltas.sh --cmd "$train_cmd" $tree_num_leaves $num_gauss data/train data/lang exp/mono_ali exp/tri1
    # utils/mkgraph.sh data/lang exp/tri1 exp/tri1/graph
fi

if [ $stage -le 3 ] && [ $endstage -ge 3 ]; then
    # tri2b [LDA+MLLT] aka "tri3"
    steps/align_si.sh --nj $nj --cmd "$train_cmd" --use-graphs true data/train data/lang exp/tri1 exp/tri1_ali
    steps/train_lda_mllt.sh --cmd "$train_cmd" --splice-opts "--left-context=3 --right-context=3" $tree_num_leaves $num_gauss data/train data/lang exp/tri1_ali exp/tri2b
    # utils/mkgraph.sh data/lang exp/tri2b exp/tri2b/graph
fi

if [ $stage -le 4 ] && [ $endstage -ge 4 ]; then
    # tri3b [LDA+MLLT+SAT] aka "tri4"?
    steps/align_si.sh --nj $nj --cmd "$train_cmd" --use-graphs true data/train data/lang exp/tri2b exp/tri2b_ali
    #????? steps/align_fmllr.sh --nj 8 --cmd "$train_cmd" --use-graphs true data/train data/lang exp/tri2b exp/tri2b_ali
    steps/train_sat.sh $tree_num_leaves $num_gauss data/train data/lang exp/tri2b_ali exp/tri3b
    # utils/mkgraph.sh data/lang exp/tri3b exp/tri3b/graph
    # utils/mkgraph.sh data/lang_ug exp/tri3b exp/tri3b/graph_ug
    # steps/decode_fmllr.sh --config conf/decode.config --nj 1 --num-threads 8 --cmd "$decode_cmd" exp/tri3b/graph_ug data/test exp/tri3b/decode_ug
    # steps/cleanup/find_bad_utts.sh --nj 1 --cmd "$train_cmd" data/train data/lang exp/tri3b_ali exp/tri3b_cleanup
    # head exp/tri3b_cleanup/all_info.sorted.txt
fi

# if [ $stage -le 5 ] && [ $endstage -ge 5 ]; then
#     # tri3b_mmi [LDA+MLLT+SAT+MMI] aka "tri4_mmi"
#     steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" --use-graphs true data/train data/lang exp/tri3b exp/tri3b_ali
#     steps/make_denlats.sh --config conf/decode.config --nj $nj --cmd "$train_cmd" --transform-dir exp/tri3b_ali data/train data/lang exp/tri3b exp/tri3b_denlats
#     steps/train_mmi.sh data/train data/lang exp/tri3b_ali exp/tri3b_denlats exp/tri3b_mmi
# fi

# if [ $stage -le 8 ] && [ $endstage -ge 8 ]; then
#     local/kaldi/run_personal_chain_tdnn_1h.sh --stage 0 --num-epochs 20
# fi

exit

# for x in exp/*/decode*; do [ -d $x ] && [[ $x =~ "$1" ]] && grep WER $x/wer_* | utils/best_wer.sh; done
# for x in exp/chain/*/decode*; do [ -d $x ] && [[ $x =~ "$1" ]] && grep WER $x/wer_* | utils/best_wer.sh; done
