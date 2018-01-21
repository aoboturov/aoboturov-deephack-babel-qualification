#!/bin/sh

# Script was adapted from:
# http://data.statmt.org/rsennrich/wmt16_systems/en-de/translate.sh
# suffix of source language
SRC=en
# suffix of target language
TRG=de
# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=/aoboturov/mosesdecoder
# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=/aoboturov/subword-nmt
# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=/aoboturov/nematus
# theano device
device=cpu
# preprocess
cat /data/input.txt | \
$mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $SRC | \
$mosesdecoder/scripts/tokenizer/tokenizer.perl -l $SRC -penn | \
$mosesdecoder/scripts/recaser/truecase.perl -model truecase-model.$SRC | \
$subword_nmt/apply_bpe.py -c $SRC$TRG.bpe | \
# translate
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/nematus/translate.py \
     -m model.npz -v \
     -k 12 -n -p 2 --suppress-unk | \
# postprocess
sed 's/\@\@ //g' | \
$mosesdecoder/scripts/recaser/detruecase.perl | \
$mosesdecoder/scripts/tokenizer/detokenizer.perl -l $TRG > /output/output.txt
