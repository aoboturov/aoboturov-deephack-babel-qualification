#!/bin/sh

SRC=en
TRG=de

mosesdecoder=/aoboturov/mosesdecoder
subword_nmt=/aoboturov/subword-nmt
marian=/aoboturov/marian
# preprocess
cat /data/input.txt | \
$mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $SRC | \
$mosesdecoder/scripts/tokenizer/tokenizer.perl -l $SRC -penn | \
$mosesdecoder/scripts/recaser/truecase.perl -model truecase-model.$SRC | \
$subword_nmt/apply_bpe.py -c $SRC$TRG.bpe | \
# translate
$marian/build/amun -m model.npz -s vocab.en.json -t vocab.de.json | \
# postprocess
sed 's/\@\@ //g' | \
$mosesdecoder/scripts/recaser/detruecase.perl | \
$mosesdecoder/scripts/tokenizer/detokenizer.perl -l $TRG > /output/output.txt
