#!/bin/sh

SRC=en
TRG=de

mosesdecoder=/aoboturov/mosesdecoder

cat /data/input.txt | \
$mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $SRC | \
$mosesdecoder/scripts/tokenizer/tokenizer.perl -l $SRC -penn | \
python3 -m sockeye.translate --models /aoboturov/model --use-cpu | \
sed 's/\@\@ //g' | \
$mosesdecoder/scripts/tokenizer/detokenizer.perl -l $TRG > /output/output.txt
