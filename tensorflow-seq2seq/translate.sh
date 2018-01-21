#!/bin/sh

python -m nmt.nmt \
  --src=en \
  --tgt=de \
  --ckpt=/aoboturov/model/translate.ckpt \
  --out_dir=/aoboturov/model \
  --vocab_prefix=/aoboturov/model/vocab.bpe.32000 \
  --inference_input_file=/data/input.txt \
  --inference_output_file=/output/output.txt \
  --inference_ref_file=/aoboturov/model/train.tok.bpe.32000.de \
  --best_bleu_dir=/aoboturov/model
