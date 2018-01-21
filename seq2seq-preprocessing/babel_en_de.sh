# Borrowed from https://github.com/google/seq2seq/blob/master/bin/data/wmt16_en_de.sh

set -e

BASE_DIR=
OUTPUT_DIR=../data
MOSESDECODER_DIR=
SUBWORD_UNITS_DIR=

# Tokenize data
for f in ${OUTPUT_DIR}/*.de; do
  if [ ${f:-7} == ".tok.de" ]; then
  continue
  fi

  echo "Tokenizing $f..."
  perl $MOSESDECODER_DIR/scripts/tokenizer/tokenizer.perl -q -l de -threads 8 < $f > ${f%.*}.tok.de
done

for f in ${OUTPUT_DIR}/*.en; do
  if [ ${f:-7} == ".tok.en" ]; then
  continue
  fi

  echo "Tokenizing $f..."
  perl $MOSESDECODER_DIR/scripts/tokenizer/tokenizer.perl -q -l en -threads 8 < $f > ${f%.*}.tok.en
done

# Clean all corpora
for f in ${OUTPUT_DIR}/*.en; do
  fbase=${f%.*}
  echo "Cleaning ${fbase}..."
  ${MOSESDECODER_DIR}/scripts/training/clean-corpus-n.perl $fbase de en "${fbase}.clean" 1 80
done

# Create character vocabulary (on tokenized data)
${BASE_DIR}/bin/tools/generate_vocab.py --delimiter "" \
  < ${OUTPUT_DIR}/train.tok.clean.en \
  > ${OUTPUT_DIR}/vocab.tok.char.en
${BASE_DIR}/bin/tools/generate_vocab.py --delimiter "" \
  < ${OUTPUT_DIR}/train.tok.clean.de \
  > ${OUTPUT_DIR}/vocab.tok.char.de

# Create character vocabulary (on non-tokenized data)
${BASE_DIR}/bin/tools/generate_vocab.py --delimiter "" \
  < ${OUTPUT_DIR}/train.clean.en \
  > ${OUTPUT_DIR}/vocab.char.en
${BASE_DIR}/bin/tools/generate_vocab.py --delimiter "" \
  < ${OUTPUT_DIR}/train.clean.de \
  > ${OUTPUT_DIR}/vocab.char.de

# Create vocabulary for EN data
$BASE_DIR/bin/tools/generate_vocab.py \
   --max_vocab_size 50000 \
  < ${OUTPUT_DIR}/train.tok.clean.en \
  > ${OUTPUT_DIR}/vocab.50k.en \

# Create vocabulary for DE data
$BASE_DIR/bin/tools/generate_vocab.py \
  --max_vocab_size 50000 \
  < ${OUTPUT_DIR}/train.tok.clean.de \
  > ${OUTPUT_DIR}/vocab.50k.de \

# Learn Shared BPE
for merge_ops in 32000; do
  echo "Learning BPE with merge_ops=${merge_ops}. This may take a while..."
  cat "${OUTPUT_DIR}/train.tok.clean.de" "${OUTPUT_DIR}/train.tok.clean.en" | \
    ${SUBWORD_UNITS_DIR}/learn_bpe.py -s $merge_ops > "${OUTPUT_DIR}/bpe.${merge_ops}"

  echo "Apply BPE with merge_ops=${merge_ops} to tokenized files..."
  for lang in en de; do
    for f in ${OUTPUT_DIR}/*.tok.${lang} ${OUTPUT_DIR}/*.tok.clean.${lang}; do
      outfile="${f%.*}.bpe.${merge_ops}.${lang}"
      ${SUBWORD_UNITS_DIR}/apply_bpe.py -c "${OUTPUT_DIR}/bpe.${merge_ops}" < $f > "${outfile}"
      echo ${outfile}
    done
  done

  # Create vocabulary file for BPE
  cat "${OUTPUT_DIR}/train.tok.clean.bpe.${merge_ops}.en" "${OUTPUT_DIR}/train.tok.clean.bpe.${merge_ops}.de" | \
    ${SUBWORD_UNITS_DIR}/get_vocab.py | cut -f1 -d ' ' > "${OUTPUT_DIR}/vocab.bpe.${merge_ops}"

done

echo "All done."
