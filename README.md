# [DeepHack.Babel](https://deephack-babel.arktur.io/)

This time the topic of our new DeepHack hackathon is machine translation (MT).
At the hackathon we will tackle the task of semi-supervised neural MT: we'll try to improve a neural MT model with monolingual data.
However, our qualification task is traditional supervised MT It serves for familiarising participants with the MT field.

For this task we have decided to pick a well-studied language pair (English-German) and put no restrictions on the used data and model architecture.

## Submissions
| #  | Score   | Description                                                                    | Image                    |
|----|---------|--------------------------------------------------------------------------------|--------------------------|
| 1  | 0.17709 | OpenNMT-py with the `demo.rnn__acc_47.64_ppl_20.48_e13-fixed.pt` model         | aoboturov/deephack       |
| 2  | 0.23514 | OpenNMT with the `onmt_baseline_wmt15-all.en-de_epoch13_7.19_release.t7` model | aoboturov/deephack-v2    |
| 3  | 0.21108 | OpenNMT with the `standard_full_model_epoch4_4.34_release.t7` model            | aoboturov/deephack-v3    |
| 4  | 0.24765 | OpenNMT with the `standard_full_model_epoch9_3.65_release.t7` model            | aoboturov/deephack-v4    |
| 5  | 0.25349 | OpenNMT with the `standard_full_model_epoch13_3.51_release.t7` model           | aoboturov/deephack-v5    |
| 6  | 0.13219 | Fairseq-py with the `wmt14.en-de.fconv-py.tar.bz2` model                       | aoboturov/deephack-v6    |
| 7  | TLE     | Tensorflow NMT with the `ende_gnmt_model_8_layer` model                        | aoboturov/deephack-v7    |
| 8  | 0.07289 | Seq2Seq small model with computational budget 325000                           | aoboturov/deephack-v8    |
| 9  | 0.07624 | Seq2Seq small model with computational budget 667001                           | aoboturov/deephack-v9    |
| 10 | TLE     | Nematus pretrained single model                                                | kwakinalabs/abjskllld-v1 |
| 11 | TLE     | T2T default single model                                                       | kwakinalabs/cfbfgdfds-v1 |
| 12 | 0.18967 | Sockeye default model trained with custom data                                 | kwakinalabs/fkskskkas-v1 |
| 13 | 0.27924 | MarianNMT default single model                                                 | kwakinalabs/slkkajjss-v1 |

## Task
The task is translation of IT-texts from English into German, in line with WMT'16 IT translation task: http://www.statmt.org/wmt16/it-translation-task.html

There is no restriction on training data for the task.
You can use the data provided for the WMT'16 IT translation task or any other datasets.
Test data consists of novel texts in IT domain.

## Metric
We will evaluate submissions with BLEU score.

BLEU - the BLEU score measures how many words and ngrams (n consecutive words) overlap in a given translation and a reference translation.
The most commonly used BLEU version is BLEU-4, which considers words, bigrams, trigrams and 4-grams.
It also uses a penalty for too short traslations.

For local validation you can use the reference implementation of BLEU from MOSES or its Python version from Google.

## Submission
Your solution should be a Docker file which mounts two folders input & output and is run by the `docker run -v /path/to/input_data:/data -v /path/to/output:/output -t {image} {entry_point}` command.
Submission must be in the form of zip-archive containing file `metadata.json`.

`metadata.json` should contain the following fields:

- entry_point - command to run Docker container with
- image - Docker repo name

input folder contains a file input.txt with the test sentences in the source language (sentences that the model should translate).
The model should write translated sentences to output.txt in output folder.

We provide a sample solution.
The sample submission is included into this repository as `sample_submission.zip`.

Read [Tips to Reduce Docker Image Sizes](https://hackernoon.com/tips-to-reduce-docker-image-sizes-876095da3b34).

```
docker build -t aoboturov/deephack .
docker login
docker push aoboturov/deephack
```

## Baseline solution for DeepHack.Babel
- [Baseline solution for DeepHack.Babel](https://github.com/deepmipt/babel-baseline)

## [OpenNMT](http://opennmt.net/)
### w/ Demo Model
Built from the standard [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) with a [demo model](https://drive.google.com/file/d/0B6N7tANPyVeBWE9WazRYaUd2QTg/view?usp=sharing).
This model has to be [fixed](fix_for_model_deserialization.patch) because serialization format has changed.

`zip aoboturov-openmnt-demo-model.zip metadata.json run.sh`

### w/ Custom Model
#### Data Preprocessing

Data preprocessing takes about 40 mins for the small vocabulary size.
The 50k vocabulary runs in about the same time.

```
cut -f 3 gazetteerDE.tsv | sed 's/^"*//;s/"*$//' > gazetteer.en
cut -f 4 gazetteerDE.tsv | sed 's/^"*//;s/"*$//' > gazetteer.de

cat train_src/* > train.en
cat train_tgt/* > train.de

cat train.en | grep "^$" | wc -l
cat train.de | grep "^$" | wc -l

python3 ../data_cleaner.py train.en train.de

python3 preprocess.py -train_src data/train.en -train_tgt data/train.de -valid_src wmt16-it-task-references/Batch3a_en.txt -valid_tgt wmt16-it-task-references/Batch3a_de.txt -save_data data/out -src_vocab_size 1000 -tgt_vocab_size 1000 -max_shard_size 67108864
```

#### OpenNMT Custom Model training
Here's a [tutorial](http://forum.opennmt.net/t/training-english-german-wmt15-nmt-engine/29)

```
th tools/tokenize.lua < data/train.en > data/train.en.tok
th tools/tokenize.lua < data/train.de > data/train.de.tok
th tools/tokenize.lua < wmt16-it-task-references/Batch3a_en.txt > data/valid.en.tok
th tools/tokenize.lua < wmt16-it-task-references/Batch3a_de.txt > data/valid.de.tok

th preprocess.lua -train_src data/train.en.tok -train_tgt /data/train.de.tok -valid_src data/valid.en.tok -valid_tgt data/valid.de.tok -save_data data/standard_full

nohup th train.lua -data data/standard_full-train.t7  -save_model data/standard_full_model -gpuid 1 &
nvidia-smi
top
tail -f nohup.out
```

Once trained, the model has to be released (allows it to be ran on the CPU):

```
lua tools/release_model.lua -model data/standard_full_model_epoch4_4.34.t7 -gpuid 1
```

To resume a stopped training one could use:

```
nohup th train.lua -data data/standard_full-train.t7  -save_model data/standard_full_model -gpuid 1 -train_from data/standard_full_model_epoch10_3.58.t7 -continue &
```

    Training Sequence to Sequence with Attention model...
    Loading data from 'data/standard_full-train.t7'...
     - vocabulary size: source = 50004; target = 50004
     - additional features: source = 0; target = 0
     - maximum sequence length: source = 50; target = 51
     - number of training sentences: 4341009
     - number of batches: 67852
       - source sequence lengths: equal
       - maximum size: 64
       - average size: 63.98
       - capacity: 100.00%
    Building model...
     - Encoder:
       - word embeddings size: 500
       - type: unidirectional RNN
       - structure: cell = LSTM; layers = 2; rnn_size = 500; dropout = 0.3 (naive)
     - Decoder:
       - word embeddings size: 500
       - attention: global (general)
       - structure: cell = LSTM; layers = 2; rnn_size = 500; dropout = 0.3 (naive)
     - Bridge: copy
    Initializing parameters...
     - number of parameters: 84822004
    Preparing memory optimization...
     - sharing 69% of output/gradInput tensors memory between clones
    Start training from epoch 1 to 13...

| Epoch | Validation perplexity | File name |
|-------|-----------------------|-----------|
| 1  | 5.52 | data/standard_full_model_epoch1_5.52.t7 |
| 2  | 4.73 | data/standard_full_model_epoch2_4.73.t7 |
| 3  | 4.54 | data/standard_full_model_epoch3_4.54.t7 |
| 4  | 4.34 | data/standard_full_model_epoch4_4.34.t7 |
| 5  | 4.46 | data/standard_full_model_epoch5_4.46.t7 |
| 6  | 3.96 | data/standard_full_model_epoch6_3.96.t7 |
| 7  | 3.83 | data/standard_full_model_epoch7_3.83.t7 |
| 8  | 3.65 | data/standard_full_model_epoch8_3.65.t7 |
| 9  | 3.65 | data/standard_full_model_epoch9_3.65.t7 |
| 10 | 3.58 | data/standard_full_model_epoch10_3.58.t7 |
| 11 | 3.54 | data/standard_full_model_epoch11_3.54.t7 |
| 12 | 3.55 | data/standard_full_model_epoch12_3.55.t7 |
| 13 | 3.51 | data/standard_full_model_epoch13_3.51.t7 |

### Model averaging

```
lua tools/average_models.lua -models data/standard_full_model_epoch14_3.51.t7 data/standard_full_model_epoch15_3.51.t7 data/standard_full_model_epoch16_3.50.t7 data/standard_full_model_epoch17_3.49.t7 -force -gpuid 1
```

## [Seq2Seq](https://github.com/google/seq2seq)
No default model was provided - we had to train model by ourselves.
Preprocessing is done with custom script `seq2seq-preprocessing/babel_en_de.sh`.
Once finished the model could be trained with the `seq2seq-preprocessing/babel_training.sh` script.

Training process could be observer in the tensorboard:

```
nohup ./babel_en_de.sh &
nvidia-smi
top
tail -f nohup.out
tensorboard --logdir $MODEL_DIR
```

| Computational Budget | Validation Score |
|----------------------|------------------|
|  325000              | 0.07289          |
|  447001              | 0.07518          |
|  667001              | 0.07624          |

model.ckpt-447001.meta

## [Facebook AI Research Sequence-to-Sequence Toolkit](https://github.com/facebookresearch/fairseq-py)
Model training takes multiple more than two weeks - so there's a very strong incentive to use the pre-trained model.

The original [Facebook AI Research Sequence-to-Sequence Toolkit](https://github.com/facebookresearch/fairseq) would likely OOM because of memory restrictions of the Lua VM.

## [Tensorflow Seq2Seq NMT](https://www.tensorflow.org/tutorials/seq2seq)
The default pre-trained model has some problems.

To debug the checkpoint one could use:

```
python3 -m tensorflow.python.tools.inspect_checkpoint --all_tensor_names --file_name checkpoint.ckpt
```

## [Marian-NMT](https://github.com/marian-nmt/marian)
### Pre-trained models single models
One could use a [script](http://data.statmt.org/rsennrich/wmt16_systems/en-de/translate.sh) to run the inference for a single model.
All the pre-trained systems could be obtained from [The University of Edinburgh’s WMT16 systems](http://data.statmt.org/rsennrich/wmt16_systems/).

## [NEMATUS](https://github.com/EdinburghNLP/nematus)

## [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)
T2T default model is based on Google Brain team [notebook](https://colab.research.google.com/notebook#fileId=/v2/external/notebooks/t2t/hello_t2t.ipynb&scrollTo=oILRLCWN_16u).

```
gsutil cp gs://tensor2tensor-data/vocab.ende.32768 .
gsutil -q cp -R gs://tensor2tensor-checkpoints/ ckpt
```

## [Sockeye](https://github.com/awslabs/sockeye)
Training is done by running the following commands:

```
pip3 install sockeye --no-deps numpy mxnet-cu80==1.0.0 pyyaml typing

python3 -m learn_joint_bpe_and_vocab --input train.clean.en train.clean.de \
                                     -s 30000 \
                                     -o bpe.codes \
                                     --write-vocabulary bpe.vocab.en bpe.vocab.de

python3 -m apply_bpe -c bpe.codes --vocabulary bpe.vocab.en --vocabulary-threshold 50 < train.clean.en > train.clean.BPE.en
python3 -m apply_bpe -c bpe.codes --vocabulary bpe.vocab.de --vocabulary-threshold 50 < train.clean.de > train.clean.BPE.de

python3 -m apply_bpe -c bpe.codes --vocabulary bpe.vocab.en --vocabulary-threshold 50 < Batch3a_en.txt > Batch3a.BPE.en
python3 -m apply_bpe -c bpe.codes --vocabulary bpe.vocab.de --vocabulary-threshold 50 < Batch3a_de.txt > Batch3a.BPE.de

python -m sockeye.train -s train.clean.BPE.en \
                        -t train.clean.BPE.de \
                        -vs Batch3a.BPE.en \
                        -vt Batch3a.BPE.de \
                        --num-embed 256 \
                        --rnn-num-hidden 512 \
                        --rnn-attention-type dot \
                        --max-seq-len 60 \
                        --decode-and-evaluate 500 \
                        --device-ids 0 \
                        -o wmt_model
```

## The sentences to be translated

```
In the Bullets And Numbering dialog box , select Bullets from the List Type menu .
Acrobat 4.0 and 5.0 require that sounds be embedded and movies be linked .
A new bitmap object based on the pixel selection is created in the current layer , and the selected pixels are removed from the original bitmap object .
To finish the measurement , right-click and select Complete Measurement .
Select the text , frame , or graphic you want to be the source of the hyperlink .
```
