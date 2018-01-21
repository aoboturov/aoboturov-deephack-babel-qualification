FROM python:3.6-slim-stretch
MAINTAINER OBOTUROV Artem

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
	git libstdc++6 \
	&& rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip

RUN mkdir -p /aoboturov
WORKDIR /aoboturov/

RUN git clone https://github.com/tensorflow/nmt --depth 1

WORKDIR /aoboturov/nmt
RUN pip3 install tensorflow==1.5.0rc0

# playground will contain user defined scripts, it should be run as:
# docker run -v `pwd`:/data -it basel-baseline
RUN mkdir /data
RUN mkdir /output

# Add the model as the last part to enable re-use
COPY vocab.bpe.32000 /aoboturov/model/vocab.bpe.32000.de
COPY vocab.bpe.32000 /aoboturov/model/vocab.bpe.32000.en
COPY train.tok.bpe.32000.de /aoboturov/model
COPY translate.sh /aoboturov/translate.sh
COPY ende_gnmt_model_8_layer/* /aoboturov/model/
COPY hparams /aoboturov/model

CMD ["/aoboturov/translate.sh"]
