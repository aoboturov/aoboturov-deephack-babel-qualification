FROM python:2.7-slim
MAINTAINER OBOTUROV Artem

RUN apt-get update && apt-get install -y --no-install-recommends \
	git g++ \
	&& rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install numpy numexpr cython theano tables bottle bottle-log tornado

RUN mkdir -p /aoboturov/model
WORKDIR /aoboturov/

RUN git clone https://github.com/moses-smt/mosesdecoder --depth 1
RUN git clone https://github.com/rsennrich/subword-nmt --depth 1
RUN git clone https://github.com/rsennrich/nematus --depth 1

WORKDIR /aoboturov/nematus
RUN python setup.py install

# playground will contain user defined scripts, it should be run as:
# docker run -v `pwd`:/data -it basel-baseline
RUN mkdir /data
RUN mkdir /output

# Add the model as the last part to enable re-use
COPY vocab.en.json /aoboturov/model
COPY vocab.de.json /aoboturov/model
COPY ende.bpe /aoboturov/model

COPY translate.sh /aoboturov

COPY truecase-model.en /aoboturov/model
COPY model.npz /aoboturov/model
COPY model.npz.json /aoboturov/model

WORKDIR /aoboturov/model

CMD ["/aoboturov/translate.sh"]
