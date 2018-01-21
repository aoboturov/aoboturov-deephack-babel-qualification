FROM python:3.6-slim-stretch
MAINTAINER OBOTUROV Artem

RUN apt-get update && apt-get install -y --no-install-recommends \
	git tk-dev \
	&& rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip

RUN mkdir -p /aoboturov/model
WORKDIR /aoboturov/

RUN git clone https://github.com/aoboturov/seq2seq --depth 1

WORKDIR /aoboturov/seq2seq
RUN pip3 install -e .

# playground will contain user defined scripts, it should be run as:
# docker run -v `pwd`:/data -it basel-baseline
RUN mkdir /data
RUN mkdir /output

# Add the model as the last part to enable re-use
COPY vocab.bpe.32000 /aoboturov/model
COPY train_options.json /aoboturov/model
COPY tasks.yaml /aoboturov/model
COPY translate.sh /aoboturov/translate.sh
COPY model/* /aoboturov/model/

CMD ["/aoboturov/translate.sh"]
