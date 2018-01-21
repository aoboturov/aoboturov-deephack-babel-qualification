FROM python:3.6-slim-stretch
MAINTAINER OBOTUROV Artem

RUN apt-get update && apt-get install -y --no-install-recommends \
	git libgfortran3 libgomp1 \
	&& rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip pyyaml 'mxnet==1.0.0' 'numpy>=1.12' typing sockeye

# playground will contain user defined scripts, it should be run as:
# docker run -v `pwd`:/data -it basel-baseline
RUN mkdir -p /data /output /aoboturov/model

WORKDIR /aoboturov/
RUN git clone https://github.com/moses-smt/mosesdecoder --depth 1

# Add the model as the last part to enable re-use
COPY args.json /aoboturov/model
COPY symbol.json /aoboturov/model
COPY vocab.src.json /aoboturov/model
COPY vocab.trg.json /aoboturov/model
COPY version /aoboturov/model
COPY config /aoboturov/model

COPY translate.sh /aoboturov

COPY params.best /aoboturov/model

CMD ["/aoboturov/translate.sh"]
