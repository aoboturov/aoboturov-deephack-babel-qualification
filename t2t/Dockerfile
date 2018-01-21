FROM python:3.6-slim-stretch
MAINTAINER OBOTUROV Artem

#RUN apt-get update \
#	&& rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install -q 'tensor2tensor==1.4.1' 'tensorflow==1.5.0rc0' numpy

RUN mkdir -p /aoboturov/model
WORKDIR /aoboturov/

# playground will contain user defined scripts, it should be run as:
# docker run -v `pwd`:/data -it basel-baseline
RUN mkdir /data
RUN mkdir /output

# Add the model as the last part to enable re-use
COPY runner.py /aoboturov/model
COPY translate.sh /aoboturov/model

WORKDIR /aoboturov/model

COPY transformer_ende_test/* /aoboturov/model/

CMD ["/aoboturov/model/translate.sh"]
