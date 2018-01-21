FROM python:3.5-slim
MAINTAINER OBOTUROV Artem

RUN apt-get update && apt-get install -y --no-install-recommends \
	git g++ \
	&& rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip

RUN mkdir -p /aoboturov
WORKDIR /aoboturov/
COPY translate.sh /aoboturov/translate.sh

RUN git clone https://github.com/aoboturov/fairseq-py --depth 1

WORKDIR /aoboturov/fairseq-py
RUN pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl
RUN pip3 install -r requirements.txt
RUN python3 setup.py install

# playground will contain user defined scripts, it should be run as:
# docker run -v `pwd`:/data -it basel-baseline
RUN mkdir /data
RUN mkdir /output

# Add the model as the last part to enable re-use
COPY wmt14.en-de.fconv-py /aoboturov/model

CMD ["/aoboturov/translate.sh"]
