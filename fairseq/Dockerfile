# Start with CUDA Torch dependencies 2
FROM kaixhin/torch:latest
MAINTAINER OBOTUROV Artem

# playground will contain user defined scripts, it should be run as:
# docker run -v `pwd`:/data -it basel-baseline
RUN mkdir /aoboturov && mkdir /data && mkdir /output
COPY translate.sh /aoboturov/translate.sh

WORKDIR /aoboturov/
RUN git clone https://github.com/facebookresearch/fairseq
WORKDIR /aoboturov/fairseq

RUN luarocks make rocks/fairseq-cpu-scm-1.rockspec

# Add the model as the last part to enable re-use
COPY standard_full_model_epoch9_3.65_release.t7 /aoboturov/model.t7

CMD ["/aoboturov/translate.sh"]
