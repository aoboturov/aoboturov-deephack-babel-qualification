# Start with CUDA Torch dependencies 2
FROM kaixhin/torch:latest
MAINTAINER OBOTUROV Artem

# playground will contain user defined scripts, it should be run as:
# docker run -v `pwd`:/data -it basel-baseline
RUN mkdir /aoboturov && mkdir /data && mkdir /output && luarocks install tds && luarocks install bit32
COPY translate.sh /aoboturov/translate.sh

WORKDIR /aoboturov/
RUN git clone https://github.com/OpenNMT/OpenNMT
WORKDIR /aoboturov/OpenNMT

# Add the model as the last part to enable re-use
COPY standard_full_model_epoch9_3.65_release.t7 /aoboturov/model.t7

CMD ["/aoboturov/translate.sh"]
