FROM kwakinalabs/marian
MAINTAINER OBOTUROV Artem

COPY ende.bpe /aoboturov
COPY model.npz /aoboturov
COPY model.npz.json /aoboturov
COPY truecase-model.en /aoboturov
COPY vocab.en.json /aoboturov
COPY vocab.de.json /aoboturov

COPY translate.sh /aoboturov

CMD ["/aoboturov/translate.sh"]
