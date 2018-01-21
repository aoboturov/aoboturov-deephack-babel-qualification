#!/bin/sh

fairseq generate -sourcelang en -targetlang de -beam 10 -nbest 2 -datadir /aoboturov/model/ -path /aoboturov/model/model.th7
