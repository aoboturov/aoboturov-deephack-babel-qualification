#!/bin/sh

python3 /aoboturov/OpenNMT-py/translate.py -model /aoboturov/model.pt -src /data/input.txt -output /output/output.txt -replace_unk -verbose
