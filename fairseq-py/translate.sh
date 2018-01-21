#!/bin/sh

python3 /aoboturov/fairseq-py/interactive.py --cpu --path /aoboturov/model/model.pt --output-file /output/output.txt -s en -t de /aoboturov/model < /data/input.txt
