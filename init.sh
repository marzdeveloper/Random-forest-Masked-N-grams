#!/bin/bash

python3 top15_features.py "CLEAN" alexa-completo.txt > clean-completo.csv
python3 top15_features.py "MALWARE" dga-completo.txt > dga-completo.csv

Rscript run.R
