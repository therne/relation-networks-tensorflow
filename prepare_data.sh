#!/bin/sh
mkdir -p data
wget -O CLEVR.zip https://s3-us-west-1.amazonaws.com/clevr/CLEVR_v1.0.zip
unzip -d data/ CLEVR.zip
rm CLEVR.zip

./process_clevr_data.py --split train
./process_clevr_data.py --split val