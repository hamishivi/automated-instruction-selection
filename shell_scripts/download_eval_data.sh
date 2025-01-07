#!/bin/bash

wget https://huggingface.co/datasets/hamishivi/lsds_data/resolve/main/eval.zip
unzip eval.zip
mkdir -p data
mv eval/ data/eval/
