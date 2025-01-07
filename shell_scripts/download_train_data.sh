#!/bin/bash

wget https://huggingface.co/datasets/hamishivi/lsds_data/resolve/main/training_data.zip
unzip training_data.zip
mkdir -p data/training_data
mv training_data/ data/training_data/
