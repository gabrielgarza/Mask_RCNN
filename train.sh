#!/bin/bash

# Download data, create train and val sets
# echo Download and setup data...
# ./setup_project_and_data.sh
# echo Finished data setup

# Train
# Weights directory $1 ../../logs/ship20180815T0023
# Get last weights file i.e. mask_rcnn_ship_0067.h5
last_weights = ls $1 | tail -n 1
weights_path = "$1/$last_weights"
echo Training, staring with weights $last_weights
python3 ship.py train --dataset=./datasets --weights=$weights_path
echo Finished training

# Upload weights to s3
trained_weights = ls $1 | tail -n 1
trained_weights_path = "$1/$last_weights"
echo Uploading to s3...
aws s3 cp $trained_weights_path s3://airbus-kaggle
echo Uploaded trained weights to s3
