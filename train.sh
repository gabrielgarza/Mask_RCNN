#!/bin/bash
set -e

# Download data, create train and val sets
echo Download and setup data...
chmod 755 ./setup_project_and_data.sh
./setup_project_and_data.sh
echo Finished data setup

# Train
# Weights directory $1 ../../logs/ship20180815T0023
# Get last weights file i.e. mask_rcnn_ship_0067.h5
last_weights = ls ./Mask_RCNN/logs/weights/ | tail -n 1
weights_path = "./Mask_RCNN/logs/weights/$last_weights"
echo Training, staring with weights $last_weights
python3 ./Mask_RCNN/samples/ship/ship.py train --dataset=./datasets --weights=$weights_path
echo Finished training

# Upload weights to s3
trained_weights = ls ./Mask_RCNN/logs/weights/ | tail -n 1
trained_weights_path = "./Mask_RCNN/logs/weights/$last_weights"
echo Uploading to s3...
aws s3 cp $trained_weights_path s3://airbus-kaggle/weights
echo Uploaded trained weights to s3
