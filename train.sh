#!/bin/bash
set -e

# Download data, create train and val sets
echo Download and setup data...
chmod 755 ./setup_project_and_data.sh
./setup_project_and_data.sh
echo Finished data setup

# Train
cd ./Mask_RCNN/samples/ship

# Weights directory $1 ../../logs/ship20180815T0023
# Get last weights file i.e. mask_rcnn_ship_0067.h5
last_directory=`ls ../../logs/ | tail -n 1`
last_weights=`ls ../../logs/$last_directory/ | tail -n 1`
last_weights_path="../../logs/$last_directory/$last_weights"
echo Training, staring with weights $last_weights_path

python3 ./ship.py train --dataset=./datasets --weights=last
echo Finished training

# Upload weights to s3
trained_directory=`ls ../../logs/ | tail -n 1`
trained_weights=`ls ../../logs/$trained_directory/ | tail -n 1`
trained_weights_path="../../logs/$trained_directory/$trained_weights"

echo Uploading $trained_weights_path to s3...
aws s3 cp $trained_weights_path s3://airbus-kaggle/logs/$trained_directory/
echo Uploaded trained weights to s3

# sudo $(aws ecr get-login --no-include-email --region us-east-1)
# sudo docker pull 001413338534.dkr.ecr.us-east-1.amazonaws.com/deep-learning-gpu:latest
# sudo docker run -it 001413338534.dkr.ecr.us-east-1.amazonaws.com/deep-learning-gpu bash ./train.sh
