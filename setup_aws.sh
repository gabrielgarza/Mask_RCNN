#!/bin/bash

# Remember to first configure awscli with credentials before running this file

# Clone repo
git config --global credential.helper '!aws codecommit credential-helper $@'
git config --global credential.UseHttpPath true
git clone https://git-codecommit.us-east-1.amazonaws.com/v1/repos/Mask_RCNN

# Download datasets from s3
mkdir ./samples/ship/datasets
mkdir ./samples/ship/datasets/train_val
mkdir ./samples/ship/datasets/train
mkdir ./samples/ship/datasets/val
aws s3 cp s3://airbus-kaggle/train.zip ./samples/ship/datasets/train_val/
aws s3 cp s3://airbus-kaggle/test.zip ./samples/ship/datasets/train_val/
aws s3 cp s3://airbus-kaggle/train_ship_segmentations.csv.zip ./samples/ship/datasets/train_val/
aws s3 cp s3://airbus-kaggle/sample_submission.csv.zip ./samples/ship/datasets/train_val/
unzip ./samples/ship/datasets/train_val/train_ship_segmentations.csv.zip -d ./samples/ship/datasets/train_val/
unzip -q ./samples/ship/datasets/train_val/train.zip -d ./samples/ship/datasets/train_val/

# Split dataset into train and val folders
python3 ./samples/ship/split_train_val.py

# Log into ecr and pull image
$(aws ecr get-login --no-include-email --region us-east-1)
docker pull 001413338534.dkr.ecr.us-east-1.amazonaws.com/deep-learning-gpu:latest

# Command to run docker here for convenience as a comment
# docker run -it -p 8888:8888 -p 6006:6006 -v ~/.:/host 001413338534.dkr.ecr.us-east-1.amazonaws.com/deep-learning-gpu
