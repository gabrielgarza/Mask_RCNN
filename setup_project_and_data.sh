#!/bin/bash

# Remember to first configure awscli with credentials before running this file
set -e

# Clone repo
echo Cloning repo...
git config --global credential.helper '!aws codecommit credential-helper $@'
git config --global credential.UseHttpPath true
git clone https://git-codecommit.us-east-1.amazonaws.com/v1/repos/Mask_RCNN

# Download datasets from s3
echo Downloading dataset...
mkdir -p ./Mask_RCNN/samples/ship/datasets
mkdir -p ./Mask_RCNN/samples/ship/datasets/train_val
mkdir -p ./Mask_RCNN/samples/ship/datasets/train
mkdir -p ./Mask_RCNN/samples/ship/datasets/val
aws s3 cp s3://airbus-kaggle/train.zip ./Mask_RCNN/samples/ship/datasets/train_val/
aws s3 cp s3://airbus-kaggle/train_ship_segmentations.csv.zip ./Mask_RCNN/samples/ship/datasets/train_val/
echo Unziping train_ship_segmentations...
unzip ./Mask_RCNN/samples/ship/datasets/train_val/train_ship_segmentations.csv.zip -d ./Mask_RCNN/samples/ship/datasets/train_val/
echo Unziping train...
unzip -q ./Mask_RCNN/samples/ship/datasets/train_val/train.zip -d ./Mask_RCNN/samples/ship/datasets/train_val/


# Split dataset into train and val folders
echo Splitting train and val sets...
python3 ./Mask_RCNN/samples/ship/split_train_val.py


mkdir -p ./Mask_RCNN/logs/weights/
KEY=`aws s3 ls s3://airbus-kaggle/weights --recursive | sort | tail -n 1 | awk '{print $4}'`
echo "Downloading weights... $KEY"
aws s3 cp s3://airbus-kaggle/$KEY ./Mask_RCNN/logs/weights/

echo Done with setup!

# Log into ecr and pull image
# $(aws ecr get-login --no-include-email --region us-east-1)
# docker pull 001413338534.dkr.ecr.us-east-1.amazonaws.com/deep-learning-gpu:latest

# Command to run docker here for convenience as a comment
# docker run -it -p 8888:8888 -p 6006:6006 -v ~/.:/host 001413338534.dkr.ecr.us-east-1.amazonaws.com/deep-learning-gpu
# docker run 001413338534.dkr.ecr.us-east-1.amazonaws.com/deep-learning-gpu python train.sh
