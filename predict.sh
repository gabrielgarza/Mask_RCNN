#!/bin/bash
set -e

# Clone repo
echo Cloning repo...
git config --global credential.helper '!aws codecommit credential-helper $@'
git config --global credential.UseHttpPath true
git clone https://git-codecommit.us-east-1.amazonaws.com/v1/repos/Mask_RCNN

# Download sample submission csv
mkdir -p ./Mask_RCNN/samples/ship/datasets/test
echo Downloading sample_submission...
aws s3 cp s3://airbus-kaggle/sample_submission.csv ./Mask_RCNN/samples/ship/datasets/test/

# Download test set
echo Downloading test.zip...
aws s3 cp s3://airbus-kaggle/test.zip ./Mask_RCNN/samples/ship/datasets/test/

# Unzip sample submission csv
# echo Unziping sample_submission...
# unzip -q ./Mask_RCNN/samples/ship/datasets/test/sample_submission.csv.zip -d ./Mask_RCNN/samples/ship/datasets/test/

# Unzip test
echo Unziping test...
unzip -q ./Mask_RCNN/samples/ship/datasets/test/test.zip -d ./Mask_RCNN/samples/ship/datasets/test/

# Gets the latest folder and make the same directory locally
# ship20180823T0000/
LATEST_DIR=`aws s3 ls s3://airbus-kaggle/logs/ | grep / | sort | tail -n 1 | awk '{print $2}'`
mkdir -p ./Mask_RCNN/logs/$LATEST_DIR

# Gets the latest file from the latest directory in weights
# logs/ship20180823T0000/mask_rcnn_ship_0001.h5
KEY=`aws s3 ls s3://airbus-kaggle/logs/ --recursive | sort | tail -n 1 | awk '{print $4}'`
echo "Downloading weights... $KEY"
aws s3 cp s3://airbus-kaggle/$KEY ./Mask_RCNN/$KEY

# Create submissions folder
mkdir -p ./Mask_RCNN/samples/ship/submissions
echo Created submissions folder

# Change directory
cd Mask_RCNN/samples/ship/

echo Starting inference...
python3 ./generate_predictions.py --weights=./$KEY
echo Finished inference

# Upload submission to s3
echo "Uploading submission to s3..."
aws s3 cp ./submissions/ s3://airbus-kaggle/submissions/ --recursive
echo Uploaded submission to s3

# sudo docker run -it 001413338534.dkr.ecr.us-east-1.amazonaws.com/deep-learning-gpu bash ./predict.sh
