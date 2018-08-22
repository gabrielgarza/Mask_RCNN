#!/bin/bash
set -e


mkdir -p ./Mask_RCNN/samples/ship/datasets/test
echo Downloading sample_submission...
aws s3 cp s3://airbus-kaggle/sample_submission.csv.zip ./Mask_RCNN/samples/ship/datasets/test/

echo Downloading test.zip...
aws s3 cp s3://airbus-kaggle/test.zip ./Mask_RCNN/samples/ship/datasets/test/

echo Unziping sample_submission...
unzip -q ./Mask_RCNN/samples/ship/datasets/test/sample_submission.csv.zip -d ./Mask_RCNN/samples/ship/datasets/test/

echo Unziping test...
unzip -q ./Mask_RCNN/samples/ship/datasets/test/test.zip -d ./Mask_RCNN/samples/ship/datasets/test/

mkdir -p ./Mask_RCNN/logs/weights/
KEY=`aws s3 ls s3://airbus-kaggle/weights --recursive | sort | tail -n 1 | awk '{print $4}'`
echo "Downloading weights... $KEY"
aws s3 cp s3://airbus-kaggle/$KEY ./Mask_RCNN/logs/weights/

# Create submissions folder
mkdir -p ./Mask_RCNN/samples/ship/submissions
echo Created submissions folder

echo Starting inference...
python3 ./Mask_RCNN/samples/ship/generate_predictions.py
echo Finished inference

# Upload submission to s3
echo "Uploading submission to s3..."
aws s3 cp ./Mask_RCNN/samples/ship/submissions  s3://airbus-kaggle/submissions --recursive
echo Uploaded submission to s3

# sudo docker run -it 001413338534.dkr.ecr.us-east-1.amazonaws.com/deep-learning-gpu bash ./predict.sh
