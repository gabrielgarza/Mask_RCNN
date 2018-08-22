#!/bin/bash
set -e

echo Starting inference...
python3 generate_predictions.py
echo Finished inference

# Upload weights to s3
trained_weights = ls ./submissions/ | tail -n 1
trained_weights_path = "./submissions/$last_weights"
echo "Uploading to s3 $trained_weights_path..."
aws s3 cp $trained_weights_path s3://airbus-kaggle
echo Uploaded trained weights to s3
