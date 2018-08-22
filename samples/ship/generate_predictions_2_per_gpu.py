import pandas
import math
import re
import datetime
import time
import numpy as np
import os
import sys
import skimage.io
import pandas as pd
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.ship import ship

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights
SHIP_WEIGHTS_PATH = "./logs/ship20180815T0023/mask_rcnn_ship_0067.h5"

# Config
config = ship.ShipConfig()
SHIP_DIR = os.path.join(ROOT_DIR, "/samples/ship/datasets")

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    DETECTION_MIN_CONFIDENCE = 0.95
    DETECTION_NMS_THRESHOLD = 0.0

# Create model object in inference mode.
config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Instantiate dataset
dataset = ship.ShipDataset()

# Load weights
model.load_weights(os.path.join(ROOT_DIR, SHIP_WEIGHTS_PATH), by_name=True)

class_names = ['BG', 'ship']

# Run detection
# Load image ids (filenames) and run length encoded pixels
images_path = "datasets/test"
sample_sub_csv = "sample_submission.csv"
# images_path = "datasets/val"
# sample_sub_csv = "val_ship_segmentations.csv"
sample_submission_df = pd.read_csv(os.path.join(images_path,sample_sub_csv))
unique_image_ids = sample_submission_df.ImageId.unique()



out_pred_rows = []
count = 0
for image_id_1, image_id_2 in zip(unique_image_ids[0::2], unique_image_ids[1::2]):
    image_path_1 = os.path.join(images_path, image_id_1)
    image_path_2 = os.path.join(images_path, image_id_2)
    if os.path.isfile(image_path_1) and os.path.isfile(image_path_2):
        count += 1
        print("Step: ", count)

        # Start counting prediction time
        tic = time.clock()

        image_1 = skimage.io.imread(image_path_1)
        image_2 = skimage.io.imread(image_path_2)
        results = model.detect([image_1, image_2], verbose=1)
        r0 = results[0]
        r1 = results[1]

        # First Image
        re_encoded_to_rle_list = []
        for i in np.arange(np.array(r0['masks']).shape[-1]):
            boolean_mask = r0['masks'][:,:,i]
            re_encoded_to_rle = dataset.rle_encode(boolean_mask)
            re_encoded_to_rle_list.append(re_encoded_to_rle)

        if len(re_encoded_to_rle_list) == 0:
            out_pred_rows += [{'ImageId': image_id_1, 'EncodedPixels': None}]
        else:
            for rle_mask in re_encoded_to_rle_list:
                out_pred_rows += [{'ImageId': image_id_1, 'EncodedPixels': rle_mask}]

        # Second Image
        re_encoded_to_rle_list = []
        for i in np.arange(np.array(r1['masks']).shape[-1]):
            boolean_mask = r1['masks'][:,:,i]
            re_encoded_to_rle = dataset.rle_encode(boolean_mask)
            re_encoded_to_rle_list.append(re_encoded_to_rle)

        if len(re_encoded_to_rle_list) == 0:
            out_pred_rows += [{'ImageId': image_id_2, 'EncodedPixels': None}]
        else:
            for rle_mask in re_encoded_to_rle_list:
                out_pred_rows += [{'ImageId': image_id_2, 'EncodedPixels': rle_mask}]

        toc = time.clock()
        print("Prediction time: ",toc-tic)


submission_df = pd.DataFrame(out_pred_rows)[['ImageId', 'EncodedPixels']]

filename = "{}{:%Y%m%dT%H%M}.csv".format("./submissions/submission_", datetime.datetime.now())
submission_df.to_csv(filename, index=False)


print("Submission CSV Shape", submission_df.shape)

# print("ROIS",r['rois'])
# print("Masks",r['masks'])
# print("Masks Shape",np.array(r['masks']).shape)
# print("Class IDs",r['class_ids'])
# print("Scores",r['scores'])
