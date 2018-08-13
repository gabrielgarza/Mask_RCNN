import os
import random
import pandas as pd
import numpy as np

train_ship_segmentations_df = pd.read_csv(os.path.join("./datasets/train_val/train_ship_segmentations.csv"))

msk = np.random.rand(len(train_ship_segmentations_df)) < 0.9

train = train_ship_segmentations_df[msk]
test = train_ship_segmentations_df[~msk]

print("Total", train_ship_segmentations_df.shape)
print("Train",len(train))
print("Validation",len(test))

#  Move train set
for index, row in train.iterrows():
    image_id = row["ImageId"]
    old_path = "./datasets/train_val/{}".format(image_id)
    new_path = "./datasets/train/{}".format(image_id)
    if os.path.isfile(old_path):
        os.rename(old_path, new_path)

# Move val set
for index, row in test.iterrows():
    image_id = row["ImageId"]
    old_path = "./datasets/train_val/{}".format(image_id)
    new_path = "./datasets/val/{}".format(image_id)
    if os.path.isfile(old_path):
        os.rename(old_path, new_path)

# Count files
path, dirs, files = next(os.walk("./datasets/train_val"))
file_count = len(files)
print("files in train_val:", file_count)

path, dirs, files = next(os.walk("./datasets/train"))
file_count = len(files)
print("files in train:", file_count)


path, dirs, files = next(os.walk("./datasets/val"))
file_count = len(files)
print("files in val:", file_count)

#  Save new csv files
train = train.dropna()
test = test.dropna()

train.to_csv("./datasets/train/train_ship_segmentations.csv", index=False)
test.to_csv("./datasets/val/val_ship_segmentations.csv", index=False)

# Verify new vsc files
train_segs = pd.read_csv(os.path.join("./datasets/train/train_ship_segmentations.csv"))
val_segs = pd.read_csv(os.path.join("./datasets/val/val_ship_segmentations.csv"))


print("train_segs", train_segs.shape)
print("val_segs", val_segs.shape)
