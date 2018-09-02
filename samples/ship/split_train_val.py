import os
import random
import pandas as pd
import numpy as np

train_ship_segmentations_df = pd.read_csv(os.path.join("./datasets/train_val/train_ship_segmentations.csv"))
# Remove corrupted image
train_ship_segmentations_df = train_ship_segmentations_df.loc[train_ship_segmentations_df["ImageId"] != "6384c3e78.jpg"]

# Undersample empty images
# Used to remove all of them -> train_ship_segmentations_df = train_ship_segmentations_df.dropna()
train_ship_segmentations_df_null = train_ship_segmentations_df["EncodedPixels"].isnull()
nulls_df = train_ship_segmentations_df[train_ship_segmentations_df_null]
nulls_sample_df = nulls_df.sample(frac=0.5) # remove frac % of empty images
train_ship_segmentations_df = train_ship_segmentations_df.loc[~train_ship_segmentations_df["ImageId"].isin(nulls_sample_df["ImageId"])].sample(frac=0.5)


# Select 90% random rows for train set
msk = np.random.rand(len(train_ship_segmentations_df)) < 0.9

# Split train and val sets
train = train_ship_segmentations_df[msk]
val = train_ship_segmentations_df[~msk]

print("Total", train_ship_segmentations_df.shape)
print("Train",len(train))
print("Validation",len(val))

#  Move train set
for index, row in train.iterrows():
    image_id = row["ImageId"]
    old_path = "./datasets/train_val/{}".format(image_id)
    new_path = "./datasets/train/{}".format(image_id)
    if os.path.isfile(old_path):
        os.rename(old_path, new_path)

# Move val set
for index, row in val.iterrows():
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
train.to_csv("./datasets/train/train_ship_segmentations.csv", index=False)
val.to_csv("./datasets/val/val_ship_segmentations.csv", index=False)

# Verify new vsc files
train_segs = pd.read_csv(os.path.join("./datasets/train/train_ship_segmentations.csv"))
val_segs = pd.read_csv(os.path.join("./datasets/val/val_ship_segmentations.csv"))


print("train_segs", train_segs.shape)
print("val_segs", val_segs.shape)
