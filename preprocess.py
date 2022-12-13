# Description: This script creates a sample of the dataset to be used for training and testing
# @author: Lance Warden (lwarden1@trinity.edu)
import pandas as pd
from utils import load_image
import tensorflow as tf
# there are only 581 malignant images and we want somewhere around 0.1 dropout to avoid overfitting on bad samples
n_data = 2000
print(f"Sampling {n_data}")
n_benign = int(n_data * 0.8)
n_malignant = n_data - n_benign
metadata = pd.read_csv("dataset/ISIC_2020_Training_GroundTruth_v2.csv")
dupes = pd.read_csv("dataset/ISIC_2020_Training_Duplicates.csv")[
    "image_name_2"].values
metadata = metadata[~metadata["image_name"].isin(dupes)]
metadata.reset_index(drop=True, inplace=True)
sample = pd.concat([metadata[metadata["target"] == 0].sample(
    n_benign), metadata[metadata["target"] == 1].sample(n_malignant)])
sample.reset_index(drop=True, inplace=True)
sample = sample.sample(frac=1)
sample.reset_index(drop=True, inplace=True)
# save dataset
ds = tf.data.Dataset.from_tensor_slices((sample["image_name"].values, sample["target"].values))
ds.map(lambda x, y: (tf.image.encode_jpeg(tf.image.convert_image_dtype(load_image("dataset/train/" + x + ".jpg"), tf.uint8), quality=100, chroma_downsampling=False, format="grayscale"), y))
ds.save("dataset/sample.tfrecord", compression="GZIP")
