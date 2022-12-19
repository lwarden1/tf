# Description: This script creates a sample of the dataset to be used for training and testing
# @author: Lance Warden (lwarden1@trinity.edu)
import pandas as pd
from utils import load_image
import tensorflow as tf
# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()
# there are only 581 malignant images and we want somewhere around 0.1 dropout to avoid overfitting on bad samples
sample_path = "dataset/sample-blur.tfrecord"
n_data = 2000
n_benign = n_data - 581
n_malignant = n_data - n_benign
print(f"Sampling {n_data} ({n_benign},{n_malignant})")
metadata = pd.read_csv("dataset/ISIC_2020_Training_GroundTruth_v2.csv")
dupes = pd.read_csv("dataset/ISIC_2020_Training_Duplicates.csv")["image_name_2"].values
metadata = metadata[~metadata["image_name"].isin(dupes)]
metadata.reset_index(drop=True, inplace=True)
sample = pd.concat([metadata[metadata["target"] == 0].sample(n_benign), metadata[metadata["target"] == 1].sample(n_malignant)])
sample.reset_index(drop=True, inplace=True)
sample = sample.sample(frac=1)
sample.reset_index(drop=True, inplace=True)
# save dataset
ds = tf.data.Dataset.from_tensor_slices((sample["image_name"].values, sample["target"].values))
ds = ds.map(lambda x, y: (load_image("dataset/train/" + x + ".jpg"), y), num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
try:
    ds.save(sample_path, compression="GZIP")
except:
    tf.data.experimental.save(ds, sample_path, compression="GZIP")
