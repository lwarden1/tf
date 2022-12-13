import random
from utils import load_image
from keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
n_test = 5
n_benign = random.randint(int(n_test * 0.6), int(n_test * 0.8))
n_malignant = n_test - n_benign
meta = pd.read_csv("dataset/ISIC_2020_Training_GroundTruth_v2.csv")
dupes = pd.read_csv("dataset/ISIC_2020_Training_Duplicates.csv")["image_name_2"].values
meta = meta[~meta["image_name"].isin(dupes)]
meta.reset_index(drop=True, inplace=True)
# get random samples of benign and malignant
benign = meta[meta["target"] == 0]
benign = benign.sample(n_benign)
malignant = meta[meta["target"] == 1]
malignant = malignant.sample(n_malignant)
sample = pd.concat([benign, malignant])
sample.reset_index(drop=True, inplace=True)
sample = sample.sample(frac=1).reset_index(drop=True)
# load images and predict
imgs = [load_image(path) for path in "dataset/train/" + sample['image_name'] + ".jpg"]
# show accuracy
for img in imgs:
    plt.imshow(img)
    plt.show()
