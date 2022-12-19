import random
from utils import load_image
from keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
n_test = 1000
n_benign = random.randint(int(n_test * 0.6), int(n_test * 0.8))
n_malignant = n_test - n_benign
model = load_model("model.h5")
if model is None:
    print("Model not found")
    exit()
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
lbls = sample['target'].values
preds = model.predict(np.array(imgs))
# show accuracy
preds = np.sum(preds, axis=1)
for x in range(0,10):
    x = x / 10
    p = np.where(preds > x, 1, 0)
    print(np.mean(p == lbls))
