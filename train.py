# Description: This script creates a model to classify images as benign or malignant using the EfficientNetV2 architecture
# @author: Lance Warden (lwarden1@trinity.edu)
import numpy as np
import keras
from keras import utils, layers, models, optimizers, losses, callbacks, backend as K, regularizers, metrics
import keras.preprocessing.image as image
from keras.preprocessing.image import ImageDataGenerator
import tensorflow_io as tfio
import keras.applications as applications
import keras.applications.efficientnet_v2 as efficientnet_v2
import pandas as pd
import tensorflow as tf
# parameters
sample_path = "dataset/sample-blur.tfrecord"
n_epochs = 30
batch_size = 64
# there are only ~581 malignant images of the ~30_000, so we need to oversample
validation_split = 0.2

try:
    ds = tf.data.Dataset.load(sample_path, compression="GZIP", element_spec=(tf.TensorSpec(shape=(256, 256, 3), dtype=tf.uint8, name=None), tf.TensorSpec(shape=(), dtype=tf.int64, name=None)))
except:
    ds = tf.data.experimental.load(sample_path, compression="GZIP", element_spec=(tf.TensorSpec(shape=(256, 256, 3), dtype=tf.uint8, name=None), tf.TensorSpec(shape=(), dtype=tf.int64, name=None)))
print(f"Loaded dataset {ds}")
n_val = int(validation_split * len(ds))
print(f"Training on {len(ds) - n_val} samples, validating on {n_val} samples")
val_ds = ds.take(n_val).batch(batch_size)
ds = ds.skip(n_val).batch(batch_size)


# for d in ds.take(1):
#     print(d)
## create model
efficientnet = efficientnet_v2.EfficientNetV2B0(include_top=False, input_shape=(256, 256, 3), weights='imagenet')
efficientnet.trainable = True
model = models.Sequential()
model.add(efficientnet)
model.add(layers.GlobalMaxPooling2D())
model.add(layers.Dense(3, activation='relu', name='dense_14'))
model.add(layers.Dense(1, activation='softmax', name='dense_final'))
optimizer=optimizers.Adam(learning_rate=1e-19, clipnorm=1.0, amsgrad=True)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()
## train model
cbs = [
    callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3),
    callbacks.ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True)
]
# with tf.device('/GPU:1'):
history = model.fit(ds, epochs=n_epochs, validation_data=val_ds, batch_size=batch_size, callbacks=cbs, shuffle=True, class_weight={0: 1, 1: 5})
model.save('model.h5')
