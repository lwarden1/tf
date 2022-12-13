# Description: This script creates a model to classify images as benign or malignant using the EfficientNetV2 architecture
# @author: Lance Warden (lwarden1@trinity.edu)
import numpy as np
import keras
from keras import utils, layers, models, optimizers, losses, callbacks, backend as K, regularizers
import keras.preprocessing.image as image
from keras.preprocessing.image import ImageDataGenerator
import tensorflow_io as tfio
import keras.applications as applications
import keras.applications.efficientnet_v2 as efficientnet_v2
import pandas as pd
import tensorflow as tf
# parameters
n_epochs = 20
# there are only ~581 malignant images of the ~30_000, so we need to oversample
validation_split = 0.2
# create model
efficientnet = efficientnet_v2.EfficientNetV2B0(include_top=False, input_shape=(256, 256, 3))
efficientnet.trainable = True
model = models.Sequential()
model.add(efficientnet)
model.add(layers.Conv2D(512, kernel_size=5, activation='relu', kernel_regularizer='l2', input_shape=(256, 256, 3), padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(layers.Conv2D(256, kernel_size=3, activation='relu', kernel_regularizer='l2', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(layers.Conv2D(128, kernel_size=3, activation='relu', kernel_regularizer='l2', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(layers.Conv2D(64, kernel_size=3, activation='relu', kernel_regularizer='l2', padding='same'))
model.add(layers.Flatten())
model.add(layers.Dropout(0.1))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
              loss=losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()
# load data
sample = pd.read_csv("dataset/sample.csv")
train_images = [tf.image.decode_jpeg(tf.io.read_file(path), channels=3) for path in "dataset/sample/" + sample['image_name'] + ".jpg"]
train_labels = sample['target'].values
# train model
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=3)
model_checkpoint = callbacks.ModelCheckpoint(
    'model.h5', monitor='val_loss', save_best_only=True)
history = model.fit(np.array(train_images), train_labels, epochs=n_epochs, validation_split=validation_split,
                    batch_size=32, callbacks=[early_stopping, reduce_lr, model_checkpoint], shuffle=True)
model.save('model.h5')

# test_meta = pd.read_csv("dataset/ISIC_2020_Test_Metadata.csv")
# test_sample = test_meta.sample(10)
# test_images = [load_image(path) for path in "dataset/test/" + test_sample['image'] + ".jpg"]
# test_labels = test_sample['target'].values
# predict = model.predict(np.array(test_images)[:10])
# print(predict)
# # actual results for first 4 images in test set
# print(test_labels[:10])
