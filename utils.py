import itertools
import os
from PIL import Image
import cv2
from skimage import segmentation, filters, feature, morphology, transform, measure
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
import pandas as pd
import random

def normalize(image):
    """Normalize image to [0, 1]."""
    min = tf.reduce_min(image)
    max = tf.reduce_max(image)
    image = (image - min) / (max - min)
    return image

def crop_to_square(image):
    """Crops image to a square."""
    shape = image.shape
    if shape[0] > shape[1]:
        oy = (shape[0] - shape[1]) // 2
        image = tf.image.crop_to_bounding_box(image, oy, 0, shape[1], shape[1])
    else:
        ox = (shape[1] - shape[0]) // 2
        image = tf.image.crop_to_bounding_box(image, 0, ox, shape[0], shape[0])
    return image

def crop_out_background(image):
    """Crops out background from image using a simple heuristic.
    Args:
        image: Image to crop out background from.
    """
    def corners_uniform(image, bbox):
        f = bbox[0]
        tl = image[f, f]
        l = bbox[0] + bbox[1]
        o = bbox[1] // 4
        for x in [image[l, f], image[f, l], image[l, l],
            image[f, f + o], image[f + o, f], [l, l - o], image[l - o, l],
            [l, f + o], image[f + o, l], [l - o, f], image[f, l - o]]:
            if x[0] != tl[0] or x[1] != tl[1] or x[2] != tl[2]:
                return False
        return True
    bbox = [0, image.shape[0] - 1]
    for _ in range(10):
        if corners_uniform(image, bbox):
            bbox = [bbox[0] + 5, bbox[1] - 10]
        else:
            break
    image = tf.image.crop_to_bounding_box(image, bbox[0], bbox[0], bbox[1], bbox[1])
    return image

def unvignette(image, strength: float = 0.2, exp: int = 2):
    """Removes vignetting from image
    Args:
        image: Image to remove vignetting from.
        strength: Strength of vignetting to remove.
        exp: Exponent to apply to the vignetting mask.
    """
    dist = np.zeros(image.shape[:2], dtype=np.float32)
    center = np.array([image.shape[0] // 2, image.shape[1] // 2], dtype=np.float32)
    for x, y in itertools.product(range(image.shape[0]), range(image.shape[1])):
        dist[x, y] = (x - center[0]) ** exp + (y - center[1]) ** exp
    cv2.normalize(dist, dist, 1.0 - strength, 1.0, cv2.NORM_MINMAX)
    dist = np.expand_dims(dist, axis=2)
    dist = np.repeat(dist, 3, axis=2)
    image = image * dist
    return image

# standardize image by removing background and small artifacts and using normalized histogram to do simple binarization
# assuming skin lesions are usually darker than the healthy skin and larger than small artifacts
def load_image(path):
    """Loads and preprocesses images."""
    # The delta threshold for morphological operations e.g. removing moles and hair.
    morphology_threshold = 20
    # load image and clip to foreground square
    image = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
    image = crop_to_square(image)
    image = crop_out_background(image)
    image = tf.image.resize(image, (256, 256))
    # remove vignetting and adjust saturation
    image = tf.image.adjust_saturation(image, 1.25)
    image = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2XYZ)
    image = unvignette(image)
    # remove unimportant small dark objects from the image e.g. moles and hair
    img = np.stack([morphology.closing(image[:, :, 0], morphology.disk(3)),
                    morphology.closing(image[:, :, 1], morphology.disk(3)),
                    morphology.closing(image[:, :, 2], morphology.disk(3))], axis=-1)
    img = filters.gaussian(img, sigma=1)
    image = np.where(img - morphology_threshold > image, img, image)
    # remove unimportant small bright objects from the image e.g. artifacts and glare
    img = np.stack([morphology.opening(image[:, :, 0], morphology.disk(3)),
                    morphology.opening(image[:, :, 1], morphology.disk(3)),
                    morphology.opening(image[:, :, 2], morphology.disk(3))], axis=-1)
    img = filters.gaussian(img, sigma=1)
    image = np.where(img + morphology_threshold < image, img, image).astype(np.uint8)
    # simple segmentation using normalized histograms and otsu thresholding
    blurred = cv2.GaussianBlur(image, (5, 5), 5)
    er = cv2.equalizeHist(blurred[:, :, 0]).astype(np.uint8)
    eg = cv2.equalizeHist(blurred[:, :, 1]).astype(np.uint8)
    eb = cv2.equalizeHist(blurred[:, :, 2]).astype(np.uint8)
    thresh_r = filters.threshold_otsu(er)
    thresh_g = filters.threshold_otsu(eg)
    thresh_b = filters.threshold_otsu(eb)
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[er < thresh_r] = True
    mask[eg < thresh_g] = True
    mask[eb < thresh_b] = True
    # slightly more aggressive morphological operations to remove unimportant objects in the thresholded mask
    mask = morphology.binary_opening(mask, morphology.disk(5))
    mask = morphology.binary_closing(mask, morphology.disk(5))
    mask = np.repeat(np.expand_dims(mask, axis=2), 3, axis=-1)
    # use 255 as background color to avoid possible issues with very dark objects of interest and normalization
    image = np.where(mask, image, blurred)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    glcm = feature.graycomatrix(image, [1, 3], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256, symmetric=True)
    glcm = np.mean(glcm, axis=(-1, -2))
    return glcm.astype(np.uint8)
