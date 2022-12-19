import cv2
from skimage import segmentation, filters, feature, morphology, transform, measure
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
from keras import backend as K, layers
def normalize(src, To=(0,1), From=None):
    """Normalize range."""
    if From is None:
        From = (tf.reduce_min(src), tf.reduce_max(src))
    return (src - From[0]) * ((To[1] - To[0]) / (From[1] - From[0])) + To[0]

def crop_to_square(image):
    """Crops image to a square."""
    shape = tf.shape(image)
    if shape[0] > shape[1]:
        oy = (shape[0] - shape[1]) // 2
        image = tf.image.crop_to_bounding_box(image, oy, 0, shape[1], shape[1])
    else:
        ox = (shape[1] - shape[0]) // 2
        image = tf.image.crop_to_bounding_box(image, 0, ox, shape[0], shape[0])
    return image

def crop_out_background(image):
    """Crops out background from image using a simple heuristic."""
    shape = tf.shape(image)
    def corners_uniform(image, bbox):
        f = int(bbox[0])
        tl = tf.abs(image[f, f])
        tlr = 1e-6 * tl + 0.1
        l = int(bbox[0] + bbox[1])
        o = int(bbox[1] // 4)
        ret = True
        for x in [image[l, f], image[f, l], image[l, l],
            image[f, f + o], image[f + o, f], image[l, l - o], image[l - o, l],
            image[l, f + o], image[f + o, l], image[l - o, f], image[f, l - o]]:
            if not tf.math.reduce_all(tf.abs(x - tl) <= tlr):
                ret = False
        return ret
    bbox = [0, shape[0] - 1]
    for _ in range(10):
        if corners_uniform(image, bbox):
            bbox = [bbox[0] + 5, bbox[1] - 10]
        else:
            continue
    image = tf.image.crop_to_bounding_box(image, bbox[0], bbox[0], bbox[1], bbox[1])
    return image

def unvignette(image, strength: float = 0.2, exp: int = 2):
    """Removes vignetting from image
    Args:
        image: Image to remove vignetting from.
        strength: Strength of vignetting to remove.
        exp: Exponent to apply to the vignetting mask.
    """
    shape = tf.shape(image)
    row = tf.abs(tf.range(shape[1], dtype=tf.float32) - float(shape[1] // 2)) ** exp
    col = tf.abs(tf.range(shape[0], dtype=tf.float32) - float(shape[0] // 2)) ** exp
    dist = normalize(tf.broadcast_to(tf.expand_dims(row, axis=0), shape[:2]) + tf.expand_dims(col, axis=1), (1.0 - strength, 1.0))
    return image * tf.stack([dist, dist, dist], axis=-1)

def gaussian_filter(image: tf.Tensor, ksize: int = 3, sigma: float = 1.0, dtype: tf.dtypes.DType | None = None) -> tf.Tensor:
    """Applies gaussian filter to image.
    Args:
        image: Image to apply gaussian blur to.
        kernel_size: Size of the kernel to use.
        sigma: Sigma to use for gaussian blur.
    """
    ksize = abs(ksize)
    ksize = ksize + (1 - (ksize % 2))
    radius = ksize // 2
    r = tf.exp(-((tf.range(-radius, radius + 1, dtype=tf.float32) ** 2) / (2 * (sigma ** 2))))
    r = r / tf.reduce_sum(r)
    r = tf.matmul(tf.expand_dims(r, axis=1), tf.expand_dims(r, axis=0))
    r = tf.reshape(r, [ksize, ksize, 1, 1])
    r = tf.tile(r, [1, 1, image.shape[-1], 1])
    image = tf.pad(image, [[0,0], [radius, radius], [radius, radius], [0, 0]], 'REFLECT')
    return tf.nn.depthwise_conv2d(image, r, [1,1,1,1], 'VALID')

def disk(radius: int) -> tf.Tensor:
    """Returns a disk shaped boolean mask of radius r."""
    diameter = radius * 2 + 1
    x = tf.range(-radius, radius + 1, dtype=tf.int32) ** 2
    x = tf.broadcast_to(tf.expand_dims(x, axis=0), [diameter, diameter]) + tf.expand_dims(x, axis=1)
    return (x <= radius ** 2)

def dilation(image: tf.Tensor, x) -> tf.Tensor:
    x = tf.broadcast_to(x, [2])
    patch_len = x[0] * x[1]
    patches = tf.reshape(tf.image.extract_patches(image, [1, x[0], x[1], 1], [1,1,1,1], [1,1,1,1], 'SAME'), [image.shape[0], image.shape[1], image.shape[2], patch_len, image.shape[-1]])
    return tf.reduce_max(tf.boolean_mask(patches, tf.reshape(x, [patch_len]), axis=3), axis=3)
def dilation2(image: tf.Tensor, mask) -> tf.Tensor:
    patch_len = mask.shape[0] * mask.shape[1]
    patches = tf.reshape(tf.image.extract_patches(image, [1, mask.shape[0], mask.shape[1], 1], [1,1,1,1], [1,1,1,1], 'SAME'), [image.shape[0], image.shape[1], image.shape[2], patch_len, image.shape[-1]])
    return tf.reduce_max(tf.boolean_mask(patches, tf.reshape(mask, [patch_len]), axis=3), axis=3)
def erosion(image: tf.Tensor, x) -> tf.Tensor:
    x = tf.broadcast_to(x, [2])
    patch_len = x[0] * x[1]
    radius = [x[0] // 2, x[1] // 2]
    padded = tf.pad(image, [[0, 0], radius, radius, [0, 0]], 'CONSTANT', constant_values=255)
    patches = tf.reshape(tf.image.extract_patches(padded, [1, x[0], x[1], 1], [1,1,1,1], [1,1,1,1], 'VALID'), [image.shape[0], image.shape[1], image.shape[2], patch_len, image.shape[-1]])
    return tf.reduce_min(patches, axis=3)
def erosion2(image: tf.Tensor, mask) -> tf.Tensor:
    patch_len = mask.shape[0] * mask.shape[1]
    radius = [mask.shape[0] // 2, mask.shape[1] // 2]
    padded = tf.pad(image, [[0, 0], radius, radius, [0, 0]], 'CONSTANT', constant_values=255)
    patches = tf.reshape(tf.image.extract_patches(padded, [1, mask.shape[0], mask.shape[1], 1], [1,1,1,1], [1,1,1,1], 'VALID'), [image.shape[0], image.shape[1], image.shape[2], patch_len, image.shape[-1]])
    return tf.reduce_min(tf.boolean_mask(patches, tf.reshape(mask, [patch_len]), axis=3), axis=3)

def opening(image: tf.Tensor, x) -> tf.Tensor:
    return dilation(erosion(image, x), x)
def opening2(image: tf.Tensor, mask) -> tf.Tensor:
    return dilation2(erosion2(image, mask), mask)
def closing(image: tf.Tensor, x) -> tf.Tensor:
    return erosion(dilation(image, x), x)
def closing2(image: tf.Tensor, mask) -> tf.Tensor:
    return erosion2(dilation2(image, mask), mask)


def rgb_to_xyz(image: tf.Tensor) -> tf.Tensor:
    """Converts an RGB image to XYZ.
    Args:
        image: Image to convert.
    """
    return tf.einsum('...ij,...j->...i', tf.constant([
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227]
    ], dtype=tf.float32), tf.cast(image, tf.float32))

# @tf.function
# def equalize_histogram(image: tf.Tensor) -> tf.Tensor:
#     """Equalizes the histogram of an 4D tensor.
#     Args:
#         image: Image to equalize.
#     """
#     channels = tf.unstack(image, axis=-1)
#     hists = tf.convert_to_tensor([tf.histogram_fixed_width(c, [0, 255], nbins=256) for c in channels])
#     cdfs = tf.cumsum(hists, axis=1)
#     cdfs = tf.cast(cdfs, tf.float32)
#     cdfs = cdfs / tf.reduce_sum(cdfs, axis=1, keepdims=True)
#     cdfs = cdfs / tf.reduce_max(cdfs, axis=1, keepdims=True)
#     cdfs = cdfs * 255
#     cdfs = tf.cast(cdfs, tf.uint8)
#     cdf = tf.gather(cdfs, tf.cast(channels, tf.int32), axis=1)
#     return tf.reshape(tf.stack(cdf, axis=-1), image.shape)

def kmeans_clustering(image: tf.Tensor, k: int, iterations: int = 10) -> tf.Tensor:
    """Performs kmeans clustering on an image.
    Args:
        image: Image to cluster.
        k: Number of clusters to use.
        iterations: Number of iterations to run.
    Returns:
        Image with k bins as values.
    """
    dtype = image.dtype
    if image.shape[-1] == 3:
        image = tf.reshape(tf.image.rgb_to_grayscale(image), tf.concat([image.shape[:-1], [1]], 0))
    else:
        image = tf.reshape(image, tf.concat([image.shape[:-1], [1]], 0))
    lin = tf.expand_dims(tf.linspace(0., 255., 256), 1)
    image_int = tf.cast(image, tf.int32)
    image_float = tf.cast(image, tf.float32)
    centroids = tf.random.uniform([1, k], 0, 255, dtype=tf.float32)
    for _ in range(iterations):
        binned = tf.gather(tf.argmin(tf.abs(lin - centroids) ** 2, 1, tf.int32), image_int)
        centroids = tf.expand_dims(tf.map_fn(lambda i: tf.reduce_mean(tf.boolean_mask(image_float, binned == i)), tf.range(k), dtype=tf.float32),0)
    return tf.cast(tf.gather(tf.argmin(tf.abs(lin - centroids) ** 2, 1), image_int), dtype) # type: ignore

# standardize image by removing background and small artifacts and using normalized histogram to do simple binarization
# assuming skin lesions are usually darker than the healthy skin and larger than small artifacts
def load_image(path):
    """Loads and preprocesses images."""
    # The delta threshold for morphological operations e.g. removing moles and hair.
    morphology_threshold = 20
    # load image and clip to foreground square
    disk3 = disk(3)
    disk5 = disk(5)
    image = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
    image = crop_to_square(image)
    image = tf.image.resize(image, (256, 256))
    image = crop_out_background(image)
    image = tf.image.resize(image, (256, 256))
    ## remove vignetting and adjust saturation
    image = tf.image.adjust_saturation(image, 1.25)
    image = unvignette(image)
    image = rgb_to_xyz(image)
    image = tf.reshape(image, tf.concat([[1], image.shape], 0))
    ## remove unimportant small dark objects from the image e.g. moles and hair
    img = closing2(image, disk3)
    img = gaussian_filter(img)
    image = tf.where(img - morphology_threshold > image, img, image)
    ## remove unimportant small bright objects from the image e.g. artifacts and glare
    img = opening2(image, disk3)
    img = gaussian_filter(img)
    image = tf.where(img + morphology_threshold < image, img, image)
    ## simple segmentation using normalized histograms and kmeans clustering
    blurred = gaussian_filter(image, sigma=3, ksize=5)
    gray = tf.image.rgb_to_grayscale(image)
    eq = tfa.image.equalize(gray)
    mask = kmeans_clustering(eq, 2, 20)
    if tf.reduce_mean(tf.boolean_mask(gray, mask == 1)) > tf.reduce_mean(tf.boolean_mask(gray, mask == 0)):
        mask = tf.where(mask == 1, 0, 1)
    else:
        mask = tf.where(mask == 1, 1, 0)
    ## slightly more aggressive morphological operations to remove unimportant objects in the thresholded mask
    mask = opening2(mask, disk5)
    mask = closing2(mask, disk5)
    mask = tf.squeeze(mask, axis=-1)
    mask = tf.stack([mask, mask, mask], axis=-1)
    image = gaussian_filter(tf.where(mask == 1, image, blurred), sigma=1)
    return tf.cast(tf.reshape(image, image.shape[1:]), tf.uint8)
