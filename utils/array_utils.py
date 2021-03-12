import numpy as np
import scipy
from scipy.spatial.distance import cdist


def get_mean_intensive_value(x):
    pixels_brightness = np.expand_dims(x.mean(axis=1), 1)
    nearest_idx = scipy.spatial.distance.cdist(
        pixels_brightness,
        np.expand_dims(np.expand_dims(pixels_brightness.mean(), 0), 0)
    ).squeeze(axis=1).argmin()
    return x[nearest_idx]


def get_max_intensive_value(x):
    pixels_brightness = np.expand_dims(x.mean(axis=1), 1)
    nearest_idx = scipy.spatial.distance.cdist(
        pixels_brightness,
        np.expand_dims(np.expand_dims(pixels_brightness.mean(), 0), 0)
    ).squeeze(axis=1).argmax()
    return x[nearest_idx]


def get_min_intensive_value(x):
    pixels_brightness = np.expand_dims(x.mean(axis=1), 1)
    nearest_idx = scipy.spatial.distance.cdist(
        pixels_brightness,
        np.expand_dims(np.expand_dims(pixels_brightness.mean(), 0), 0)
    ).squeeze(axis=1).argmin()
    return x[nearest_idx]


def argmedian(x):
    return np.argpartition(x, len(x) // 2, axis=0)[len(x) // 2]


def get_median_intensive_value(x):
    pixels_brightness = x.mean(axis=1)
    return x[argmedian(pixels_brightness)]
