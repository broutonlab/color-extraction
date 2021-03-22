import numba
import numpy as np


@numba.njit()
def check_arr(a: list, value: np.ndarray) -> bool:
    for k in range(len(a)):
        if a[k][0] == value[0] and a[k][1] == value[1] and a[k][2] == value[2]:
            return True
    return False


@numba.njit()
def get_colors_pack(img: np.ndarray) -> list:
    """
    Get unique colors from image
    Args:
        img: RGB image
    Returns:
        List of unique colors
    """
    colors_pack = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            cl = img[i, j]
            if not check_arr(colors_pack, cl):
                colors_pack.append(cl)
    return colors_pack


@numba.njit()
def get_equal_indexes(img: np.ndarray, value: np.ndarray) -> list:
    """
    Get color's class coordinates
    Args:
        img: RGB image
        value: RGB color value
    Returns:
        List of color class coordinates
    """
    eq_indexes = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            cl = img[i, j]
            if cl[0] == value[0] and cl[1] == value[1] and cl[2] == value[2]:
                eq_indexes.append([i, j])
    return eq_indexes


def multidim_intersect(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    Get intersection value, multi-dim
    Args:
        arr1: the first array (dim > 1)
        arr2: the second array (dim > 1)
    Returns:
        Intersection value of 2 arrays
    """
    arr1_view = arr1.view([('', arr1.dtype)] * arr1.shape[1])
    arr2_view = arr2.view([('', arr2.dtype)] * arr2.shape[1])
    intersected = np.intersect1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])


def get_unique_colors(img: np.ndarray, counts: bool = False):
    return np.unique(img.reshape((-1, 3)), axis=0, return_counts=counts)


def sort_classes(colors: np.ndarray, rates: np.ndarray) -> list:
    """
    Sort cover rates of each color for estimation
    Args:
        colors: array with all colors in image
        rates: array with cover rates of each color
    Returns:
        Sorted list of colors cover rates
    """
    all_classes = []
    for i in range(len(rates)):
        tup = (colors[i], rates[i])
        all_classes.append(tup)
    all_classes = sorted(all_classes, key=lambda x: x[1], reverse=True)
    sorted_rates = []
    for color_rates in all_classes:
        sorted_rates.append(color_rates[0])
    return sorted_rates


def IOU(pred, truth):
    """
    IOU metric with classes
    Args:
        pred: RGB quantized image
        truth: ground truth image (mask)
    Returns:
        IOU value for the pair image - mask
    """
    assert pred.shape == truth.shape, 'shapes'

    colors, rates = get_unique_colors(pred, counts=True)
    
    pred_classes = sort_classes(colors, rates)
    truth_classes = get_colors_pack(truth)

    pred_classes_indexes = [
        get_equal_indexes(pred, cl)
        for cl in pred_classes
    ]

    truth_classes_indexes = [
        get_equal_indexes(truth, cl)
        for cl in truth_classes
    ]

    correct_inter_sum = 0
    used = [False for _ in truth_classes]

    for pr_k, pred_cl in enumerate(pred_classes):
        max_inter = -1
        max_match_idx = -1
        for gt_k, gt_cl in enumerate(truth_classes_indexes):
            if used[gt_k]:
                continue

            inter_count = 0
            inter_count = multidim_intersect(
                np.array(pred_classes_indexes[pr_k]),
                np.array(truth_classes_indexes[gt_k])
            ).shape[0]

            if inter_count > max_inter:
                max_inter = inter_count
                max_match_idx = gt_k

        used[max_match_idx] = True

        correct_inter_sum += max_inter

    return correct_inter_sum / pred.shape[0] / pred.shape[1]

