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
    colors_pack = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            cl = img[i, j]
            if cl[0] == 0 and cl[1] == 0 and cl[2] == 0:
                continue
            if not check_arr(colors_pack, cl):
                colors_pack.append(cl)
    return colors_pack


@numba.njit()
def get_equal_indexes(img: np.ndarray, value: np.ndarray) -> list:
    eq_indexes = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            cl = img[i, j]
            if cl[0] == value[0] and cl[1] == value[1] and cl[2] == value[2]:
                eq_indexes.append([i, j])

    return eq_indexes


def estimate_measure(pred, truth):
    assert pred.shape == truth.shape, 'shapes'

    pred_classes = get_colors_pack(pred)
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
            for gt_pl_idx in truth_classes_indexes[gt_k]:
                for pr_pl_idx in pred_classes_indexes[pr_k]:
                    if gt_pl_idx == pr_pl_idx:
                        inter_count += 1

            if inter_count > max_inter:
                max_inter = inter_count
                max_match_idx = gt_k

        used[max_match_idx] = True

        print(pr_k, '->', max_match_idx)

        correct_inter_sum += max_inter

    return correct_inter_sum / pred.shape[0] / pred.shape[1]
