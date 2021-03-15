import cv2
import numpy as np
from sklearn.cluster import DBSCAN, KMeans

from utils.color_embedding import TFColorEmbedding
from utils.array_utils import get_median_intensive_value

import sys
sys.path.insert(0, '../third_party/umato/')
import umato


def colors_grouping(
        pixel_values: list,
        th: float = 25,
        measure_f: callable = lambda cl1, cl2: np.linalg.norm(cl1 - cl2)):
    """
    Grouping colors to clusters
    Args:
        pixel_values: RGB colors list
        th: values matching threshold
        measure_f: distance estimation function

    Returns:
        Clusters with elements indexes
    """
    clusters_indexes = []

    for i in range(len(pixel_values)):
        if len(clusters_indexes) == 0:
            clusters_indexes = [[i]]
            continue

        min_d = 1000
        min_cluster_idx = -1

        for j in range(len(clusters_indexes)):
            for k in clusters_indexes[j]:
                d = measure_f(
                    pixel_values[i].astype(np.float32),
                    pixel_values[k].astype(np.float32)
                )
                if d - min_d < 1E-5:
                    min_d = d
                    min_cluster_idx = j

        if min_cluster_idx >= 0 and min_d - th < 1E-5:
            clusters_indexes[min_cluster_idx].append(i)
        else:
            clusters_indexes.append([i])

    return clusters_indexes


def get_class_index_from_groups(groups, idx) -> int:
    for group in groups:
        if idx in group:
            return group[len(group) // 2]


def extract_common_colors(
        image: np.ndarray,
        matching_threshold: float = 65 / 255,
        n_jobs: int = 4) -> tuple:
    """
    Extract common colors with cover rates
    Args:
        image: image in RGB HWC uint8 format
        matching_threshold: matching threshold, more value - fewer colors
        n_jobs: count of parallel processes

    Returns:
        (list with colors in RGB format, list with correspondent cover rates)
    """
    resized = image.copy()

    # To optimize algorithm performance
    if max(resized.shape) > 50:
        resized_k = 50 / max(resized.shape)
        resized = cv2.resize(
            resized,
            None,
            fx=resized_k,
            fy=resized_k,
            interpolation=cv2.INTER_CUBIC
        )

    color_embedding_model = TFColorEmbedding()

    pixels = np.array(
        [
            color_embedding_model(cl)
            for cl in resized.reshape((-1, 3))
        ]
    )

    comp_pixels = umato.UMATO(
        n_neighbors=250,
        global_n_epochs=50,
        local_n_epochs=50,
        hub_num=300,
        verbose=False
    ).fit_transform(pixels)

    clustering = DBSCAN(
        eps=0.2,
        min_samples=20,
        n_jobs=n_jobs
    ).fit(comp_pixels)

    clustering = KMeans(
        n_clusters=len(set(clustering.labels_)),
        n_init=10,
        n_jobs=n_jobs
    ).fit(comp_pixels)

    cluster_values = [
        resized.reshape((-1, 3))[clustering.labels_ == i]
        for i in set(clustering.labels_)
    ]

    cluster_values = [
        get_median_intensive_value(np.array(cluster_values[i]))
        for i in range(len(cluster_values))
    ]

    cluster_values_with_sorted_indexes = list(
        zip(cluster_values, range(len(cluster_values))))
    cluster_values_with_sorted_indexes.sort(key=lambda x: x[0].mean(),
                                            reverse=True)

    print(int(matching_threshold * 255))
    grouping_indexes = colors_grouping(
        [cl[0] for cl in cluster_values_with_sorted_indexes],
        int(matching_threshold * 255)
    )

    grouping_indexes = [
        [cluster_values_with_sorted_indexes[v][1] for v in g]
        for g in grouping_indexes
    ]

    clusters_set = list(set(clustering.labels_))

    quantized_resized_img = np.array([
        cluster_values[
            get_class_index_from_groups(
                grouping_indexes,
                clusters_set.index(clustering.labels_[i])
            )
        ]
        for i, p in enumerate(resized.reshape((-1, 3)))
    ]).reshape(resized.shape)

    total_colors, cover_rates = np.unique(
        quantized_resized_img.reshape((-1, 3)),
        axis=0,
        return_counts=True
    )

    cover_rates = np.array([
        count / np.sum(cover_rates)
        for count in cover_rates
    ])

    colors_with_rates = [
        (total_colors[i], cover_rates[i]) for i in range(len(cover_rates))
    ]
    colors_with_rates = sorted(
        colors_with_rates,
        key=lambda x: x[1],
        reverse=True
    )

    total_colors = [tup[0] for tup in colors_with_rates]
    cover_rates = [tup[1] for tup in colors_with_rates]

    return total_colors, cover_rates
