from utils.topological_colors_extraction import *
from utils.evaluation_utils import *
import os


def extract_common_colors_with_image(
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
        (list with colors in RGB format, list with correspondent cover rates, quantized image)
    """
    resized = image.copy()
    # we don't need to do resize, because mask have a certain size

    # To optimize algorithm performance
    #     if max(resized.shape) > 50:
    #         resized_k = 50 / max(resized.shape)
    #         resized = cv2.resize(
    #             resized,
    #             None,
    #             fx=resized_k,
    #             fy=resized_k,
    #             interpolation=cv2.INTER_CUBIC
    #         )
    #     color_embedding_model = TFColorEmbedding()

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
        verbose=False,
        random_state=42,
        init='spectral'
    ).fit_transform(pixels)

    clustering = DBSCAN(
        eps=0.2,
        min_samples=4,
        n_jobs=n_jobs
    ).fit(comp_pixels)

    clusters_centroids = np.array(
        [
            comp_pixels[clustering.labels_ == i].mean(axis=0)
            for i in set(clustering.labels_)
            if i != -1
        ]
    )

    clustering = KMeans(
        n_clusters=len([cl for cl in set(clustering.labels_) if cl != -1]),
        init=clusters_centroids,
        n_init=1,
        n_jobs=n_jobs,
        random_state=42
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

    return total_colors, cover_rates, quantized_resized_img

color_embedding_model = TFColorEmbedding()

originals_path = './evaluation_data/images/'
ground_truth_path = './evaluation_data/masks/'
print(f'Total amount of evaluation images:{len(os.listdir(originals_path))}')
measure = []
for img_path in (os.listdir(originals_path)):
    img = cv2.imread(os.path.join(originals_path, img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError('Can\'t open image: {}'.format(img_path))
    img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
    try:
        true_path = (os.path.join(ground_truth_path, img_path[:-3] + 'bmp'))
        gr = cv2.cvtColor(
            cv2.imread(
                true_path,
                cv2.IMREAD_COLOR
            ),
            cv2.COLOR_BGR2RGB
        )
    except:
        true_path = (os.path.join(ground_truth_path, img_path))
        gr = cv2.cvtColor(
            cv2.imread(
                true_path,
                cv2.IMREAD_COLOR
            ),
            cv2.COLOR_BGR2RGB
        )

    total_colors, cover_rates, result_img = extract_common_colors_with_image(img)
    metric = IOU(result_img, gr)
    print(f'mIOU = {metric}, for {os.path.join(originals_path, img_path)}')
    measure.append(metric)

print('Mean value for all data = ', round(sum(measure) / len(measure)), 3)

