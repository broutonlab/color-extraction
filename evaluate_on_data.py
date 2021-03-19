from utils.topological_colors_extraction import *
from utils.evaluation_utils import *
import os
from tqdm import tqdm
import random
import argparse

random.seed(42)
np.random.seed(42)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--low_memory', default=False)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    memory = parser.parse_args(sys.argv[1:])

    color_embedding_model = TFColorEmbedding()

    originals_path = './evaluation_data/images/'
    ground_truth_path = './evaluation_data/masks/'
    print(f'Total amount of evaluation images:{len(os.listdir(originals_path))}')
    measure = []  # for the mean value
    for img_path in tqdm(os.listdir(originals_path)):
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

        total_colors, cover_rates, result_img = extract_principal_colors(img,
                                                                         low_memory=memory.low_memory,
                                                                         without_resizing=True)
        metric = IOU(result_img, gr)
        print(f'mIOU = {metric}, for {os.path.join(originals_path, img_path)}')
        measure.append(metric)

    print('Mean IOU for all images = ', round(sum(measure) / len(measure), 3))

