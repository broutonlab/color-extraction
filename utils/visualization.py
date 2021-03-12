import cv2
import numpy as np


def add_color_bar_to_image(
        img: np.ndarray,
        colors: list,
        cover_rates: list = None) -> np.ndarray:
    """
    Plot colors on input image
    Args:
        img: image in uint8 RGB HWC format
        colors: numpy array of colors in RGB format
        cover_rates: numpy float array of colors cover rates,
        if not usage then not plot rates

    Returns:
        Image with colors visualization
    """
    d = int(img.shape[0] * 0.1)
    res = np.zeros((img.shape[0] + d * len(colors), img.shape[1], 3), dtype=img.dtype)
    res[:img.shape[0], :] = img

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = d / 25
    thickness = 1 + d // 15

    for i in range(len(colors)):
        res[img.shape[0] + i*d:img.shape[0] + (i + 1)*d] = colors[i]

        if cover_rates is not None:
            text_color = tuple([int(255 - colors[i][ch]) for ch in range(3)])

            org = (0, img.shape[0] + i * d + int(d * 0.92))
            res = cv2.putText(
                res,
                '{:.5f}'.format(cover_rates[i]),
                org,
                font,
                fontScale,
                text_color,
                thickness,
                cv2.LINE_AA
            )

    return res
