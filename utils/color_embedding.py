import numpy as np

from third_party.palette_embedding.python.palette_embedding \
    import PaletteEmbeddingModel


def rgb_to_hex(color_rgb):
    """Returns a color in hex format from a given color in rgb

    :param color_rgb: color in rgb format ex [256, 256, 256]
    :type color_rgb: list

    :return: color in hex format
    :rtype: string
    """
    return '%02x%02x%02x' % tuple(color_rgb)


class TFColorEmbedding(object):
    def __init__(self):
        self.color_embedding_model = PaletteEmbeddingModel()

    def __call__(self, rgb_color: list) -> np.ndarray:
        """
        Call method
        Args:
            rgb_color: RGB color list with uint8 values

        Returns:
            Embedding with size 15
        """
        hex_input = '-'.join([rgb_to_hex(cl) for cl in [rgb_color] * 5])

        embedding = self.color_embedding_model.Embed(hex_input)
        embedding = embedding / (np.linalg.norm(embedding) + 1E-5)

        return embedding

    def estimate_on_5_colors(self, rgb_colors: list) -> np.ndarray:
        hex_input = '-'.join([rgb_to_hex(cl) for cl in rgb_colors])

        embedding = self.color_embedding_model.Embed(hex_input)
        embedding = embedding / (np.linalg.norm(embedding) + 1E-5)

        return embedding
