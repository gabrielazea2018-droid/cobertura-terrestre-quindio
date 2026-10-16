"""Image preprocessing: NDVI calculation and tiling."""
import numpy as np


def calculate_ndvi(image_array):
    """Calculate NDVI from image array.

    Args:
        image_array: numpy array with bands [B4, B3, B2, B8].

    Returns:
        NDVI array.
    """
    red = image_array[:, :, 0].astype(float)
    nir = image_array[:, :, 3].astype(float)
    ndvi = np.where((nir + red) == 0, 0, (nir - red) / (nir + red))
    return np.clip(ndvi, -1, 1)


def create_rgb_ndvi(image_array):
    """Combine RGB bands with NDVI into a 4-channel image.

    Args:
        image_array: numpy array with bands [B4, B3, B2, B8].

    Returns:
        4-channel numpy array [R, G, B, NDVI] normalized to 0-1.
    """
    ndvi = calculate_ndvi(image_array)
    rgb = image_array[:, :, :3].astype(float) / 10000.0
    rgb = np.clip(rgb, 0, 1)

    ndvi_norm = (ndvi + 1) / 2

    return np.dstack([rgb, ndvi_norm])


def split_into_tiles(image, tile_size=256):
    """Split image into tiles.

    Args:
        image: numpy array (H, W, C).
        tile_size: Size of each square tile.

    Returns:
        List of tiles and their positions.
    """
    h, w = image.shape[:2]
    tiles = []
    positions = []

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = image[y:y + tile_size, x:x + tile_size]
            if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                tiles.append(tile)
                positions.append((x, y))

    return np.array(tiles), positions
