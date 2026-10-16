"""Inference pipeline: classify tiles and rebuild the label map."""
import numpy as np


def classify_tiles(tiles, positions, classifier, num_classes=6, tile_size=256):
    """Classify all tiles and reconstruct the full label map.

    Args:
        tiles: numpy array of image tiles.
        positions: List of (x, y) tile positions.
        classifier: LandCoverClassifier instance.
        num_classes: Number of land cover classes.
        tile_size: Size of each tile.

    Returns:
        Full label map as numpy array.
    """
    predictions, probabilities = classifier.predict(tiles)

    if len(positions) == 0:
        return np.array([]), np.array([])

    max_x = max(p[0] for p in positions) + tile_size
    max_y = max(p[1] for p in positions) + tile_size

    label_map = np.zeros((max_y, max_x), dtype=np.uint8)
    confidence_map = np.zeros((max_y, max_x), dtype=np.float32)

    for (x, y), pred, probs in zip(positions, predictions, probabilities):
        label_map[y:y + tile_size, x:x + tile_size] = pred
        confidence_map[y:y + tile_size, x:x + tile_size] = probs[pred]

    return label_map, confidence_map


def calculate_statistics(label_map, class_names):
    """Calculate coverage statistics.

    Args:
        label_map: 2D array of class indices.
        class_names: List of class names.

    Returns:
        Dictionary with class names and their percentages.
    """
    total_pixels = label_map.size
    unique, counts = np.unique(label_map, return_counts=True)

    stats = {}
    for i, name in enumerate(class_names):
        count = counts[unique == i].sum() if i in unique else 0
        stats[name] = {
            "pixels": int(count),
            "percentage": round(float(count) / total_pixels * 100, 2),
        }

    return stats
