"""
Main pipeline for land cover classification of Armenia, Quindio, Colombia.

This script:
1. Connects to Google Earth Engine
2. Downloads the most recent Sentinel-2 image
3. Preprocesses the image (RGB + NDVI, tiling)
4. Classifies each tile using ResNet-50
5. Generates an interactive HTML map with results

Usage:
    python main.py
"""

import json
import os
import sys

import numpy as np
import ee

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.gee_auth import initialize_ee
from modules.download import download_sentinel2
from modules.preprocess import create_rgb_ndvi, split_into_tiles
from modules.model import LandCoverClassifier
from modules.inference import classify_tiles, calculate_statistics
from modules.visualize import create_interactive_map


def load_config(config_path="config.json"):
    """Load project configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def image_to_array(image, bbox, scale=10):
    """Convert an ee.Image to a numpy array.

    Args:
        image: ee.Image with bands B4, B3, B2, B8.
        bbox: Region to export.
        scale: Resolution in meters.

    Returns:
        numpy array (H, W, 4) with values 0-10000.
    """
    url = image.getDownloadURL({
        "region": bbox,
        "scale": scale,
        "format": "NPY",
    })

    import urllib.request
    import io

    response = urllib.request.urlopen(url)
    data = np.load(io.BytesIO(response.read()))
    return data


def main():
    """Execute the full land cover classification pipeline."""
    print("=" * 60)
    print("LAND COVER CLASSIFICATION - ARMENIA, QUINDIO")
    print("=" * 60)

    # Step 1: Load configuration
    print("\n[Step 1/6] Loading configuration...")
    config = load_config()
    lat = config["location"]["latitude"]
    lon = config["location"]["longitude"]
    buffer_km = config["location"]["buffer_km"]
    tile_size = config["model"]["tile_size"]
    class_names = [c["name"] for c in config["classes"]]
    class_colors = {c["name"]: c["color"] for c in config["classes"]}
    print(f"  Location: {config['location']['name']}")
    print(f"  Coordinates: ({lat}, {lon})")
    print(f"  Buffer: {buffer_km} km")
    print(f"  Tile size: {tile_size}x{tile_size}")

    # Step 2: Initialize Earth Engine
    print("\n[Step 2/6] Initializing Google Earth Engine...")
    initialize_ee()

    # Step 3: Download Sentinel-2 image
    print("\n[Step 3/6] Downloading Sentinel-2 image...")
    bbox = ee.Geometry.Rectangle(
        [lon - buffer_km / 111.0, lat - buffer_km / 111.0,
         lon + buffer_km / 111.0, lat + buffer_km / 111.0]
    )
    image = download_sentinel2(
        lat, lon,
        buffer_km=buffer_km,
        max_cloud_cover=config["image"]["max_cloud_cover"],
    )
    print("  Image downloaded successfully.")

    # Step 4: Preprocess image
    print("\n[Step 4/6] Preprocessing image...")
    print("  Converting to numpy array...")
    image_array = image_to_array(image, bbox, scale=config["image"]["resolution_m"])
    print(f"  Image shape: {image_array.shape}")

    print("  Creating RGB + NDVI composite...")
    rgb_ndvi = create_rgb_ndvi(image_array)

    print("  Splitting into tiles...")
    tiles, positions = split_into_tiles(rgb_ndvi, tile_size=tile_size)
    print(f"  Number of tiles: {len(tiles)}")

    # Step 5: Classify tiles
    print("\n[Step 5/6] Classifying tiles with ResNet-50...")
    device = "cuda" if False else "cpu"  # Set to "cuda" if GPU available
    classifier = LandCoverClassifier(num_classes=len(class_names), device=device)

    label_map, confidence_map = classify_tiles(
        tiles, positions, classifier,
        num_classes=len(class_names), tile_size=tile_size,
    )

    stats = calculate_statistics(label_map, class_names)
    print("\n  Coverage Statistics:")
    for name, info in stats.items():
        print(f"    {name:15s}: {info['percentage']:6.2f}%")

    # Step 6: Generate interactive map
    print("\n[Step 6/6] Generating interactive map...")
    output_path = "mapa_armenia.html"
    create_interactive_map(lat, lon, label_map, class_colors, stats, output_path)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Open '{output_path}' in your browser to view the results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
