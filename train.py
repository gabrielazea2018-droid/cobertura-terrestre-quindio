"""
Training script for ResNet-50 land cover classification model.

Uses ESA WorldCover labels from Google Earth Engine to train on
Sentinel-2 images from Colombia.

Usage:
    python train.py
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.model import create_model
from modules.gee_auth import initialize_ee
import ee


class LandCoverDataset(Dataset):
    """Dataset for land cover classification using ESA WorldCover labels."""

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def download_training_data(region, label_class="WorldCover", max_images=50):
    """Download paired Sentinel-2 and WorldCover data from GEE.

    Args:
        region: ee.Geometry defining the study area.
        label_class: Label collection name.
        max_images: Maximum number of images to download.

    Returns:
        Tuple of (images_array, labels_array).
    """
    # ESA WorldCover class mapping to our 6 classes
    # 10: Tree cover -> Forest
    # 20: Shrubland -> Grassland
    # 30: Grassland -> Grassland
    # 40: Cropland -> Agriculture
    # 50: Built-up -> Urban
    # 80: Permanent water bodies -> Water
    # 90: Herbaceous wetland -> Bare Soil (closest match)
    # 95: Mangroves -> Forest
    worldcover_map = {
        10: 0,   # Forest
        20: 1,   # Grassland (Shrubland)
        30: 1,   # Grassland
        40: 2,   # Agriculture (Cropland)
        50: 3,   # Urban (Built-up)
        60: 5,   # Bare Soil
        70: 5,   # Bare Soil
        80: 4,   # Water
        90: 5,   # Bare Soil (Herbaceous wetland)
        95: 0,   # Forest (Mangroves)
        100: 0,  # Forest (Moss and lichen)
    }

    # Get WorldCover label image
    worldcover = (
        ee.Image("ESA/WorldCover/v200/2021")
        .clip(region)
        .remap(list(worldcover_map.keys()), list(worldcover_map.values()))
    )

    # Get Sentinel-2 collection
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
        .filterDate("2024-01-01", "2026-12-31")
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    # Take first images
    image_list = collection.toList(max_images)

    images = []
    labels = []

    for i in range(max_images):
        try:
            image = ee.Image(image_list.get(i))
            # Select bands: B4, B3, B2, B8
            image = image.select(["B4", "B3", "B2", "B8"])

            # Download as numpy
            image_url = image.getDownloadURL({
                "region": region,
                "scale": 10,
                "format": "NPY",
            })
            label_url = worldcover.getDownloadURL({
                "region": region,
                "scale": 10,
                "format": "NPY",
            })

            import urllib.request
            import io

            img_data = np.load(io.BytesIO(urllib.request.urlopen(image_url).read()))
            lbl_data = np.load(io.BytesIO(urllib.request.urlopen(label_url).read()))

            # Match shapes
            min_h = min(img_data.shape[0], lbl_data.shape[0])
            min_w = min(img_data.shape[1], lbl_data.shape[1])

            img_data = img_data[:min_h, :min_w]
            lbl_data = lbl_data[:min_h, :min_w]

            images.append(img_data)
            labels.append(lbl_data)

            print(f"  Downloaded image {i + 1}/{max_images}")

        except Exception as e:
            print(f"  Skipped image {i + 1}: {e}")
            continue

    return np.array(images), np.array(labels)


def create_training_pairs(images, labels, tile_size=256, samples_per_image=100):
    """Create training pairs by sampling tiles from images.

    Args:
        images: Array of Sentinel-2 images.
        labels: Array of WorldCover labels.
        tile_size: Size of each tile.
        samples_per_image: Number of random samples per image.

    Returns:
        Tuple of (tiles_array, labels_array).
    """
    all_tiles = []
    all_labels = []

    for img, lbl in zip(images, labels):
        h, w = img.shape[:2]

        if h < tile_size or w < tile_size:
            continue

        red = img[:, :, 0].astype(float)
        nir = img[:, :, 3].astype(float)
        ndvi = np.where((nir + red) == 0, 0, (nir - red) / (nir + red))
        ndvi_norm = (ndvi + 1) / 2

        rgb = img[:, :, :3].astype(float) / 10000.0
        rgb = np.clip(rgb, 0, 1)

        rgb_ndvi = np.dstack([rgb, ndvi_norm])

        # Sample random tiles
        for _ in range(samples_per_image):
            y = np.random.randint(0, h - tile_size)
            x = np.random.randint(0, w - tile_size)

            tile = rgb_ndvi[y:y + tile_size, x:x + tile_size]
            tile_label = lbl[y + tile_size // 2, x + tile_size // 2]

            all_tiles.append(tile)
            all_labels.append(tile_label)

    return np.array(all_tiles), np.array(all_labels)


def train_model(train_loader, val_loader, num_classes=6, epochs=20, device="cpu"):
    """Train the ResNet-50 model.

    Args:
        train_loader: Training data loader.
        val_loader: Validation data loader.
        num_classes: Number of output classes.
        epochs: Number of training epochs.
        device: Device to train on.

    Returns:
        Trained model.
    """
    model = create_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                "best_model.pth",
            )
            print("  -> Best model saved!")

    return model


def main():
    """Execute the full training pipeline."""
    print("=" * 60)
    print("MODEL TRAINING - LAND COVER CLASSIFICATION")
    print("=" * 60)

    # Load configuration
    with open("config.json", "r") as f:
        config = json.load(f)

    lat = config["location"]["latitude"]
    lon = config["location"]["longitude"]
    buffer_km = config["location"]["buffer_km"]
    tile_size = config["model"]["tile_size"]

    # Step 1: Initialize Earth Engine
    print("\n[Step 1/5] Initializing Google Earth Engine...")
    initialize_ee()

    # Step 2: Define training region
    print("\n[Step 2/5] Defining training region...")
    region = ee.Geometry.Rectangle(
        [lon - buffer_km / 111.0, lat - buffer_km / 111.0,
         lon + buffer_km / 111.0, lat + buffer_km / 111.0]
    )

    # Step 3: Download training data
    print("\n[Step 3/5] Downloading training data...")
    print("  (This may take several minutes...)")
    images, labels = download_training_data(region, max_images=30)

    if len(images) == 0:
        print("ERROR: No images were downloaded. Try a different region or date range.")
        return

    print(f"  Downloaded {len(images)} image-label pairs")

    # Step 4: Create training pairs
    print("\n[Step 4/5] Creating training tiles...")
    tiles, tile_labels = create_training_pairs(
        images, labels, tile_size=tile_size, samples_per_image=200
    )
    print(f"  Created {len(tiles)} training samples")

    # Convert to PyTorch format
    tiles_tensor = torch.from_numpy(tiles).permute(0, 3, 1, 2).float()
    labels_tensor = torch.from_numpy(tile_labels).long()

    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406, 0.5]).view(1, 4, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225, 0.25]).view(1, 4, 1, 1)
    tiles_tensor = (tiles_tensor - mean) / std

    # Split into train/validation
    dataset = torch.utils.data.TensorDataset(tiles_tensor, labels_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Step 5: Train model
    print("\n[Step 5/5] Training ResNet-50 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    model = train_model(train_loader, val_loader, epochs=20, device=device)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("Best model saved as: best_model.pth")
    print("Run 'python main.py' to classify new images.")
    print("=" * 60)


if __name__ == "__main__":
    main()
