"""ResNet-50 model for land cover classification."""
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes=6):
    """Create a ResNet-50 model with a custom classification head.

    Args:
        num_classes: Number of land cover classes.

    Returns:
        Modified ResNet-50 model.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

    with torch.no_grad():
        model.conv1.weight[:, :3] = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        ).conv1.weight

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def load_model(checkpoint_path, num_classes=6, device="cpu"):
    """Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint.
        num_classes: Number of output classes.
        device: Device to load the model on.

    Returns:
        Loaded model in eval mode.
    """
    model = create_model(num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


class LandCoverClassifier:
    """Wrapper for land cover classification inference."""

    def __init__(self, num_classes=6, device="cpu"):
        self.device = torch.device(device)
        self.model = create_model(num_classes).to(self.device)
        self.model.eval()

    def predict(self, tiles, batch_size=32):
        """Run inference on image tiles.

        Args:
            tiles: numpy array of shape (N, 256, 256, 4).
            batch_size: Number of tiles per batch.

        Returns:
            Predicted class indices and probabilities.
        """
        # Convert to PyTorch format: (N, C, H, W)
        tiles_tensor = torch.from_numpy(tiles).permute(0, 3, 1, 2).float()
        tiles_tensor = tiles_tensor.to(self.device)

        # Normalize using ImageNet statistics for RGB channels
        mean = torch.tensor([0.485, 0.456, 0.406, 0.5]).view(1, 4, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225, 0.25]).view(1, 4, 1, 1).to(self.device)
        tiles_tensor = (tiles_tensor - mean) / std

        all_preds = []
        all_probs = []

        with torch.no_grad():
            for i in range(0, len(tiles_tensor), batch_size):
                batch = tiles_tensor[i:i + batch_size]
                outputs = self.model(batch)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_preds), np.array(all_probs)
