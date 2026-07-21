#!/usr/bin/env python3
"""Predict gender from reconstructed MotionSense windows.

Example:
    python predict_gender_from_reconstruction.py \
        --reconstruction ../data/_reconstructions/motionsense_reconstruction_run_0_batch_0.pt

The reconstruction file may be either:
  * a tensor containing reconstructed windows; or
  * a dictionary with ``reconstructed_data`` and optionally ``gender``.

By default, reconstructed windows are assumed to already be standardized in
the same representation used to train the gender model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn


DEFAULT_MODEL_PATH = (
    "/scratch/ejk5818/motion-sense/codes/gen_paper_codes/"
    "motion_sense_gender_cnn.pt"
)


class GenderCNN(nn.Module):
    """Gender CNN architecture used by motion_sense_gender_cnn.pt."""

    def __init__(self, num_features: int = 12, window_size: int = 50):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=(1, 5)),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(0.2),
            nn.Conv2d(50, 40, kernel_size=(1, 5)),
            nn.ReLU(),
            nn.Conv2d(40, 40, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Dropout(0.2),
            nn.Conv2d(40, 20, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_features, window_size)
            flattened_size = self.features(dummy).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 400),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(400, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input: [batch, features, time].
        x = self.features(x.unsqueeze(1))
        return self.classifier(x).squeeze(1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reconstruction",
        default="/scratch/ejk5818/ts-inverse/reconstructions/motionsense_reconstruction_run_0_batch_0.pt",
        help="Path to the saved reconstructed-data .pt file",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to motion_sense_gender_cnn.pt",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Male-probability decision threshold",
    )
    parser.add_argument(
        "--input-is-raw",
        action="store_true",
        help=(
            "Standardize inputs using the model checkpoint statistics. "
            "Do not use this for reconstructions already in standardized space."
        ),
    )
    return parser.parse_args()


def trusted_torch_load(path: Path, device):
    """Load a trusted local file across older and newer PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_reconstruction(saved_object):
    if torch.is_tensor(saved_object) or isinstance(saved_object, np.ndarray):
        return saved_object, None

    if not isinstance(saved_object, dict):
        raise TypeError(
            "The reconstruction file must contain a tensor, NumPy array, or dictionary"
        )

    if "reconstructed_data" not in saved_object:
        raise KeyError(
            "The reconstruction dictionary does not contain 'reconstructed_data'"
        )

    return saved_object["reconstructed_data"], saved_object.get("gender")


def prepare_windows(
    reconstructed_data,
    num_features: int,
    window_size: int,
    checkpoint: dict,
    input_is_raw: bool,
):
    windows = torch.as_tensor(reconstructed_data, dtype=torch.float32)

    if windows.ndim == 2:
        windows = windows.unsqueeze(0)

    if windows.ndim != 3:
        raise ValueError(
            f"Expected a 2D or 3D reconstructed tensor, got shape {tuple(windows.shape)}"
        )

    # Convert either [batch, time, features] or [batch, features, time]
    # into the format expected by GenderCNN: [batch, features, time].
    if tuple(windows.shape[1:]) == (window_size, num_features):
        windows = windows.transpose(1, 2)
    elif tuple(windows.shape[1:]) != (num_features, window_size):
        raise ValueError(
            "Unexpected reconstructed-data shape "
            f"{tuple(windows.shape)}; expected [batch, {window_size}, {num_features}] "
            f"or [batch, {num_features}, {window_size}]"
        )

    if input_is_raw:
        if "normalization_mean" not in checkpoint or "normalization_std" not in checkpoint:
            raise KeyError(
                "--input-is-raw requires normalization_mean and normalization_std "
                "in the model checkpoint"
            )

        mean = torch.as_tensor(
            checkpoint["normalization_mean"], dtype=torch.float32
        ).view(1, num_features, 1)
        std = torch.as_tensor(
            checkpoint["normalization_std"], dtype=torch.float32
        ).view(1, num_features, 1)
        windows = (windows - mean) / std.clamp_min(1e-8)

    return windows.contiguous()


def main():
    args = parse_args()
    device = select_device()
    model_path = Path(args.model)
    reconstruction_path = Path(args.reconstruction)

    if not model_path.exists():
        raise FileNotFoundError(f"Gender model not found: {model_path}")
    if not reconstruction_path.exists():
        raise FileNotFoundError(f"Reconstruction file not found: {reconstruction_path}")

    checkpoint = trusted_torch_load(model_path, device)
    if not isinstance(checkpoint, dict):
        raise TypeError("The gender checkpoint must be a dictionary or state dict")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        # Also support a file containing only the state dict.
        state_dict = checkpoint

    num_features = int(checkpoint.get("num_features", 12))
    window_size = int(checkpoint.get("window_size", 50))

    model = GenderCNN(num_features, window_size).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    reconstruction_object = trusted_torch_load(reconstruction_path, "cpu")
    reconstructed_data, true_genders = extract_reconstruction(reconstruction_object)
    windows = prepare_windows(
        reconstructed_data,
        num_features,
        window_size,
        checkpoint,
        args.input_is_raw,
    ).to(device)

    with torch.inference_mode():
        logits = model(windows)
        male_probabilities = torch.sigmoid(logits).cpu()
        predictions = (male_probabilities >= args.threshold).long()

    true_gender_tensor = None
    if true_genders is not None:
        true_gender_tensor = torch.as_tensor(true_genders).long().reshape(-1)
        if len(true_gender_tensor) != len(predictions):
            raise ValueError(
                f"Found {len(predictions)} reconstructions but "
                f"{len(true_gender_tensor)} true gender labels"
            )

    print(f"Device: {device}")
    print(f"Reconstructed windows: {tuple(windows.shape)}")
    print("Label convention: 0 = female, 1 = male\n")

    for index, (probability, prediction) in enumerate(
        zip(male_probabilities, predictions)
    ):
        predicted_name = "male" if prediction.item() == 1 else "female"
        line = (
            f"Sample {index}: probability_male={probability.item():.6f}, "
            f"prediction={prediction.item()} ({predicted_name})"
        )

        if true_gender_tensor is not None:
            truth = true_gender_tensor[index].item()
            truth_name = "male" if truth == 1 else "female"
            line += (
                f", true_gender={truth} ({truth_name}), "
                f"correct={prediction.item() == truth}"
            )

        print(line)

    if true_gender_tensor is not None:
        accuracy = (predictions == true_gender_tensor).float().mean().item()
        print(f"\nAccuracy on reconstructed samples: {accuracy:.4f}")


if __name__ == "__main__":
    main()
