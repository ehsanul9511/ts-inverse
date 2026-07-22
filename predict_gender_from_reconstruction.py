#!/usr/bin/env python3
"""Predict gender from reconstructed MotionSense windows.

Example:
    python predict_gender_from_reconstruction.py \
        --reconstruction ../data/_reconstructions/ \
        --output-csv gender_predictions.csv

The reconstruction file may be either:
  * a tensor containing reconstructed windows; or
  * a dictionary with ``reconstructed_data`` and optionally ``gender``.

By default, reconstructed windows are assumed to already be standardized in
the same representation used to train the gender model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
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
        default="/scratch/ejk5818/ts-inverse/reconstructions/",
        help="Path to one reconstruction .pt file or a directory containing them",
    )
    parser.add_argument(
        "--pattern",
        default="motionsense_reconstruction_run_*_batch_*.pt",
        help="Glob used when --reconstruction is a directory",
    )
    parser.add_argument(
        "--output-csv",
        default="gender_predictions.csv",
        help="Destination CSV file",
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


def reconstruction_sort_key(path: Path):
    """Sort run and batch numbers numerically instead of lexicographically."""
    match = re.search(r"run_(\d+)_batch_(\d+)", path.stem)
    if match:
        return int(match.group(1)), int(match.group(2))
    return float("inf"), path.name


def main():
    args = parse_args()
    device = select_device()
    model_path = Path(args.model)
    reconstruction_path = Path(args.reconstruction)
    output_csv = Path(args.output_csv)

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

    if reconstruction_path.is_dir():
        reconstruction_files = sorted(
            reconstruction_path.glob(args.pattern),
            key=reconstruction_sort_key,
        )
    else:
        reconstruction_files = [reconstruction_path]

    if not reconstruction_files:
        raise FileNotFoundError(
            f"No files matching {args.pattern!r} found in {reconstruction_path}"
        )

    all_windows = []
    all_genders = []
    source_files = []

    for file_path in reconstruction_files:
        reconstruction_object = trusted_torch_load(file_path, "cpu")
        reconstructed_data, genders = extract_reconstruction(reconstruction_object)
        prepared = prepare_windows(
            reconstructed_data,
            num_features,
            window_size,
            checkpoint,
            args.input_is_raw,
        )

        if genders is None:
            raise KeyError(f"No gender labels found in {file_path}")

        genders = torch.as_tensor(genders, dtype=torch.long).reshape(-1)
        if len(genders) != len(prepared):
            raise ValueError(
                f"{file_path} has {len(prepared)} reconstructions but "
                f"{len(genders)} gender labels"
            )

        all_windows.append(prepared)
        all_genders.append(genders)
        source_files.extend([file_path.name] * len(prepared))

    windows = torch.cat(all_windows, dim=0).to(device)
    true_gender_tensor = torch.cat(all_genders, dim=0)

    with torch.inference_mode():
        logits = model(windows)
        male_probabilities = torch.sigmoid(logits).cpu()
        predictions = (male_probabilities >= args.threshold).long()

    if len(true_gender_tensor) != len(predictions):
        raise ValueError(
            f"Found {len(predictions)} predictions but "
            f"{len(true_gender_tensor)} true gender labels"
        )

    results = pd.DataFrame(
        {
            "sample_index": np.arange(len(predictions)),
            "source_file": source_files,
            "probability_male": male_probabilities.numpy(),
            "prediction": predictions.numpy(),
            "predicted_gender": np.where(predictions.numpy() == 1, "male", "female"),
            "true_gender": true_gender_tensor.numpy(),
            "true_gender_name": np.where(
                true_gender_tensor.numpy() == 1, "male", "female"
            ),
            "correct": (predictions == true_gender_tensor).numpy(),
        }
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)

    accuracy = results["correct"].mean()
    print(f"Loaded {len(reconstruction_files)} reconstruction files")
    print(f"Evaluated {len(results)} reconstructed windows on {device}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Saved predictions: {output_csv.resolve()}")


if __name__ == "__main__":
    main()
