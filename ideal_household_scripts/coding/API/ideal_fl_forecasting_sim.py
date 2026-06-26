#!/usr/bin/env python3
"""
Barebones FL simulation for the IDEAL Household dataset.

Primary task
------------
Short-horizon forecasting: predict future aggregate electricity from a recent
multivariate household sensor window.

FL setup
--------
- Each household/home is one FL client.
- Each global round uses the current global model.
- For each selected client, one batch is drawn, a PyTorch gradient is computed,
  and the server immediately averages those client gradients.
- The averaged gradient is applied once to the global model.

Expected raw IDEAL local API layout
-----------------------------------
This script expects the sensor CSV files used by IdealDataInterface.py, i.e.,
files named like home*_room*_sensor*_<category>_<subtype>.csv.gz in DATA_DIR.
Put IdealDataInterface.py either in the same directory as this script or pass
--interface-path.

Example
-------
python ideal_fl_forecasting_sim.py \
  --data-dir /path/to/sensordata \
  --interface-path ./IdealDataInterface.py \
  --rounds 20 \
  --batch-size 16 \
  --resample-rule 30min \
  --input-steps 48 \
  --horizon-steps 1 \
  --max-clients 20 \
  --log-file ideal_fl.log
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import logging
import math
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_FEATURE_GROUPS = {
    # These regexes are intentionally broad because IDEAL file naming differs
    # across sensor types. The script searches category, subtype, and room_type.
    "electricity": {
        "regex": r"electric|mains|power|apparent|active",
        "agg": "mean",
    },
    "gas": {
        "regex": r"\bgas\b",
        "agg": "sum",
    },
    "temperature": {
        "regex": r"temperature|temp|hot-pipe|cold-pipe|boiler|radiator",
        "agg": "mean",
    },
    "humidity": {
        "regex": r"humidity",
        "agg": "mean",
    },
    "light": {
        "regex": r"light|lux",
        "agg": "mean",
    },
}


@dataclass
class ClientInfo:
    home_id: str
    num_samples: int
    num_batches: int
    features: List[str]
    date_min: str
    date_max: str


class HouseholdWindowDataset(Dataset):
    """Sliding-window dataset for one household/client."""

    def __init__(
        self,
        home_id: str,
        frame: pd.DataFrame,
        input_steps: int,
        horizon_steps: int,
        target_col: str = "electricity",
        standardize: bool = True,
    ) -> None:
        if target_col not in frame.columns:
            raise ValueError(f"target_col={target_col!r} not found in columns={list(frame.columns)}")
        if len(frame) <= input_steps + horizon_steps:
            raise ValueError(
                f"Not enough rows for home {home_id}: rows={len(frame)}, "
                f"input_steps={input_steps}, horizon_steps={horizon_steps}"
            )

        self.home_id = str(home_id)
        self.columns = list(frame.columns)
        self.target_idx = self.columns.index(target_col)
        self.input_steps = input_steps
        self.horizon_steps = horizon_steps

        values = frame.to_numpy(dtype=np.float32)
        if standardize:
            mean = np.nanmean(values, axis=0, keepdims=True)
            std = np.nanstd(values, axis=0, keepdims=True)
            std[std < 1e-6] = 1.0
            values = (values - mean) / std

        # Final finite check after cleaning/standardization.
        if not np.isfinite(values).all():
            bad = np.argwhere(~np.isfinite(values))[:10]
            raise ValueError(f"Non-finite values remain for home {home_id}. First bad indices: {bad}")

        self.values = values
        self.num_samples = len(values) - input_steps - horizon_steps + 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        start = idx
        end = idx + self.input_steps
        target_t = end + self.horizon_steps - 1

        x = self.values[start:end, :]  # [T, C]
        y = self.values[target_t, self.target_idx]  # scalar future electricity

        target_series_in_x = x[:, self.target_idx]
        peak_pos = int(np.argmax(target_series_in_x))
        peak_value = float(target_series_in_x[peak_pos])

        # Conv1d expects [C, T].
        x_tensor = torch.from_numpy(x.T.copy()).float()
        y_tensor = torch.tensor([y], dtype=torch.float32)
        peak_value_tensor = torch.tensor([peak_value], dtype=torch.float32)
        peak_pos_tensor = torch.tensor(peak_pos, dtype=torch.long)
        return x_tensor, y_tensor, peak_value_tensor, peak_pos_tensor


class TinyForecastCNN(nn.Module):
    """Small 1D CNN forecasting model: [B, C, T] -> scalar."""

    def __init__(self, num_features: int, hidden: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(num_features, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def setup_logging(log_file: Optional[str], verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def load_ideal_interface(interface_path: Optional[str]):
    if interface_path:
        path = Path(interface_path).expanduser().resolve()
    else:
        path = Path(__file__).with_name("IdealDataInterface.py").resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find IdealDataInterface.py at {path}. "
            "Pass --interface-path /path/to/IdealDataInterface.py"
        )

    spec = importlib.util.spec_from_file_location("IdealDataInterface", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import IdealDataInterface from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.IdealDataInterface


def compile_feature_groups(path: Optional[str]) -> Dict[str, Dict[str, str]]:
    if path is None:
        return DEFAULT_FEATURE_GROUPS
    with open(path, "r", encoding="utf-8") as f:
        groups = json.load(f)
    for name, cfg in groups.items():
        if "regex" not in cfg:
            raise ValueError(f"Feature group {name!r} must contain a regex field")
        cfg.setdefault("agg", "mean")
    return groups


def row_matches_group(row: pd.Series, regex: str) -> bool:
    text = " ".join(
        str(row.get(k, ""))
        for k in ["homeid", "roomid", "room_type", "category", "subtype", "sensorid", "filename"]
    ).lower()
    return re.search(regex.lower(), text) is not None


def aggregate_series(series_list: List[pd.Series], agg: str, group_name: str) -> Optional[pd.Series]:
    if not series_list:
        return None
    df = pd.concat(series_list, axis=1)
    if agg == "sum":
        out = df.sum(axis=1, min_count=1)
    elif agg == "median":
        out = df.median(axis=1)
    elif agg == "max":
        out = df.max(axis=1)
    else:
        out = df.mean(axis=1)
    out.name = group_name
    return out


def clean_household_frame(
    df: pd.DataFrame,
    min_non_nan_fraction: float,
    max_interpolate_gap: int,
) -> pd.DataFrame:
    # Remove columns that are almost entirely missing.
    keep_cols = [c for c in df.columns if df[c].notna().mean() >= min_non_nan_fraction]
    dropped = sorted(set(df.columns) - set(keep_cols))
    if dropped:
        logging.getLogger("data").debug("Dropping sparse columns: %s", dropped)
    df = df[keep_cols]

    # Fill small gaps. Remaining missing rows are dropped.
    df = df.sort_index()
    df = df.interpolate(method="time", limit=max_interpolate_gap, limit_direction="both")
    df = df.ffill(limit=max_interpolate_gap).bfill(limit=max_interpolate_gap)
    before = len(df)
    df = df.dropna(axis=0, how="any")
    after = len(df)
    if before != after:
        logging.getLogger("data").debug("Dropped %d rows with remaining NaN", before - after)
    return df


def build_client_frame(
    ideal,
    mapping: pd.DataFrame,
    data_dir: Path,
    home_id: str,
    feature_groups: Dict[str, Dict[str, str]],
    resample_rule: str,
    min_non_nan_fraction: float,
    max_interpolate_gap: int,
    max_sensors_per_group: int,
) -> Optional[pd.DataFrame]:
    log = logging.getLogger("data")
    home_rows = mapping[mapping["homeid"].astype(str) == str(home_id)].copy()
    if home_rows.empty:
        log.warning("Home %s has no sensor rows in mapping", home_id)
        return None

    group_outputs = []
    for group_name, cfg in feature_groups.items():
        regex = cfg["regex"]
        agg = cfg.get("agg", "mean")
        rows = home_rows[home_rows.apply(lambda r: row_matches_group(r, regex), axis=1)]
        if rows.empty:
            log.debug("Home %s: no sensors matched group=%s regex=%s", home_id, group_name, regex)
            continue
        if max_sensors_per_group > 0 and len(rows) > max_sensors_per_group:
            log.debug(
                "Home %s: group=%s matched %d sensors; limiting to %d",
                home_id, group_name, len(rows), max_sensors_per_group,
            )
            rows = rows.head(max_sensors_per_group)

        series_list: List[pd.Series] = []
        for _, row in rows.iterrows():
            fname = data_dir / str(row["filename"])
            if not fname.exists():
                log.warning("Home %s: missing file %s", home_id, fname)
                continue
            try:
                ts = ideal.read_csv_(fname, subtype=str(row["subtype"]))
                ts = pd.to_numeric(ts, errors="coerce")
                ts = ts.resample(resample_rule).mean()
                if ts.notna().sum() == 0:
                    log.debug("Home %s: all-NaN after resampling file=%s", home_id, fname.name)
                    continue
                series_list.append(ts)
            except Exception:
                log.exception("Home %s: failed reading/resampling %s", home_id, fname)

        combined = aggregate_series(series_list, agg=agg, group_name=group_name)
        if combined is not None:
            log.debug(
                "Home %s: feature=%s from %d sensors, non_nan=%d, start=%s, end=%s",
                home_id, group_name, len(series_list), int(combined.notna().sum()),
                combined.dropna().index.min() if combined.notna().any() else None,
                combined.dropna().index.max() if combined.notna().any() else None,
            )
            group_outputs.append(combined)

    if not group_outputs:
        log.warning("Home %s: no usable feature groups", home_id)
        return None

    df = pd.concat(group_outputs, axis=1)
    df = clean_household_frame(
        df,
        min_non_nan_fraction=min_non_nan_fraction,
        max_interpolate_gap=max_interpolate_gap,
    )
    if df.empty:
        log.warning("Home %s: empty frame after cleaning", home_id)
        return None
    return df


def build_federated_dataloaders(args) -> Tuple[Dict[str, DataLoader], List[ClientInfo], List[str]]:
    log = logging.getLogger("data")
    IdealDataInterface = load_ideal_interface(args.interface_path)
    data_dir = Path(args.data_dir).expanduser().resolve()
    log.info("Initializing IDEAL interface with data_dir=%s", data_dir)
    ideal = IdealDataInterface(data_dir)

    mapping = ideal.sensorid_mapping.reset_index()
    mapping["homeid"] = mapping["homeid"].astype(str)
    log.info("Sensor mapping rows=%d", len(mapping))
    log.info("Available categories/subtypes sample:\n%s", mapping[["category", "subtype"]].drop_duplicates().head(50).to_string(index=False))

    feature_groups = compile_feature_groups(args.feature_groups_json)
    log.info("Feature groups: %s", json.dumps(feature_groups, indent=2))
    if args.target_col not in feature_groups:
        log.warning(
            "target_col=%s is not a feature group name. It must still appear as a built frame column.",
            args.target_col,
        )

    home_ids = sorted(mapping["homeid"].unique(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    if args.max_clients > 0:
        home_ids = home_ids[: args.max_clients]
    log.info("Candidate households=%d; first ids=%s", len(home_ids), home_ids[:20])

    dataloaders: Dict[str, DataLoader] = {}
    infos: List[ClientInfo] = []
    common_features: Optional[List[str]] = None

    for home_id in home_ids:
        log.info("Building client dataset for home=%s", home_id)
        try:
            frame = build_client_frame(
                ideal=ideal,
                mapping=mapping,
                data_dir=data_dir,
                home_id=home_id,
                feature_groups=feature_groups,
                resample_rule=args.resample_rule,
                min_non_nan_fraction=args.min_non_nan_fraction,
                max_interpolate_gap=args.max_interpolate_gap,
                max_sensors_per_group=args.max_sensors_per_group,
            )
            if frame is None:
                continue

            if args.target_col not in frame.columns:
                log.warning("Home %s skipped: target_col=%s absent; columns=%s", home_id, args.target_col, list(frame.columns))
                continue

            # To keep the model input dimension fixed, use the first valid client's
            # columns as the common set and require later clients to contain them.
            if common_features is None:
                common_features = list(frame.columns)
                log.info("Common feature columns set from home %s: %s", home_id, common_features)
            missing = [c for c in common_features if c not in frame.columns]
            if missing:
                log.warning("Home %s skipped: missing common features %s; available=%s", home_id, missing, list(frame.columns))
                continue
            frame = frame[common_features]

            ds = HouseholdWindowDataset(
                home_id=home_id,
                frame=frame,
                input_steps=args.input_steps,
                horizon_steps=args.horizon_steps,
                target_col=args.target_col,
                standardize=not args.no_standardize,
            )
            if len(ds) < args.min_samples_per_client:
                log.warning("Home %s skipped: only %d samples", home_id, len(ds))
                continue
            dl = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=args.num_workers,
            )
            dataloaders[home_id] = dl
            infos.append(
                ClientInfo(
                    home_id=home_id,
                    num_samples=len(ds),
                    num_batches=len(dl),
                    features=list(frame.columns),
                    date_min=str(frame.index.min()),
                    date_max=str(frame.index.max()),
                )
            )
            log.info(
                "Home %s accepted: rows=%d samples=%d batches=%d date_range=[%s, %s] features=%s",
                home_id, len(frame), len(ds), len(dl), frame.index.min(), frame.index.max(), list(frame.columns),
            )
        except Exception:
            log.exception("Failed to build client for home=%s", home_id)

    if not dataloaders:
        raise RuntimeError("No client dataloaders were built. Check data_dir, feature regexes, and target_col.")

    assert common_features is not None
    log.info("Built %d client dataloaders", len(dataloaders))
    log.info("Total client samples=%d", sum(i.num_samples for i in infos))
    return dataloaders, infos, common_features


def cycle_next(iterator_dict: Dict[str, Iterable], dataloaders: Dict[str, DataLoader], client_id: str):
    try:
        return next(iterator_dict[client_id])
    except (StopIteration, KeyError):
        iterator_dict[client_id] = iter(dataloaders[client_id])
        return next(iterator_dict[client_id])


def flatten_grads(grads: List[torch.Tensor]) -> np.ndarray:
    return torch.cat([g.detach().cpu().flatten() for g in grads]).numpy()


def grad_norm(grads: List[torch.Tensor]) -> float:
    total = 0.0
    for g in grads:
        total += float(torch.sum(g.detach() ** 2).cpu())
    return math.sqrt(total)


def evaluate(model: nn.Module, dataloaders: Dict[str, DataLoader], device: torch.device, max_batches: int = 5) -> float:
    model.eval()
    loss_fn = nn.MSELoss(reduction="sum")
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for cid, dl in dataloaders.items():
            for b, batch in enumerate(dl):
                if b >= max_batches:
                    break
                x, y, _, _ = batch
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)
                total_loss += float(loss.detach().cpu())
                total_n += x.shape[0]
    model.train()
    return total_loss / max(total_n, 1)


def run_fedsgd(args, dataloaders: Dict[str, DataLoader], features: List[str]) -> None:
    log = logging.getLogger("fl")
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        if args.device != "cpu":
            log.warning("Requested device=%s but it is unavailable; using CPU", args.device)
        device = torch.device("cpu")

    model = TinyForecastCNN(num_features=len(features), hidden=args.hidden, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    client_ids = list(dataloaders.keys())
    client_iters: Dict[str, Iterable] = {}

    log.info("Model=%s", model)
    log.info("Device=%s num_clients=%d features=%s", device, len(client_ids), features)

    if args.save_gradient_npz:
        Path(args.save_gradient_npz).parent.mkdir(parents=True, exist_ok=True)
        saved_rounds = []
        saved_clients = []
        saved_gradients = []
        saved_peak_values = []
        saved_peak_positions = []
    else:
        saved_rounds = saved_clients = saved_gradients = saved_peak_values = saved_peak_positions = None

    for rnd in range(1, args.rounds + 1):
        if args.client_fraction < 1.0:
            k = max(1, int(round(args.client_fraction * len(client_ids))))
            selected = random.sample(client_ids, k)
        else:
            selected = client_ids

        log.info("Round %d/%d started: selected_clients=%d", rnd, args.rounds, len(selected))
        client_grads: List[List[torch.Tensor]] = []
        client_weights: List[int] = []
        client_losses: List[float] = []
        client_grad_norms: List[float] = []

        for cid in selected:
            try:
                batch = cycle_next(client_iters, dataloaders, cid)
                x, y, peak_value, peak_pos = batch
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad(set_to_none=True)
                pred = model(x)
                loss = loss_fn(pred, y)
                if not torch.isfinite(loss):
                    log.warning("Round %d client %s: non-finite loss=%s; skipping", rnd, cid, loss.item())
                    continue
                loss.backward()

                grads = []
                for p in model.parameters():
                    if p.grad is None:
                        grads.append(torch.zeros_like(p, device="cpu"))
                    else:
                        grads.append(p.grad.detach().cpu().clone())

                gnorm = grad_norm(grads)
                client_grads.append(grads)
                client_weights.append(int(x.shape[0]))
                client_losses.append(float(loss.detach().cpu()))
                client_grad_norms.append(gnorm)

                log.debug(
                    "Round %d client %s: batch=%d loss=%.6f grad_norm=%.6f "
                    "peak_value_mean=%.4f peak_pos_mode=%s",
                    rnd, cid, x.shape[0], client_losses[-1], gnorm,
                    float(peak_value.float().mean().item()),
                    int(torch.mode(peak_pos.flatten()).values.item()) if peak_pos.numel() > 0 else -1,
                )

                if args.save_gradient_npz and (rnd % args.save_gradient_every == 0):
                    saved_rounds.append(rnd)
                    saved_clients.append(cid)
                    saved_gradients.append(flatten_grads(grads))
                    saved_peak_values.append(peak_value.numpy())
                    saved_peak_positions.append(peak_pos.numpy())
            except Exception:
                log.exception("Round %d client %s failed; skipping this client", rnd, cid)

        if not client_grads:
            log.error("Round %d: no valid client gradients; stopping", rnd)
            break

        total_weight = float(sum(client_weights))
        avg_grads: List[torch.Tensor] = []
        for param_i in range(len(client_grads[0])):
            weighted = sum(client_grads[j][param_i] * (client_weights[j] / total_weight) for j in range(len(client_grads)))
            avg_grads.append(weighted)

        optimizer.zero_grad(set_to_none=True)
        for p, g in zip(model.parameters(), avg_grads):
            p.grad = g.to(device)
        optimizer.step()

        mean_loss = float(np.average(client_losses, weights=client_weights))
        mean_gnorm = float(np.mean(client_grad_norms))
        log.info(
            "Round %d done: used_clients=%d examples=%d weighted_loss=%.6f mean_client_grad_norm=%.6f",
            rnd, len(client_grads), sum(client_weights), mean_loss, mean_gnorm,
        )

        if args.eval_every > 0 and rnd % args.eval_every == 0:
            eval_mse = evaluate(model, dataloaders, device=device, max_batches=args.eval_max_batches)
            log.info("Round %d eval_mse_over_limited_batches=%.6f", rnd, eval_mse)

    if args.model_out:
        out = Path(args.model_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "features": features, "args": vars(args)}, out)
        log.info("Saved model checkpoint to %s", out)

    if args.save_gradient_npz and saved_gradients:
        # Object arrays for per-batch peak arrays because last batch sizes may differ.
        np.savez_compressed(
            args.save_gradient_npz,
            round=np.asarray(saved_rounds),
            client=np.asarray(saved_clients),
            gradient=np.stack(saved_gradients, axis=0),
            peak_value=np.asarray(saved_peak_values, dtype=object),
            peak_position=np.asarray(saved_peak_positions, dtype=object),
        )
        log.info("Saved gradient/peak records to %s", args.save_gradient_npz)


def parse_args():
    p = argparse.ArgumentParser(description="Barebones household-as-client FL simulation for IDEAL forecasting")
    p.add_argument("--data-dir", help="Directory containing IDEAL home*.csv.gz sensor files", default="/scratch/ejk5818/ts-inverse/data/IDEAL_household_energy/household_sensors/sensordata")
    p.add_argument("--interface-path", default="/scratch/ejk5818/ts-inverse/data/IDEAL_household_energy/coding/coding/API/IdealDataInterface.py", help="Path to IdealDataInterface.py")
    p.add_argument("--feature-groups-json", default=None, help="Optional JSON file defining feature groups")
    p.add_argument("--target-col", default="electricity", help="Forecast target column after feature grouping")

    p.add_argument("--resample-rule", default="30min", help="Pandas resample rule, e.g., 15min, 30min, 1H")
    p.add_argument("--input-steps", type=int, default=48, help="Number of past resampled steps in model input")
    p.add_argument("--horizon-steps", type=int, default=1, help="Forecast horizon in resampled steps")
    p.add_argument("--min-non-nan-fraction", type=float, default=0.2, help="Minimum non-NaN fraction required to keep a feature")
    p.add_argument("--max-interpolate-gap", type=int, default=4, help="Max consecutive resampled rows to fill")
    p.add_argument("--max-sensors-per-group", type=int, default=20, help="Limit sensors loaded per feature group per home; <=0 means no limit")
    p.add_argument("--min-samples-per-client", type=int, default=20, help="Skip clients with fewer windows")
    p.add_argument("--max-clients", type=int, default=0, help="Limit number of households for debugging; 0 means all")
    p.add_argument("--no-standardize", action="store_true", help="Disable per-client feature standardization")

    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--client-fraction", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--eval-max-batches", type=int, default=3)

    p.add_argument("--save-gradient-npz", default=None, help="Optional .npz path to save per-client gradients and peak labels")
    p.add_argument("--save-gradient-every", type=int, default=1)
    p.add_argument("--model-out", default=None)

    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--log-file", default="log.txt", help="Optional log file path; default=log.txt")
    p.add_argument("--verbose", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file, args.verbose)
    log = logging.getLogger("main")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    log.info("Args: %s", vars(args))

    try:
        dataloaders, infos, features = build_federated_dataloaders(args)
        log.info("Client summary:")
        for info in infos[:50]:
            log.info(
                "  home=%s samples=%d batches=%d dates=[%s, %s] features=%s",
                info.home_id, info.num_samples, info.num_batches, info.date_min, info.date_max, info.features,
            )
        if len(infos) > 50:
            log.info("  ... %d more clients not printed", len(infos) - 50)
        run_fedsgd(args, dataloaders, features)
    except Exception:
        log.exception("Fatal failure in FL simulation")
        raise


if __name__ == "__main__":
    main()
