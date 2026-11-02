from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_scalograms(scalogram_dir):
    """Return scalograms as (N, 1, H, W) float32 and the window metadata."""
    scalogram_dir = Path(scalogram_dir)
    images = np.load(scalogram_dir / "scalograms.npy").astype(np.float32)
    if images.ndim == 3:
        images = images[:, np.newaxis, :, :]
    metadata = pd.read_csv(scalogram_dir / "scalogram_metadata.csv", parse_dates=["start_datetime", "end_datetime"])
    if len(metadata) != len(images):
        raise ValueError(f"{len(images)} scalograms but {len(metadata)} metadata rows")
    return images, metadata


val_start = "2014-10-01"  # first El Nino month of the 2015-2016 episode (ONI >= 0.5), so the whole episode is in val
test_start = "2021-01-01"  # the 2023-2024 episode is in test


def date_split(dates, val_start=val_start, test_start=test_start):
    """Split day indices into train (before val_start), val (from val_start to the day before test_start) and test
    (from test_start). dates is a sequence of days in order."""
    dates = pd.to_datetime(pd.Series(dates)).dt.normalize().to_numpy()
    idx = np.arange(len(dates))
    val, test = np.datetime64(pd.Timestamp(val_start)), np.datetime64(pd.Timestamp(test_start))
    return idx[dates < val], idx[(dates >= val) & (dates < test)], idx[dates >= test]


def split_names(dates, val_start=val_start, test_start=test_start):
    """train, val or test for every day of dates."""
    dates = pd.to_datetime(pd.Series(dates)).dt.normalize()
    return np.where(dates < val_start, "train", np.where(dates < test_start, "val", "test"))


def chronological_split(n, train_ratio=0.6, val_ratio=0.2):
    """Split day indices 0..n-1 into consecutive train, val and test blocks."""
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return np.arange(0, train_end), np.arange(train_end, val_end), np.arange(val_end, n)


class ScalogramSequenceDataset(Dataset):
    """Runs of seq_len consecutive days inside one split. Returns (sequence, first day index)."""

    def __init__(self, images, indices, seq_len=30, stride=7):
        self.images = torch.from_numpy(images)
        indices = np.asarray(indices)
        if len(indices) < seq_len:
            raise ValueError(f"split has {len(indices)} days, fewer than seq_len={seq_len}")
        self.seq_len = seq_len
        self.starts = indices[0] + np.arange(0, len(indices) - seq_len + 1, stride)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        start = int(self.starts[i])
        return self.images[start:start + self.seq_len], start


def load_context(data_dir, metadata):
    """Daily frame aligned with the scalograms: real price, reservoir level, ONI and ENSO phase."""
    data_dir = Path(data_dir)
    days = pd.DataFrame({"date": metadata["start_datetime"].dt.normalize()})

    price = pd.read_csv(data_dir / "XM/precio_bolsa_diario_real.csv", parse_dates=["date"])
    reservoir = pd.read_csv(data_dir / "XM/volumen_util_pct.csv", parse_dates=["date"])
    oni = pd.read_csv(data_dir / "NOAA/oni.csv", parse_dates=["date"])

    days = days.merge(price[["date", "precio_real_cop_kwh"]], on="date", how="left")
    days = days.merge(reservoir, on="date", how="left")

    days["month"] = days["date"].dt.to_period("M")
    oni["month"] = oni["date"].dt.to_period("M")
    days = days.merge(oni[["month", "oni"]], on="month", how="left").drop(columns="month")
    days["enso_phase"] = np.where(days["oni"] >= 0.5, "El Nino", np.where(days["oni"] <= -0.5, "La Nina", "Neutral"))
    return days
