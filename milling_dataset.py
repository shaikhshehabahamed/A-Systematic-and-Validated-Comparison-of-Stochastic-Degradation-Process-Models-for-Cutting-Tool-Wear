"""Utilities for the multivariate milling tool-wear dataset.

Denkena, Klemme, and Stiehl, "Multivariate time series data of milling
processes with varying tool wear and machine tools", Mendeley Data v3,
DOI: 10.17632/zpxs87bjt8.3.

The canonical labels live in filelist.csv. Each HDF5 file contains one
milling run: force signals at 25 kHz (dynamometer) and controller signals
at 500 Hz. This module validates the index, verifies HDF5 labels, and
extracts compact signal statistics normalised per machine type.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DATASET_DIR_NAME = (
    "Multivariate time series data of milling processes with varying "
    "tool wear and machine tools"
)
FILELIST_NAME = "filelist.csv"
DATASET_DOI = "10.17632/zpxs87bjt8.3"
EXPECTED_COLUMNS = [
    "filename",
    "machine",
    "tool",
    "run",
    "cumulated_tool_contact_time",
    "wear",
]
SIGNAL_FEATURE_NAMES = [
    "sensor_force_rms",
    "sensor_force_std",
    "spindle_torque_rms",
    "position_deviation_rms",
    "run_duration_s",
]

# Machine groups: machines with similar degradation physics are grouped together.
# M1 and M3 have similar wear rates (~0.25 µm/run); M2 is lighter (~0.16 µm/run).
MACHINE_GROUPS = {
    1: "MG_heavy",   # M1
    2: "MG_light",   # M2
    3: "MG_heavy",   # M3
}


def find_dataset_dir(project_dir: str | os.PathLike, explicit: str | None = None) -> Path:
    """Resolve the dataset directory and verify that filelist.csv exists."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_dir = os.environ.get("DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.extend([
        Path(project_dir) / DATASET_DIR_NAME,
        Path(project_dir).parent / DATASET_DIR_NAME,  # sibling of project
        Path(project_dir) / "data",                    # data/ sub-folder
        Path.cwd() / DATASET_DIR_NAME,
    ])
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / FILELIST_NAME).is_file():
            return candidate
    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Milling dataset not found. Expected {FILELIST_NAME} under: {searched}\n"
        "Pass --dataset-dir or set the DATA_DIR environment variable."
    )


def file_sha256(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def h5_inventory_sha256(dataset_dir: str | os.PathLike, filenames: Iterable[str]) -> str:
    """Hash the ordered HDF5 filename/size inventory without reading the files."""
    dataset_dir = Path(dataset_dir)
    digest = hashlib.sha256()
    for name in sorted(str(v) for v in filenames):
        size = (dataset_dir / name).stat().st_size
        digest.update(f"{name}\t{size}\n".encode("utf-8"))
    return digest.hexdigest()


def load_file_index(
    dataset_dir: str | os.PathLike,
    *,
    require_h5_files: bool = True,
) -> pd.DataFrame:
    """Load and validate the dataset index (one row per HDF5 milling run)."""
    dataset_dir = Path(dataset_dir)
    index_path = dataset_dir / FILELIST_NAME
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing dataset index: {index_path}")

    df = pd.read_csv(index_path)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{FILELIST_NAME} missing columns: {missing}")
    df = df[EXPECTED_COLUMNS].copy()

    if df.empty:
        raise ValueError(f"{FILELIST_NAME} contains no runs.")
    if df.isna().any().any():
        raise ValueError(f"{FILELIST_NAME} contains null values.")
    if df.duplicated().any():
        raise ValueError(f"{FILELIST_NAME} contains duplicate rows.")
    if df["filename"].duplicated().any():
        raise ValueError(f"{FILELIST_NAME} contains duplicate filenames.")

    for col in ("machine", "tool", "run", "cumulated_tool_contact_time", "wear"):
        df[col] = pd.to_numeric(df[col], errors="raise")
    for col in ("machine", "tool", "run", "wear"):
        vals = df[col].to_numpy(float)
        if not np.all(np.equal(vals, np.floor(vals))):
            raise ValueError(f"{col} must be integer-valued.")
        df[col] = df[col].astype(int)
    df["cumulated_tool_contact_time"] = df["cumulated_tool_contact_time"].astype(float)

    if (df["machine"] <= 0).any() or (df["tool"] <= 0).any() or (df["run"] <= 0).any():
        raise ValueError("machine, tool, run must be positive.")
    if (df["cumulated_tool_contact_time"] < 0).any() or (df["wear"] < 0).any():
        raise ValueError("contact time and wear must be non-negative.")

    tool_machine_counts = df.groupby("tool")["machine"].nunique()
    if (tool_machine_counts != 1).any():
        bad = tool_machine_counts[tool_machine_counts != 1].index.tolist()
        raise ValueError(f"Tools mapped to more than one machine: {bad}")

    for (machine, tool), group in df.sort_values(["machine", "tool", "run"]).groupby(
        ["machine", "tool"], sort=False
    ):
        runs = group["run"].to_numpy()
        contact = group["cumulated_tool_contact_time"].to_numpy(float)
        wear = group["wear"].to_numpy(float)
        if not np.array_equal(runs, np.arange(1, len(runs) + 1)):
            raise ValueError(f"Non-contiguous run indices for M{machine} T{tool}.")
        if np.any(np.diff(contact) <= 0):
            raise ValueError(f"Non-increasing contact time for M{machine} T{tool}.")
        if np.any(np.diff(wear) < 0):
            raise ValueError(f"Non-monotone wear for M{machine} T{tool}.")

    if require_h5_files:
        missing_h5 = [n for n in df["filename"] if not (dataset_dir / str(n)).is_file()]
        if missing_h5:
            raise FileNotFoundError(
                f"{len(missing_h5)} indexed HDF5 files missing; examples: {missing_h5[:5]}"
            )

    df = df.sort_values(["machine", "tool", "run"]).reset_index(drop=True)
    df.attrs["dataset_dir"] = str(dataset_dir.resolve())
    return df


def _scalar(dataset) -> float:
    v = np.asarray(dataset[()]).reshape(-1)
    if v.size != 1:
        raise ValueError(f"Expected scalar HDF5 label, got shape {dataset.shape}")
    return float(v[0])


def validate_hdf5_labels(
    dataset_dir: str | os.PathLike,
    index: pd.DataFrame,
    *,
    mode: str = "sample",
    sample_size: int = 30,
    seed: int = 2026,
) -> dict:
    """Cross-check HDF5 labels against filelist.csv."""
    if mode not in {"sample", "all"}:
        raise ValueError("mode must be 'sample' or 'all'")
    import h5py

    dataset_dir = Path(dataset_dir)
    if mode == "all":
        selected = index
    else:
        rng = np.random.default_rng(seed)
        keep = set()
        for _, grp in index.groupby(["machine", "tool"]):
            keep.add(int(grp.index[0]))
            keep.add(int(grp.index[-1]))
        remaining = np.array([i for i in index.index if i not in keep], dtype=int)
        n_extra = min(max(0, int(sample_size) - len(keep)), len(remaining))
        if n_extra:
            keep.update(rng.choice(remaining, n_extra, replace=False).tolist())
        selected = index.loc[sorted(keep)]

    failures = []
    expected_labels = {
        "machine": "machine",
        "tool": "tool",
        "run": "run",
        "cumulated_tool_contact_time": "cumulated_tool_contact_time",
        "wear": "wear",
    }
    for row in selected.itertuples(index=False):
        path = dataset_dir / row.filename
        try:
            with h5py.File(path, "r") as fh:
                labels = fh["labels"]
                for col, h5_name in expected_labels.items():
                    observed = _scalar(labels[h5_name])
                    target = float(getattr(row, col))
                    if not np.isclose(observed, target, rtol=0.0, atol=1e-6):
                        failures.append({
                            "filename": row.filename,
                            "label": col,
                            "index": target,
                            "hdf5": observed,
                        })
        except Exception as exc:
            failures.append({"filename": row.filename, "error": repr(exc)})

    if failures:
        raise ValueError(
            f"HDF5/index validation failed for {len(failures)} checks; "
            f"examples: {failures[:5]}"
        )
    return {"mode": mode, "files_checked": len(selected), "failures": 0}


def _read_vector(group, name: str) -> np.ndarray:
    if name not in group:
        return np.empty(0, dtype=float)
    return np.asarray(group[name][()], dtype=float).reshape(-1)


def _resultant_rms(group, names: Iterable[str]) -> float:
    arrays = [_read_vector(group, name) for name in names]
    arrays = [a for a in arrays if a.size]
    if not arrays:
        return float("nan")
    n = min(len(a) for a in arrays)
    squared = np.zeros(n, dtype=float)
    for a in arrays:
        squared += np.square(a[:n])
    return float(np.sqrt(np.mean(squared)))


def extract_timeseries_features(path: str | os.PathLike) -> dict:
    """Extract compact signal statistics from one HDF5 milling run."""
    import h5py

    with h5py.File(path, "r") as fh:
        sensor = fh["signals_sensor"]
        machine = fh["signals_machine"]

        force_axes = [
            _read_vector(sensor, "force_sensor_x"),
            _read_vector(sensor, "force_sensor_y"),
            _read_vector(sensor, "force_sensor_z"),
        ]
        force_axes = [a for a in force_axes if a.size]
        if not force_axes:
            raise ValueError(f"No dynamometer force signals in {path}")
        n_force = min(len(a) for a in force_axes)
        force_matrix = np.vstack([a[:n_force] for a in force_axes])

        time_machine = _read_vector(machine, "time_machine")

        features = {
            "sensor_force_rms": float(
                np.sqrt(np.mean(np.sum(np.square(force_matrix), axis=0)))
            ),
            "sensor_force_std": float(np.mean(np.std(force_matrix, axis=1))),
            "spindle_torque_rms": float(
                np.sqrt(np.mean(np.square(_read_vector(machine, "torque_spindle"))))
            ),
            "position_deviation_rms": _resultant_rms(
                machine, [
                    "position_control_deviation_axis_x",
                    "position_control_deviation_axis_y",
                ],
            ),
            "run_duration_s": float(time_machine[-1] - time_machine[0])
            if time_machine.size >= 2 else float("nan"),
        }

    if not all(np.isfinite(features[n]) for n in SIGNAL_FEATURE_NAMES):
        bad = {k: v for k, v in features.items() if not np.isfinite(v)}
        raise ValueError(f"Non-finite signal features in {path}: {bad}")
    return features


def load_or_build_feature_table(
    dataset_dir: str | os.PathLike,
    index: pd.DataFrame,
    cache_path: str | os.PathLike,
    *,
    rebuild: bool = False,
    max_files: int | None = None,
) -> pd.DataFrame:
    """Load a complete feature cache or build it by reading HDF5 runs."""
    dataset_dir = Path(dataset_dir)
    cache_path = Path(cache_path)
    meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")
    selected = index if max_files is None else index.head(int(max_files))
    index_hash = file_sha256(dataset_dir / FILELIST_NAME)
    inv_hash = h5_inventory_sha256(dataset_dir, index["filename"])

    if cache_path.is_file() and meta_path.is_file() and not rebuild:
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        cached = pd.read_csv(cache_path)
        required = ["filename", *SIGNAL_FEATURE_NAMES]
        ok = (
            meta.get("datasetDOI") == DATASET_DOI
            and meta.get("filelistSha256") == index_hash
            and meta.get("h5InventorySha256") == inv_hash
            and meta.get("featureNames") == SIGNAL_FEATURE_NAMES
            and int(meta.get("rows", -1)) == len(index)
            and len(cached) == len(index)
        )
        if ok and all(c in cached.columns for c in required):
            merged = selected.merge(
                cached[required], on="filename", how="left", validate="one_to_one"
            )
            if not merged[SIGNAL_FEATURE_NAMES].isna().any().any():
                return merged

    rows = []
    total = len(selected)
    for pos, row in enumerate(selected.itertuples(index=False), start=1):
        feats = extract_timeseries_features(dataset_dir / row.filename)
        rows.append({"filename": row.filename, **feats})
        if pos == 1 or pos % 250 == 0 or pos == total:
            print(f"  HDF5 feature extraction: {pos}/{total}")

    feat_df = pd.DataFrame(rows)
    merged = selected.merge(feat_df, on="filename", how="left", validate="one_to_one")
    if max_files is None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        feat_df.to_csv(cache_path, index=False)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump({
                "datasetDOI": DATASET_DOI,
                "filelistSha256": index_hash,
                "h5InventorySha256": inv_hash,
                "featureNames": SIGNAL_FEATURE_NAMES,
                "rows": len(feat_df),
            }, fh, indent=2)
    return merged


def dataset_summary(index: pd.DataFrame) -> dict:
    """Return dataset quality and tool-life statistics."""
    endpoints = (
        index.sort_values(["machine", "tool", "run"])
        .groupby(["machine", "tool"], as_index=False)
        .tail(1)
        .copy()
    )
    endpoints["tool_life_hours"] = endpoints["cumulated_tool_contact_time"] / 3600.0
    terminal_wear = endpoints["wear"].to_numpy(float)
    return {
        "runs": len(index),
        "machines": int(index["machine"].nunique()),
        "tools": int(index["tool"].nunique()),
        "wear_min_um": float(index["wear"].min()),
        "wear_max_um": float(index["wear"].max()),
        "median_terminal_contact_s": float(
            endpoints["cumulated_tool_contact_time"].median()
        ),
        "median_tool_life_h": float(endpoints["tool_life_hours"].median()),
        "terminal_wear_min_um": float(np.min(terminal_wear)),
        "terminal_wear_max_um": float(np.max(terminal_wear)),
        "terminal_wear_median_um": float(np.median(terminal_wear)),
        "rul_target_definition": (
            "remaining cumulative tool-contact seconds to the recorded "
            "trajectory endpoint; end-of-life ≈ 150 µm"
        ),
    }
