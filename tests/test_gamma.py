"""Unit tests for the Gamma process degradation model (gamma_process.py)."""

import sys
from itertools import pairwise
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from gamma_process import (
    _fit_rounded_nh_gamma,
    build_wear_windows,
)


class TestBuildWearWindows:
    def test_rejects_invalid_window_size(self):
        import pandas as pd

        with pytest.raises(ValueError):
            build_wear_windows(pd.DataFrame(), window_size=1)

    def test_production_windows_retain_rounded_zero_increments(self):
        import pandas as pd

        index = pd.DataFrame({
            "machine": [1] * 7,
            "tool": [1] * 7,
            "run": np.arange(1, 8),
            "cumulated_tool_contact_time": np.arange(7, dtype=float),
            "wear": [0, 0, 0, 1, 1, 1, 2],
        })
        out = build_wear_windows(index, window_size=2)
        np.testing.assert_array_equal(
            out["M1"]["dv"], np.array([0.0, 1.0, 1.0])
        )


class TestRoundedNonhomogeneousFit:
    def test_fits_quantized_multi_tool_data(self):
        rng = np.random.default_rng(123)
        t_start = []
        t_end = []
        tool_ids = []
        observed = []
        beta = 0.6
        for tool, (eta, power) in enumerate(
            [(22.0, 0.75), (28.0, 0.65), (18.0, 0.90)], start=1
        ):
            edges = np.linspace(10.0, 3500.0, 45)
            for left, right in pairwise(edges):
                shape = eta * (
                    (right / 1000.0) ** power
                    - (left / 1000.0) ** power
                )
                latent = rng.gamma(shape=shape, scale=1.0 / beta)
                observed.append(float(np.rint(latent)))
                t_start.append(left)
                t_end.append(right)
                tool_ids.append(tool)

        fitted = _fit_rounded_nh_gamma(
            np.asarray(observed),
            np.asarray(t_start),
            np.asarray(t_end),
            np.asarray(tool_ids),
        )
        assert fitted["model"] == "nonhomogeneous_gamma_rounded"
        assert len(fitted["components"]) == 3
        assert fitted["n_windows"] == len(observed)
        assert fitted["beta"] > 0
