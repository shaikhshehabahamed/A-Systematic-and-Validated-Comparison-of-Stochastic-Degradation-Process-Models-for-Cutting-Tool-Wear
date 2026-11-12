"""Kaplan-Meier restricted-mean time-to-failure and censored Weibull fitting.

Shared by the active degradation process-family comparison study
(``degradation_reliability_translation.py``) for both the real-data ground
truth (9 tool-life trajectories) and process-implied Monte Carlo lifetimes.
Input is a flexible list-of-dicts (``simTime``, ``firstFailureTime``) — not
tied to any particular simulation format.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special as sp_special
from scipy.optimize import minimize as sp_minimize


def _failure_time(result: dict) -> float | None:
    """Read the first failure time from an episode result, if any."""
    value = result.get("firstFailureTime")
    return None if value is None else float(value)


def _failure_observations(results: list[dict]) -> list[tuple[float, bool]]:
    observations = []
    for result in results:
        horizon = float(result.get("simTime", 0.0))
        failure_time = _failure_time(result)
        if failure_time is None:
            observations.append((horizon, False))
        else:
            observations.append((min(float(failure_time), horizon), True))
    return observations


def _km_event_curve(
    observations: list[tuple[float, bool]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return right-continuous Kaplan-Meier event-time coordinates."""
    if not observations:
        return np.array([], float), np.array([], float)

    event_times = sorted({time for time, event in observations if event})
    survival = 1.0
    times = [0.0]
    values = [1.0]
    for time in event_times:
        at_risk = sum(obs_time >= time for obs_time, _ in observations)
        failures = sum(
            event and obs_time == time for obs_time, event in observations
        )
        if at_risk <= 0 or failures <= 0:
            continue
        survival *= 1.0 - failures / at_risk
        times.append(float(time))
        values.append(float(survival))
    return np.asarray(times, float), np.asarray(values, float)


def compute_mttff(results: list[dict]) -> dict:
    """Kaplan-Meier restricted mean time to first failure.

    The reported value is the area under first-failure survival up to the
    largest episode horizon ``tau``. When all episodes are censored, the RMST
    equals ``tau`` while unrestricted MTTFF remains unidentifiable.
    """
    observations = _failure_observations(results)
    if not observations:
        return {
            "mttff": float("nan"),
            "rmst_tau": float("nan"),
            "tau": float("nan"),
            "n_observed": 0,
            "n_censored": 0,
            "all_censored": False,
            "km_t": [],
            "km_S": [],
        }

    tau = max(time for time, _ in observations)
    event_t, event_s = _km_event_curve(observations)
    rmst = 0.0
    previous_time = 0.0
    survival = 1.0
    for time, value in zip(event_t[1:], event_s[1:], strict=True):
        rmst += survival * (float(time) - previous_time)
        previous_time = float(time)
        survival = float(value)
    rmst += survival * max(0.0, tau - previous_time)

    n_observed = sum(event for _, event in observations)
    return {
        "mttff": float(rmst),
        "rmst_tau": float(rmst),
        "tau": float(tau),
        "n_observed": int(n_observed),
        "n_censored": int(len(observations) - n_observed),
        "all_censored": bool(n_observed == 0),
        "estimand": "restricted mean time to first failure up to tau",
        "km_t": event_t.tolist(),
        "km_S": event_s.tolist(),
    }


def _neg_ll_weibull_censored(
    log_params: np.ndarray,
    t_fail: np.ndarray,
    t_cens: np.ndarray,
) -> float:
    k = math.exp(float(log_params[0]))
    scale = math.exp(float(log_params[1]))
    log_likelihood = 0.0
    if len(t_fail):
        log_likelihood += len(t_fail) * (
            math.log(k) - k * math.log(scale)
        )
        log_likelihood += (k - 1.0) * float(np.sum(np.log(t_fail)))
        log_likelihood -= float(np.sum((t_fail / scale) ** k))
    if len(t_cens):
        log_likelihood -= float(np.sum((t_cens / scale) ** k))
    return -log_likelihood


def fit_weibull_mttff(results: list[dict]) -> dict:
    """Fit a two-parameter Weibull with right-censored likelihood."""
    observations = _failure_observations(results)
    t_fail = np.asarray(
        [time for time, event in observations if event], float
    )
    t_cens = np.asarray(
        [time for time, event in observations if not event], float
    )
    n_failures = len(t_fail)
    n_censored = len(t_cens)
    n_total = n_failures + n_censored

    if n_failures < 2:
        return {
            "shape": float("nan"),
            "scale": float("nan"),
            "loc": 0.0,
            "mean_weibull": float("nan"),
            "AIC": float("nan"),
            "BIC": float("nan"),
            "n_failures": n_failures,
            "n_censored": n_censored,
            "low_n": True,
        }

    all_times = np.concatenate([t_fail, t_cens]) if len(t_cens) else t_fail
    bounds = [
        (math.log(0.05), math.log(30.0)),
        (
            math.log(float(all_times.min()) / 10.0),
            math.log(float(all_times.max()) * 10.0),
        ),
    ]
    best = None
    for log_k_init in (0.0, 0.5, -0.3):
        result = sp_minimize(
            _neg_ll_weibull_censored,
            [log_k_init, math.log(float(all_times.mean()))],
            args=(t_fail, t_cens),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-9},
        )
        if best is None or result.fun < best.fun:
            best = result

    if best is None or not np.all(np.isfinite(best.x)):
        return {
            "shape": float("nan"),
            "scale": float("nan"),
            "loc": 0.0,
            "mean_weibull": float("nan"),
            "AIC": float("nan"),
            "BIC": float("nan"),
            "n_failures": n_failures,
            "n_censored": n_censored,
            "low_n": n_failures < 5,
        }

    shape = math.exp(float(best.x[0]))
    scale = math.exp(float(best.x[1]))
    log_likelihood = -float(best.fun)
    return {
        "shape": float(shape),
        "scale": float(scale),
        "loc": 0.0,
        "mean_weibull": float(
            scale * math.exp(sp_special.gammaln(1.0 + 1.0 / shape))
        ),
        "AIC": float(4.0 - 2.0 * log_likelihood),
        "BIC": float(2.0 * math.log(max(n_total, 1)) - 2.0 * log_likelihood),
        "n_failures": n_failures,
        "n_censored": n_censored,
        "low_n": bool(n_failures < 5),
    }


def mttff_bootstrap_ci(
    results: list[dict],
    blocks: list[list[dict]] | None = None,
    n_boot: int = 1000,
    ci_level: float = 0.95,
    rng_seed: int = 0,
) -> dict:
    """Bootstrap restricted MTTFF using blocks when available."""
    units = blocks if blocks is not None else [[result] for result in results]
    if len(units) < 2 or n_boot <= 0:
        return {
            "mttff_ci_lo": float("nan"),
            "mttff_ci_hi": float("nan"),
            "mttff_boot_std": float("nan"),
            "n_boot": 0,
        }
    rng = np.random.default_rng(rng_seed)
    values = []
    for _ in range(int(n_boot)):
        indices = rng.integers(0, len(units), size=len(units))
        sample = [
            result
            for index in indices
            for result in units[index]
        ]
        value = compute_mttff(sample)["mttff"]
        if np.isfinite(value):
            values.append(float(value))
    alpha = 1.0 - float(ci_level)
    return {
        "mttff_ci_lo": float(np.percentile(values, 100 * alpha / 2)),
        "mttff_ci_hi": float(np.percentile(values, 100 * (1 - alpha / 2))),
        "mttff_boot_std": float(np.std(values)),
        "n_boot": len(values),
        "resampling_unit": "experiment block" if blocks is not None else "episode",
    }
