"""Quick diagnostic for dimension-adaptive DropoutBooster settings.

This is not a pytest test. Run from the repo root:

    .venv/bin/python tests/diagnostics/dropout_dimension_config_check.py

The script compares a low-dimensional fixed configuration with a simple
dimension-adaptive rule of thumb. It is meant to test whether the proposed
scaling direction helps before promoting it to docs or presets.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.model_selection import train_test_split

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import boulevard as bd  # noqa: E402


@dataclass(frozen=True)
class Scenario:
    n_features: int
    n_samples: int
    signal: Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Result:
    scenario: str
    config_name: str
    n_features: int
    n_train: int
    rmse: float
    ci_coverage: float
    pi_coverage: float
    median_ci_width: float
    fit_seconds: float
    prep_seconds: float
    params: dict[str, int | float | None]


def make_signal(X: np.ndarray) -> np.ndarray:
    """Synthetic signal that gets harder as features are added."""
    y = np.sin(2 * np.pi * X[:, 0])
    if X.shape[1] >= 2:
        y += 0.8 * np.cos(2 * np.pi * X[:, 1])
        y += 0.35 * np.sin(2 * np.pi * (X[:, 0] + X[:, 1]))
    if X.shape[1] >= 3:
        y += 0.6 * (X[:, 2] - 0.5) ** 2
        y += 0.25 * X[:, 0] * X[:, 2]
    if X.shape[1] >= 4:
        y += 0.35 * np.sin(4 * np.pi * X[:, 3])
    if X.shape[1] >= 5:
        y += 0.25 * X[:, 1] * X[:, 4]
    return y


def fixed_low_dim_config(*, max_iter_cap: int) -> dict[str, int | float | None]:
    return {
        "max_iter": min(700, max_iter_cap),
        "learning_rate": 0.8,
        "dropout_rate": 0.3,
        "subsample_rate": 0.8,
        "max_depth": 6,
        "max_leaf_nodes": 64,
        "min_samples_leaf": 2,
        "max_bins": 64,
    }


def adaptive_dropout_config(
    *,
    n_train: int,
    n_features: int,
    max_iter_cap: int,
    max_leaf_cap: int,
) -> dict[str, int | float | None]:
    """Dimension-adaptive rule from the heuristic bias-variance calculation.

    Effective interaction dimension is approximated by min(d, 5). The partition
    count follows K ~ n_bag^(r / (r + 2)), leaf size follows
    m ~ n_bag^(2 / (r + 2)), and dropout weakens as r grows.
    """
    r = min(n_features, 5)
    subsample_rate = min(1.0, 0.75 + 0.05 * r)
    n_bag = max(1.0, subsample_rate * n_train)

    min_samples_leaf = max(2, int(math.ceil(0.25 * n_bag ** (2 / (r + 2)))))
    raw_leaf_nodes = int(math.ceil(4.0 * n_bag ** (r / (r + 2))))
    sample_leaf_cap = max(2, int(n_bag // min_samples_leaf))
    max_leaf_nodes = max(8, min(raw_leaf_nodes, sample_leaf_cap, max_leaf_cap))

    max_depth = int(math.ceil(math.log2(max_leaf_nodes))) + 1
    max_bins = min(
        255,
        max(64, int(math.ceil(3.0 * max_leaf_nodes ** (1 / r)))),
    )
    max_iter = min(max_iter_cap, max(200, int(4 * r * max_leaf_nodes)))

    return {
        "max_iter": max_iter,
        "learning_rate": 1.0,
        "dropout_rate": min(0.3, 0.3 / r),
        "subsample_rate": subsample_rate,
        "max_depth": max_depth,
        "max_leaf_nodes": max_leaf_nodes,
        "min_samples_leaf": min_samples_leaf,
        "max_bins": max_bins,
    }


def make_data(
    *,
    n_samples: int,
    n_features: int,
    signal_fn: Callable[[np.ndarray], np.ndarray],
    noise_sd: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n_samples, n_features))
    signal = signal_fn(X)
    y = signal + rng.normal(scale=noise_sd, size=n_samples)
    return X, y, signal


def coverage(target: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((target >= lower) & (target <= upper)))


def evaluate_config(
    *,
    scenario_name: str,
    config_name: str,
    scenario: Scenario,
    params: dict[str, int | float | None],
    noise_sd: float,
    seed: int,
) -> Result:
    X, y, signal = make_data(
        n_samples=scenario.n_samples,
        n_features=scenario.n_features,
        signal_fn=scenario.signal,
        noise_sd=noise_sd,
        seed=seed,
    )
    X_train, X_hold, y_train, y_hold, _signal_train, signal_hold = train_test_split(
        X,
        y,
        signal,
        test_size=0.4,
        random_state=seed,
    )
    X_calib, X_test, y_calib, y_test, _signal_calib, signal_test = train_test_split(
        X_hold,
        y_hold,
        signal_hold,
        test_size=0.5,
        random_state=seed + 1,
    )

    model = bd.DropoutBooster(**params, random_state=seed)

    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    model.prepare_inference(X_calib, y_calib)
    prep_seconds = time.perf_counter() - start

    pred = model.predict(X_test)
    ci_lower, ci_upper, _ = model.predict_intervals(
        X_test,
        level=0.95,
        mode="confidence",
    )
    pi_lower, pi_upper, _ = model.predict_intervals(
        X_test,
        level=0.95,
        mode="prediction",
    )

    return Result(
        scenario=scenario_name,
        config_name=config_name,
        n_features=scenario.n_features,
        n_train=X_train.shape[0],
        rmse=float(np.sqrt(np.mean((pred - signal_test) ** 2))),
        ci_coverage=coverage(signal_test, ci_lower, ci_upper),
        pi_coverage=coverage(y_test, pi_lower, pi_upper),
        median_ci_width=float(np.median(ci_upper - ci_lower)),
        fit_seconds=fit_seconds,
        prep_seconds=prep_seconds,
        params=params,
    )


def print_results(results: list[Result]) -> None:
    header = (
        "scenario",
        "config",
        "d",
        "n_train",
        "rmse",
        "ci_cov",
        "pi_cov",
        "ci_width",
        "fit_s",
        "prep_s",
        "max_iter",
        "leaves",
        "depth",
        "min_leaf",
        "bins",
        "dropout",
        "subsample",
    )
    print("\t".join(header))
    for result in results:
        p = result.params
        row = (
            result.scenario,
            result.config_name,
            str(result.n_features),
            str(result.n_train),
            f"{result.rmse:.4f}",
            f"{result.ci_coverage:.3f}",
            f"{result.pi_coverage:.3f}",
            f"{result.median_ci_width:.4f}",
            f"{result.fit_seconds:.2f}",
            f"{result.prep_seconds:.2f}",
            str(p["max_iter"]),
            str(p["max_leaf_nodes"]),
            str(p["max_depth"]),
            str(p["min_samples_leaf"]),
            str(p["max_bins"]),
            f"{float(p['dropout_rate']):.3f}",
            f"{float(p['subsample_rate']):.2f}",
        )
        print("\t".join(row))


def save_results(results: list[Result], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "config_name",
                "n_features",
                "n_train",
                "rmse",
                "ci_coverage",
                "pi_coverage",
                "median_ci_width",
                "fit_seconds",
                "prep_seconds",
                "params",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "scenario": result.scenario,
                    "config_name": result.config_name,
                    "n_features": result.n_features,
                    "n_train": result.n_train,
                    "rmse": result.rmse,
                    "ci_coverage": result.ci_coverage,
                    "pi_coverage": result.pi_coverage,
                    "median_ci_width": result.median_ci_width,
                    "fit_seconds": result.fit_seconds,
                    "prep_seconds": result.prep_seconds,
                    "params": result.params,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-sd", type=float, default=0.2)
    parser.add_argument("--max-iter-cap", type=int, default=400)
    parser.add_argument("--max-leaf-cap", type=int, default=128)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use larger sample sizes and caps for a slower diagnostic.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["smooth_1d", "mixed_3d", "mixed_5d"],
        help="Run only the selected scenarios.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("tests/diagnostics/results/dropout_dimension_config_check.csv"),
    )
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.full:
        scenarios = {
            "smooth_1d": Scenario(n_features=1, n_samples=1000, signal=make_signal),
            "mixed_3d": Scenario(n_features=3, n_samples=1800, signal=make_signal),
            "mixed_5d": Scenario(n_features=5, n_samples=2600, signal=make_signal),
        }
        if args.max_iter_cap == 400:
            args.max_iter_cap = 900
        if args.max_leaf_cap == 128:
            args.max_leaf_cap = 192
    else:
        scenarios = {
            "smooth_1d": Scenario(n_features=1, n_samples=600, signal=make_signal),
            "mixed_3d": Scenario(n_features=3, n_samples=900, signal=make_signal),
            "mixed_5d": Scenario(n_features=5, n_samples=1200, signal=make_signal),
        }
    if args.scenarios is not None:
        scenarios = {
            name: scenario
            for name, scenario in scenarios.items()
            if name in set(args.scenarios)
        }

    results: list[Result] = []
    for scenario_name, scenario in scenarios.items():
        fixed = fixed_low_dim_config(max_iter_cap=args.max_iter_cap)
        adaptive = adaptive_dropout_config(
            n_train=int(0.6 * scenario.n_samples),
            n_features=scenario.n_features,
            max_iter_cap=args.max_iter_cap,
            max_leaf_cap=args.max_leaf_cap,
        )
        for config_name, params in [
            ("fixed_low_dim", fixed),
            ("adaptive", adaptive),
        ]:
            print(
                f"running {scenario_name}/{config_name}: "
                f"n={scenario.n_samples}, d={scenario.n_features}, "
                f"max_iter={params['max_iter']}, leaves={params['max_leaf_nodes']}",
                flush=True,
            )
            results.append(
                evaluate_config(
                    scenario_name=scenario_name,
                    config_name=config_name,
                    scenario=scenario,
                    params=params,
                    noise_sd=args.noise_sd,
                    seed=args.seed,
                )
            )

    print_results(results)
    if not args.no_save:
        save_results(results, args.csv)
        print(f"\nsaved results to {args.csv}")


if __name__ == "__main__":
    main()
