"""Manual hyperparameter search for Boulevard's release estimators.

This is not a pytest test.  Run from the repo root:

    python tests/diagnostics/booster_hyperparameter_search.py --quick

The search objective prioritizes signal confidence-interval coverage:

- PASS: coverage >= 0.90
- OK:   coverage >= 0.80
- LOW:  coverage < 0.80

If Optuna is installed, the script uses Optuna.  Otherwise it falls back to
random search over the same parameter ranges.

Each run saves complete successful trial results to CSV plus best-result
metadata to JSON.  Use ``--no-save`` to disable this for quick smoke checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

import boulevard as bd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

try:
    import optuna
except ImportError:  # pragma: no cover - optional diagnostic dependency
    optuna = None


@dataclass(frozen=True)
class Scenario:
    name: str
    n_features: int
    default_n_samples: int
    signal: Callable[[np.ndarray], np.ndarray]


@dataclass
class Result:
    score: float
    algorithm: str
    scenario: str
    params: dict[str, Any]
    rmse: float
    coverage: float
    pi_coverage: float
    median_ci_width: float
    fit_seconds: float
    prep_seconds: float
    n_samples: int


def smooth_1d(X: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * X[:, 0])


def additive_3d(X: np.ndarray) -> np.ndarray:
    return (
        np.sin(2 * np.pi * X[:, 0])
        + 0.8 * np.cos(2 * np.pi * X[:, 1])
        + 0.5 * (X[:, 2] - 0.5) ** 2
    )


def interaction_5d(X: np.ndarray) -> np.ndarray:
    return (
        np.sin(2 * np.pi * X[:, 0])
        + 0.8 * np.cos(2 * np.pi * X[:, 1])
        + 0.5 * (X[:, 2] - 0.5) ** 2
        + 0.35 * X[:, 0] * X[:, 1]
        + 0.25 * np.sin(4 * np.pi * X[:, 3])
    )


SCENARIOS = {
    "smooth_1d": Scenario("smooth_1d", 1, 1000, smooth_1d),
    "additive_3d": Scenario("additive_3d", 3, 2500, additive_3d),
    "interaction_5d": Scenario("interaction_5d", 5, 5000, interaction_5d),
}


def coverage(target: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((target >= lower) & (target <= upper)))


def score_result(
    *,
    rmse: float,
    coverage_value: float,
    median_ci_width: float,
    min_coverage: float,
    target_coverage: float,
) -> float:
    soft_gap = max(0.0, target_coverage - coverage_value)
    hard_gap = max(0.0, min_coverage - coverage_value)
    return rmse + 0.6 * soft_gap + 2.0 * hard_gap + 0.01 * median_ci_width


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    return sum((idx + 1) * ord(char) for idx, char in enumerate(text)) % 2**32


def make_data(
    scenario: Scenario,
    *,
    n_samples: int,
    noise_sd: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n_samples, scenario.n_features))
    signal = scenario.signal(X)
    y = signal + rng.normal(scale=noise_sd, size=n_samples)
    return X, y, signal


def evaluate(
    algorithm: str,
    scenario: Scenario,
    params: dict[str, Any],
    args: argparse.Namespace,
) -> Result:
    n_samples = args.n_samples or scenario.default_n_samples
    X, y, signal = make_data(
        scenario,
        n_samples=n_samples,
        noise_sd=args.noise_sd,
        seed=args.seed,
    )
    X_train, X_hold, y_train, y_hold, _signal_train, signal_hold = train_test_split(
        X,
        y,
        signal,
        test_size=0.4,
        random_state=args.seed,
    )
    X_calib, X_test, y_calib, y_test, _signal_calib, signal_test = train_test_split(
        X_hold,
        y_hold,
        signal_hold,
        test_size=0.5,
        random_state=args.seed + 1,
    )

    model = make_model(algorithm, params, seed=args.seed)
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

    rmse = float(np.sqrt(np.mean((pred - signal_test) ** 2)))
    ci_coverage = coverage(signal_test, ci_lower, ci_upper)
    pi_coverage = coverage(y_test, pi_lower, pi_upper)
    median_ci_width = float(np.median(ci_upper - ci_lower))
    score = score_result(
        rmse=rmse,
        coverage_value=ci_coverage,
        median_ci_width=median_ci_width,
        min_coverage=args.min_coverage,
        target_coverage=args.target_coverage,
    )
    return Result(
        score=score,
        algorithm=algorithm,
        scenario=scenario.name,
        params=params,
        rmse=rmse,
        coverage=ci_coverage,
        pi_coverage=pi_coverage,
        median_ci_width=median_ci_width,
        fit_seconds=fit_seconds,
        prep_seconds=prep_seconds,
        n_samples=n_samples,
    )


def make_model(algorithm: str, params: dict[str, Any], *, seed: int) -> Any:
    if algorithm == "dropout":
        return bd.DropoutBooster(**params, random_state=seed)
    if algorithm == "parallel":
        return bd.ParallelBooster(
            **params,
            random_state=seed,
            n_jobs=1,
        )
    if algorithm == "explainable":
        return bd.ExplainableBooster(**params, random_state=seed)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def sample_params(algorithm: str, rng: np.random.Generator) -> dict[str, Any]:
    if algorithm == "dropout":
        return {
            "max_iter": int(rng.choice([200, 400, 700, 1000])),
            "learning_rate": float(rng.choice([0.8, 1.0, 1.5, 2.0, 3.0])),
            "dropout_rate": float(rng.choice([0.0, 0.05, 0.1, 0.2])),
            "subsample_rate": float(rng.choice([0.7, 0.8, 1.0])),
            "max_depth": int(rng.choice([6, 8, 10])),
            "max_leaf_nodes": int(rng.choice([64, 128, 256])),
            "min_samples_leaf": int(rng.choice([2, 4, 8])),
            "max_bins": int(rng.choice([64, 128, 255])),
        }
    if algorithm == "parallel":
        return {
            "n_rounds": int(rng.choice([20, 40, 70, 100])),
            "trees_per_round": int(rng.choice([4, 6, 8, 10])),
            "subsample_rate": float(rng.choice([0.7, 0.8, 1.0])),
            "max_depth": int(rng.choice([6, 8, 10])),
            "max_leaf_nodes": int(rng.choice([64, 128, 256])),
            "min_samples_leaf": int(rng.choice([2, 4, 8])),
            "max_bins": int(rng.choice([64, 128, 255])),
            "drop_first_round": bool(rng.choice([False, True])),
        }
    if algorithm == "explainable":
        return {
            "max_rounds": int(rng.choice([100, 160, 240, 320])),
            "max_bins": int(rng.choice([32, 64, 128])),
            "learning_rate": float(rng.choice([0.6, 0.8, 1.0, 1.5])),
            "subsample_rate": float(rng.choice([0.7, 0.8, 1.0])),
            "warmup_rounds": int(rng.choice([0, 10, 20, 40])),
            "max_depth": int(rng.choice([1, 2, 3, 4])),
            "min_samples_leaf": int(rng.choice([4, 8, 16])),
            "leave_one_out": bool(rng.choice([False, True])),
        }
    raise ValueError(f"Unknown algorithm: {algorithm}")


def suggest_params(algorithm: str, trial: Any) -> dict[str, Any]:
    if algorithm == "dropout":
        return {
            "max_iter": trial.suggest_categorical("max_iter", [200, 400, 700, 1000]),
            "learning_rate": trial.suggest_categorical(
                "learning_rate",
                [0.8, 1.0, 1.5, 2.0, 3.0],
            ),
            "dropout_rate": trial.suggest_categorical(
                "dropout_rate",
                [0.0, 0.05, 0.1, 0.2],
            ),
            "subsample_rate": trial.suggest_categorical(
                "subsample_rate",
                [0.7, 0.8, 1.0],
            ),
            "max_depth": trial.suggest_categorical("max_depth", [6, 8, 10]),
            "max_leaf_nodes": trial.suggest_categorical(
                "max_leaf_nodes",
                [64, 128, 256],
            ),
            "min_samples_leaf": trial.suggest_categorical(
                "min_samples_leaf",
                [2, 4, 8],
            ),
            "max_bins": trial.suggest_categorical("max_bins", [64, 128, 255]),
        }
    if algorithm == "parallel":
        return {
            "n_rounds": trial.suggest_categorical("n_rounds", [20, 40, 70, 100]),
            "trees_per_round": trial.suggest_categorical(
                "trees_per_round",
                [4, 6, 8, 10],
            ),
            "subsample_rate": trial.suggest_categorical(
                "subsample_rate",
                [0.7, 0.8, 1.0],
            ),
            "max_depth": trial.suggest_categorical("max_depth", [6, 8, 10]),
            "max_leaf_nodes": trial.suggest_categorical(
                "max_leaf_nodes",
                [64, 128, 256],
            ),
            "min_samples_leaf": trial.suggest_categorical(
                "min_samples_leaf",
                [2, 4, 8],
            ),
            "max_bins": trial.suggest_categorical("max_bins", [64, 128, 255]),
            "drop_first_round": trial.suggest_categorical(
                "drop_first_round",
                [False, True],
            ),
        }
    if algorithm == "explainable":
        return {
            "max_rounds": trial.suggest_categorical(
                "max_rounds",
                [100, 160, 240, 320],
            ),
            "max_bins": trial.suggest_categorical("max_bins", [32, 64, 128]),
            "learning_rate": trial.suggest_categorical(
                "learning_rate",
                [0.6, 0.8, 1.0, 1.5],
            ),
            "subsample_rate": trial.suggest_categorical(
                "subsample_rate",
                [0.7, 0.8, 1.0],
            ),
            "warmup_rounds": trial.suggest_categorical(
                "warmup_rounds",
                [0, 10, 20, 40],
            ),
            "max_depth": trial.suggest_categorical("max_depth", [1, 2, 3, 4]),
            "min_samples_leaf": trial.suggest_categorical(
                "min_samples_leaf",
                [4, 8, 16],
            ),
            "leave_one_out": trial.suggest_categorical(
                "leave_one_out",
                [False, True],
            ),
        }
    raise ValueError(f"Unknown algorithm: {algorithm}")


def run_search(
    algorithm: str,
    scenario: Scenario,
    args: argparse.Namespace,
) -> tuple[Result | None, list[Result]]:
    best: Result | None = None
    trial_results: list[Result] = []
    rng = np.random.default_rng(stable_seed(algorithm, scenario.name, args.seed))

    def record_result(result: Result) -> None:
        nonlocal best
        trial_results.append(result)
        if best is None or result.score < best.score:
            best = result

    if optuna is not None and not args.random_search:
        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def objective(trial: Any) -> float:
            try:
                result = evaluate(
                    algorithm,
                    scenario,
                    suggest_params(algorithm, trial),
                    args,
                )
            except Exception as exc:  # pragma: no cover - diagnostic robustness
                trial.set_user_attr("error", repr(exc))
                return float("inf")
            trial.set_user_attr("result", result)
            record_result(result)
            return result.score

        study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
        return best, trial_results

    for _ in range(args.trials):
        try:
            result = evaluate(
                algorithm,
                scenario,
                sample_params(algorithm, rng),
                args,
            )
        except Exception:
            continue
        record_result(result)
    return best, trial_results


def status(result: Result, args: argparse.Namespace) -> str:
    if result.coverage >= args.target_coverage:
        return "PASS"
    if result.coverage >= args.min_coverage:
        return "OK"
    return "LOW"


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def result_record(result: Result, args: argparse.Namespace, trial_index: int) -> dict:
    return {
        "trial_index": trial_index,
        "algorithm": result.algorithm,
        "scenario": result.scenario,
        "status": status(result, args),
        "score": result.score,
        "rmse": result.rmse,
        "coverage": result.coverage,
        "pi_coverage": result.pi_coverage,
        "median_ci_width": result.median_ci_width,
        "fit_seconds": result.fit_seconds,
        "prep_seconds": result.prep_seconds,
        "n_samples": result.n_samples,
        "params": result.params,
    }


def save_results(
    *,
    best_results: list[Result],
    trial_results: list[Result],
    args: argparse.Namespace,
    backend: str,
) -> Path:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    records = [
        result_record(result, args, trial_index)
        for trial_index, result in enumerate(trial_results)
    ]
    csv_path = output_dir / "all_trials.csv"
    with csv_path.open("w", newline="") as file:
        fieldnames = [
            "trial_index",
            "algorithm",
            "scenario",
            "status",
            "score",
            "rmse",
            "coverage",
            "pi_coverage",
            "median_ci_width",
            "fit_seconds",
            "prep_seconds",
            "n_samples",
            "params_json",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in fieldnames if key != "params_json"}
            row["params_json"] = json.dumps(record["params"], sort_keys=True)
            writer.writerow(row)

    summary = {
        "backend": backend,
        "arguments": vars(args),
        "best_results": [
            result_record(result, args, trial_index)
            for trial_index, result in enumerate(best_results)
        ],
        "all_trials_csv": str(csv_path),
    }
    json_path = output_dir / "best_results.json"
    with json_path.open("w") as file:
        json.dump(summary, file, indent=2, sort_keys=True)

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithms", default="dropout,parallel,explainable")
    parser.add_argument("--scenarios", default="smooth_1d,additive_3d,interaction_5d")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help=(
            "Override scenario-specific sample sizes. By default, uses "
            "1000 for smooth_1d, 2500 for additive_3d, and 5000 for "
            "interaction_5d."
        ),
    )
    parser.add_argument("--noise-sd", type=float, default=0.2)
    parser.add_argument("--min-coverage", type=float, default=0.8)
    parser.add_argument("--target-coverage", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-search", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="tests/diagnostics/results",
        help="Directory for saved CSV/JSON search results.",
    )
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.algorithms = "dropout,parallel,explainable"
        args.scenarios = "smooth_1d,additive_3d"
        args.trials = 2
        args.n_samples = 400

    algorithms = parse_csv(args.algorithms)
    scenarios = [SCENARIOS[name] for name in parse_csv(args.scenarios)]
    backend = "optuna" if optuna is not None and not args.random_search else "random"

    print("Boulevard empirical hyperparameter search")
    print(f"backend: {backend}")
    print(f"algorithms: {algorithms}")
    print(f"scenarios: {[scenario.name for scenario in scenarios]}")
    print(f"trials per algorithm/scenario: {args.trials}")
    if args.n_samples is None:
        sample_sizes = {
            scenario.name: scenario.default_n_samples for scenario in scenarios
        }
        print(f"n_samples by scenario: {sample_sizes}")
    else:
        print(f"n_samples override: {args.n_samples}")
    print(f"noise_sd: {args.noise_sd}")
    print(
        f"coverage status: PASS >= {args.target_coverage}, "
        f"OK >= {args.min_coverage}"
    )

    all_results: list[Result] = []
    all_trial_results: list[Result] = []
    for scenario in scenarios:
        for algorithm in algorithms:
            print("\n" + "=" * 78)
            print(f"searching algorithm={algorithm}, scenario={scenario.name}")
            result, trial_results = run_search(algorithm, scenario, args)
            all_trial_results.extend(trial_results)
            if result is None:
                print("no successful trial")
                continue
            all_results.append(result)
            print_result(result, args)

    print("\n" + "=" * 78)
    print("Summary")
    for result in all_results:
        print(
            f"{result.scenario:15s} {result.algorithm:11s} "
            f"{status(result, args):4s} "
            f"n={result.n_samples:<5d} "
            f"coverage={result.coverage:.3f} "
            f"rmse={result.rmse:.3f} "
            f"ci_width={result.median_ci_width:.3f}"
        )

    if not args.no_save:
        output_dir = save_results(
            best_results=all_results,
            trial_results=all_trial_results,
            args=args,
            backend=backend,
        )
        print(f"\nSaved search results to: {output_dir}")


def print_result(result: Result, args: argparse.Namespace) -> None:
    print(f"status: {status(result, args)}")
    print(f"score: {result.score:.6f}")
    print(f"RMSE vs signal: {result.rmse:.6f}")
    print(f"95% signal CI coverage: {result.coverage:.6f}")
    print(f"95% PI coverage: {result.pi_coverage:.6f}")
    print(f"median CI width: {result.median_ci_width:.6f}")
    print(f"n_samples: {result.n_samples}")
    print(f"fit/prep seconds: {result.fit_seconds:.3f}/{result.prep_seconds:.3f}")
    print("params:")
    for key, value in result.params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
