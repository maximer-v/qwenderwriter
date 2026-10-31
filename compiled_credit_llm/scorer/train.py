"""CPU XGBoost scorer: bounded search on validation, Platt or isotonic calibration on the calibration split,
locked test evaluation with bootstrap intervals. Test labels are read once, after every choice is frozen."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import expit
import sklearn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve
import xgboost as xgb
from xgboost import XGBClassifier

from ..data.uci import FEATURE_NAMES, TARGET, load_table

SEED = 20261030
BOOTSTRAP_REPLICATES = 1000
SEARCH = [
    {"max_depth": 3, "learning_rate": 0.03, "min_child_weight": 1, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 1, "reg_alpha": 0},
    {"max_depth": 3, "learning_rate": 0.05, "min_child_weight": 3, "subsample": 0.90, "colsample_bytree": 0.90, "reg_lambda": 3, "reg_alpha": 0},
    {"max_depth": 3, "learning_rate": 0.08, "min_child_weight": 5, "subsample": 0.90, "colsample_bytree": 0.90, "reg_lambda": 5, "reg_alpha": 0.1},
    {"max_depth": 4, "learning_rate": 0.03, "min_child_weight": 3, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 3, "reg_alpha": 0},
    {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 5, "subsample": 0.90, "colsample_bytree": 0.90, "reg_lambda": 5, "reg_alpha": 0.1},
    {"max_depth": 4, "learning_rate": 0.08, "min_child_weight": 8, "subsample": 0.90, "colsample_bytree": 0.90, "reg_lambda": 8, "reg_alpha": 0.2},
    {"max_depth": 5, "learning_rate": 0.03, "min_child_weight": 5, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 5, "reg_alpha": 0.1},
    {"max_depth": 5, "learning_rate": 0.05, "min_child_weight": 8, "subsample": 0.90, "colsample_bytree": 0.90, "reg_lambda": 8, "reg_alpha": 0.2},
    {"max_depth": 5, "learning_rate": 0.08, "min_child_weight": 10, "subsample": 1.0, "colsample_bytree": 0.90, "reg_lambda": 10, "reg_alpha": 0.5},
    {"max_depth": 6, "learning_rate": 0.03, "min_child_weight": 8, "subsample": 0.90, "colsample_bytree": 0.85, "reg_lambda": 8, "reg_alpha": 0.2},
    {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 10, "subsample": 0.90, "colsample_bytree": 0.90, "reg_lambda": 10, "reg_alpha": 0.5},
    {"max_depth": 6, "learning_rate": 0.08, "min_child_weight": 12, "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 12, "reg_alpha": 1.0},
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    z = np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1 - 1e-7))
    y = np.asarray(y, dtype=float)

    def objective(beta: np.ndarray):
        q = expit(beta[0] + beta[1] * z)
        value = -np.sum(y * np.log(np.clip(q, 1e-15, 1)) + (1 - y) * np.log(np.clip(1 - q, 1e-15, 1)))
        residual = q - y
        return float(value), np.array([residual.sum(), np.dot(residual, z)])

    fit = minimize(objective, np.array([0.0, 1.0]), jac=True, method="BFGS", options={"gtol": 1e-8, "maxiter": 200})
    if not np.all(np.isfinite(fit.x)):
        raise RuntimeError("calibration regression failed")
    return float(fit.x[0]), float(fit.x[1])


def metric_values(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    fpr, tpr, _ = roc_curve(y, p)
    intercept, slope = calibration_intercept_slope(y, p)
    return {"roc_auc": float(roc_auc_score(y, p)), "ks": float(np.max(tpr - fpr)), "pr_auc": float(average_precision_score(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])), "brier": float(brier_score_loss(y, p)),
            "calibration_intercept": intercept, "calibration_slope": slope}


def bootstrap_intervals(y: np.ndarray, predictions: dict[str, np.ndarray]) -> dict:
    rng = np.random.default_rng(SEED + 100)
    samples = {name: {metric: [] for metric in metric_values(y, p)} for name, p in predictions.items()}
    n = len(y)
    for _ in range(BOOTSTRAP_REPLICATES):
        idx = rng.integers(0, n, n)
        for name in predictions:
            for metric, value in metric_values(y[idx], predictions[name][idx]).items():
                samples[name][metric].append(value)
    return {name: {metric: {"lower_95": float(np.percentile(v, 2.5)), "upper_95": float(np.percentile(v, 97.5))}
                   for metric, v in metrics.items()} for name, metrics in samples.items()}


def base_model(params: dict, threads: int, n_estimators: int = 600, early_stopping_rounds: int | None = 40) -> XGBClassifier:
    return XGBClassifier(**params, n_estimators=n_estimators, objective="binary:logistic", eval_metric=["logloss", "auc"],
                         tree_method="hist", device="cpu", n_jobs=threads, random_state=SEED, seed=SEED,
                         early_stopping_rounds=early_stopping_rounds, max_bin=256, verbosity=0)


def fit_calibrators(model: XGBClassifier, X: pd.DataFrame, y: np.ndarray) -> dict:
    margin = model.predict(X, output_margin=True)
    probability = model.predict_proba(X)[:, 1]
    platt = LogisticRegression(C=np.inf, solver="lbfgs", random_state=SEED, max_iter=1000).fit(margin.reshape(-1, 1), y)
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(probability, y)
    return {"platt": platt, "isotonic": isotonic}


def calibrated_predict(model: XGBClassifier, calibrator, method: str, X: pd.DataFrame) -> np.ndarray:
    if method == "platt":
        return calibrator.predict_proba(model.predict(X, output_margin=True).reshape(-1, 1))[:, 1]
    return calibrator.predict(model.predict_proba(X)[:, 1])


def serialize_calibrator(calibrator, method: str) -> dict:
    if method == "platt":
        return {"method": method, "intercept": float(calibrator.intercept_[0]), "coefficient": float(calibrator.coef_[0, 0]), "input": "raw XGBoost margin"}
    return {"method": method, "x_thresholds": calibrator.X_thresholds_.tolist(), "y_thresholds": calibrator.y_thresholds_.tolist(),
            "input": "uncalibrated probability", "out_of_bounds": "clip"}


def train_scorer(data_csv: Path, split_manifest: Path, output: Path, threads: int = 8) -> dict:
    started = time.time()
    output.mkdir(parents=True, exist_ok=True)
    frame = load_table(data_csv)
    manifest = pd.read_csv(split_manifest)
    if len(frame) != len(manifest) or not np.array_equal(frame[TARGET].to_numpy(), manifest["target"].to_numpy()):
        raise RuntimeError("data and split manifest are not aligned")
    features = list(FEATURE_NAMES)
    X = frame[features]
    y = frame[TARGET].to_numpy(dtype=np.int8)
    masks = {name: manifest["split"].eq(name).to_numpy() for name in ("train", "validation", "calibration", "test")}
    X_train, y_train = X.loc[masks["train"]], y[masks["train"]]
    X_val, y_val = X.loc[masks["validation"]], y[masks["validation"]]

    search_rows, fitted = [], []
    for i, params in enumerate(SEARCH):
        model = base_model(params, threads)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        p_val = model.predict_proba(X_val)[:, 1]
        search_rows.append({"candidate": i, **params, "best_iteration": int(model.best_iteration), "trees": int(model.best_iteration + 1),
                            **{f"validation_{k}": v for k, v in metric_values(y_val, p_val).items()}})
        fitted.append(model)
    search = pd.DataFrame(search_rows).sort_values(["validation_roc_auc", "validation_log_loss"], ascending=[False, True])
    search.to_csv(output / "search_results.csv", index=False, lineterminator="\n")
    winner = search.iloc[0]
    winner_id = int(winner["candidate"])

    # Frozen calibration rule: isotonic only if it lowers validation log loss by >= 0.002 with >= 500 calibration defaults.
    initial = fit_calibrators(fitted[winner_id], X.loc[masks["calibration"]], y[masks["calibration"]])
    selection = {method: metric_values(y_val, calibrated_predict(fitted[winner_id], cal, method, X_val)) for method, cal in initial.items()}
    iso_gain = selection["platt"]["log_loss"] - selection["isotonic"]["log_loss"]
    chosen = "isotonic" if iso_gain >= 0.002 and int(y[masks["calibration"]].sum()) >= 500 else "platt"

    development = masks["train"] | masks["validation"]
    final_model = base_model(SEARCH[winner_id], threads, n_estimators=int(winner["trees"]), early_stopping_rounds=None)
    final_model.fit(X.loc[development], y[development], verbose=False)
    calibrator = fit_calibrators(final_model, X.loc[masks["calibration"]], y[masks["calibration"]])[chosen]

    X_test, y_test = X.loc[masks["test"]], y[masks["test"]]
    p_uncal = final_model.predict_proba(X_test)[:, 1]
    p_cal = calibrated_predict(final_model, calibrator, chosen, X_test)
    points = {"uncalibrated": metric_values(y_test, p_uncal), "calibrated": metric_values(y_test, p_cal)}
    intervals = bootstrap_intervals(y_test, {"uncalibrated": p_uncal, "calibrated": p_cal})

    final_model.get_booster().save_model(output / "xgboost_model.json")
    (output / "calibrator.json").write_text(json.dumps(serialize_calibrator(calibrator, chosen), indent=2) + "\n")
    pd.DataFrame({"opaque_row_id": manifest.loc[masks["test"], "opaque_row_id"].to_numpy(), "split": "test", "target": y_test,
                  "probability_uncalibrated": p_uncal, "probability_calibrated": p_cal}).to_csv(
        output / "test_predictions.csv", index=False, float_format="%.12g", lineterminator="\n")
    metrics = {
        "selected_candidate": winner_id, "selected_parameters": SEARCH[winner_id], "selected_trees": int(winner["trees"]),
        "calibration_selection_rule": "isotonic only when its validation log loss is >= 0.002 lower than Platt and calibration has >= 500 defaults; otherwise Platt",
        "calibration_selection_validation_metrics": selection, "calibration_method": chosen,
        "test_class_counts": {"rows": int(masks["test"].sum()), "defaults": int(y_test.sum()), "non_defaults": int(len(y_test) - y_test.sum())},
        "test_metrics": points,
        "bootstrap": {"unit": "client row", "method": "IID percentile bootstrap", "replicates": BOOTSTRAP_REPLICATES, "seed": SEED + 100, "intervals": intervals},
        "test_usage": "test labels loaded once after hyperparameters, tree count and calibration method were frozen; no test tuning",
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    names = ["search_results.csv", "xgboost_model.json", "calibrator.json", "test_predictions.csv", "metrics.json"]
    (output / "run_metadata.json").write_text(json.dumps({
        "seed": SEED, "cpu_threads": threads, "device": "cpu", "search_candidates": len(SEARCH), "elapsed_seconds": time.time() - started,
        "python": sys.version, "platform": platform.platform(),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "scipy": scipy.__version__, "xgboost": xgb.__version__},
        "feature_names": features, "excluded_columns": ["ID", TARGET],
        "input_hashes": {"data_csv": sha256(data_csv), "split_manifest": sha256(split_manifest)},
        "artifact_hashes": {name: sha256(output / name) for name in names},
    }, indent=2) + "\n")
    return metrics
