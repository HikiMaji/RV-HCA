#!/usr/bin/env python
"""Test whether causal historical MTR errors predict later MTR errors.

This is a label-free-at-inference feasibility audit.  ``peer_realized_error``
and ``ego_only_error`` are receiver-observation errors already written to the
ledger; GT is not read.  The script refuses constant-velocity probe rows and
does not fit an attention/MLP/controller.  It reports simple rank correlation
and harm AUC by held-out scene, source, receiver target, and horizon.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

try:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - the CPU audit can still run descriptively
    SimpleImputer = None
    LogisticRegression = None
    make_pipeline = None
    StandardScaler = None


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Stream only ledger fields needed for the causal audit.

    Real MTR forecasts are large multimodal arrays.  Keeping those arrays in
    memory while analysing a ledger can exceed the container limit, although
    none of the predictiveness statistics uses the forecast values after
    ingestion.  Each JSON line is therefore reduced immediately to a small
    metadata/error record; the full forecast remains intact on disk.
    """
    keep = {
        "sequence_id",
        "target_scope",
        "valid_mask",
        "peer_realized_error",
        "ego_only_error",
        "observation_time",
        "send_time",
        "receiver",
        "source",
        "receiver_target_id",
        "horizon",
        # These are receiver-visible metadata fields.  They are deliberately
        # kept as scalars; no GT-derived ID or target label enters a feature.
        "association_confidence",
        "track_age",
        "miss_count",
        "arrival_time",
    }
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            row = {key: raw.get(key) for key in keep if key in raw}
            # Only the model tag is needed to distinguish official MTR rows
            # from the constant-velocity probe; drop the large arrays.
            for field in ("forecast", "ego_forecast"):
                value = raw.get(field)
                if isinstance(value, Mapping):
                    row[field] = {"model": value.get("model", "")}
                else:
                    row[field] = value
            yield row


def _model(row: Mapping[str, Any]) -> str:
    forecast = row.get("forecast")
    return str(forecast.get("model", "")) if isinstance(forecast, Mapping) else ""


def _is_real_mtr(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    model = str(value.get("model", "")).lower()
    return model not in {"", "constant_velocity_probe", "mean_probe_not_cmp"} and (
        "mtr" in model or "cmp" in model
    )


def _auc(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    positives = [float(score) for score, label in zip(scores, labels) if label]
    negatives = [float(score) for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return None
    # Mann-Whitney formulation, with average credit for ties.
    wins = 0.0
    for positive in positives:
        wins += sum(1.0 if positive > negative else 0.5 if positive == negative else 0.0 for negative in negatives)
    return wins / (len(positives) * len(negatives))


def _finite_float(value: Any, default: float = float("nan")) -> float:
    """Convert an optional JSON scalar to a finite float or NaN."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _history_features(entries: Sequence[Tuple[float, float]]) -> Dict[str, float]:
    """Summarize a strict-causal history without looking at the current row."""
    ordered = sorted(entries, key=lambda item: item[0])
    if not ordered:
        return {
            "latest": float("nan"),
            "ewma": float("nan"),
            "count": 0.0,
            "available": 0.0,
        }
    values = [float(item[1]) for item in ordered]
    tail = values[-5:]
    return {
        "latest": values[-1],
        "ewma": float(np.mean(tail)),
        "count": float(len(values)),
        "available": 1.0,
    }


def _diagnostic_auc(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    """Fit a tiny held-out logistic probe and return only its test AUC.

    This is an offline information test, not a reliability controller: it
    emits no gate, action, or checkpoint.  The probe is intentionally linear
    so that an A/B/C AUC change can be attributed to adding feature groups.
    """
    result: Dict[str, Any] = {
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "features": list(feature_names),
    }
    if SimpleImputer is None or LogisticRegression is None:
        result["status"] = "SKLEARN_UNAVAILABLE"
        result["auc"] = None
        return result
    if len(train_rows) < 20 or len(test_rows) < 20:
        result["status"] = "INSUFFICIENT_ROWS"
        result["auc"] = None
        return result

    def matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return np.asarray(
            [[_finite_float(row.get(name)) for name in feature_names] for row in rows],
            dtype=np.float64,
        )

    y_train = np.asarray([bool(row["harm"]) for row in train_rows], dtype=np.int64)
    y_test = [bool(row["harm"]) for row in test_rows]
    result["train_harm_rate"] = float(np.mean(y_train)) if len(y_train) else None
    result["test_harm_rate"] = float(np.mean(y_test)) if y_test else None
    if len(np.unique(y_train)) < 2 or len(set(y_test)) < 2:
        result["status"] = "ONE_CLASS_TRAIN_OR_TEST"
        result["auc"] = None
        return result

    # Median imputation is estimated on the training scenes only.  Missing
    # history is also represented by explicit *_available features.
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            C=1.0,
            max_iter=300,
            random_state=0,
        ),
    )
    try:
        with warnings.catch_warnings():
            # At long horizons a held-out training split can have no matured
            # history at all.  The explicit *_available/count features retain
            # that fact; suppress only sklearn's redundant all-NaN warning.
            warnings.filterwarnings(
                "ignore",
                message="Skipping features without any observed values.*",
                category=UserWarning,
                module="sklearn.impute",
            )
            model.fit(matrix(train_rows), y_train)
            score = model.predict_proba(matrix(test_rows))[:, 1]
    except Exception as exc:  # keep per-cell status explicit, never abort all scenes
        result["status"] = "FIT_ERROR:%s" % type(exc).__name__
        result["auc"] = None
        return result
    result["auc"] = _auc(score.tolist(), y_test)
    result["status"] = "EVALUATED" if result["auc"] is not None else "ONE_CLASS_TEST"
    return result


def _feature_group_names(scope: str, variant: str) -> Dict[str, List[str]]:
    """Feature names for metadata (A), ego history (B), and peer history (C)."""
    prefix = "%s_" % scope
    metadata = ["meta_assoc_confidence", "meta_track_age", "meta_miss_count", "meta_delay_s"]
    ego = [
        prefix + "ego_latest",
        prefix + "ego_ewma",
        prefix + "ego_count",
        prefix + "ego_available",
    ]
    peer_field = prefix + ("peer_latest" if variant == "latest" else "peer_ewma")
    peer = [peer_field, prefix + "peer_count", prefix + "peer_available"]
    return {"A": metadata, "B": metadata + ego, "C": metadata + ego + peer}


def _harm_loso(
    causal_rows: Sequence[Mapping[str, Any]],
    scenes: Sequence[str],
    horizons: Sequence[float],
) -> Dict[str, Any]:
    """LOSO A/B/C relative-harm probe for target- and source-level histories."""
    output: Dict[str, Any] = {
        "description": (
            "Offline linear diagnostic only. A=receiver metadata; B=A plus ego "
            "realized-error history; C=B plus peer realized-error history. "
            "All history features are strict causal prefixes."
        ),
        "label": "peer_realized_error > ego_only_error + 0.1m",
        "scenes": list(scenes),
        "by_scope": {},
    }
    for scope in ("target", "source"):
        scope_output: Dict[str, Any] = {"by_heldout_scene": {}}
        for heldout in scenes:
            scene_output: Dict[str, Any] = {"by_horizon": {}}
            for horizon in horizons:
                h = round(float(horizon), 3)
                train = [
                    row
                    for row in causal_rows
                    if str(row.get("sequence_id")) != heldout
                    and round(float(row.get("horizon", 0.0)), 3) == h
                ]
                test = [
                    row
                    for row in causal_rows
                    if str(row.get("sequence_id")) == heldout
                    and round(float(row.get("horizon", 0.0)), 3) == h
                ]
                cells: Dict[str, Any] = {
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "test_harm_rate": float(np.mean([bool(r["harm"]) for r in test])) if test else None,
                }
                variants: Dict[str, Any] = {}
                for variant in ("latest", "ewma"):
                    names = _feature_group_names(scope, variant)
                    a = _diagnostic_auc(train, test, names["A"])
                    b = _diagnostic_auc(train, test, names["B"])
                    c = _diagnostic_auc(train, test, names["C"])
                    delta = None
                    if c.get("auc") is not None and b.get("auc") is not None:
                        delta = float(c["auc"] - b["auc"])
                    variants[variant] = {
                        "A": a,
                        "B": b,
                        "C": c,
                        "delta_auc_C_minus_B": delta,
                    }
                cells["latest"] = variants["latest"]
                cells["ewma"] = variants["ewma"]
                scene_output["by_horizon"][str(h)] = cells
            scope_output["by_heldout_scene"][heldout] = scene_output
        output["by_scope"][scope] = scope_output
    return output


def _short_to_long_harm(
    rows: Sequence[Mapping[str, Any]],
    short_horizons: Sequence[float] = (0.3, 0.5, 1.0),
    long_horizons: Sequence[float] = (2.0, 3.0, 5.0),
) -> Dict[str, Any]:
    """Evaluate short-horizon peer error as a causal predictor of long harm."""
    result: Dict[str, Any] = {
        "description": (
            "For the same scene/receiver/source/target, the latest or EWMA "
            "peer error already matured at send time predicts current long-horizon relative harm."
        ),
        "by_scene": {},
        "pooled": {},
    }
    eligible = [
        row for row in rows
        if row.get("target_scope") == "common"
        and row.get("valid_mask")
        and row.get("peer_realized_error") is not None
        and row.get("ego_only_error") is not None
        and _is_real_mtr(row.get("forecast"))
        and _is_real_mtr(row.get("ego_forecast"))
    ]
    by_scene: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_scene[str(row.get("sequence_id"))].append(row)

    # Replay independently per scene, retaining only matured peer errors by
    # target and short horizon.  No GT is read here.
    all_pairs: MutableMapping[Tuple[float, float], List[Dict[str, Any]]] = defaultdict(list)
    for scene, scene_rows in sorted(by_scene.items()):
        scene_rows = sorted(scene_rows, key=lambda row: float(row.get("send_time", 0.0)))
        send_times = sorted({float(row.get("send_time", 0.0)) for row in scene_rows})
        matured_candidates = sorted(
            [row for row in scene_rows if row.get("observation_time") is not None],
            key=lambda row: float(row["observation_time"]),
        )
        pending: List[Mapping[str, Any]] = []
        pointer = 0
        history: MutableMapping[Tuple[str, str, str, str, float], List[Tuple[float, float]]] = defaultdict(list)
        for send_time in send_times:
            while pointer < len(matured_candidates) and float(matured_candidates[pointer]["observation_time"]) <= send_time + 1e-9:
                pending.append(matured_candidates[pointer])
                pointer += 1
            still_pending: List[Mapping[str, Any]] = []
            for old in pending:
                old_send = float(old.get("send_time", 0.0))
                if old_send >= send_time - 1e-9:
                    still_pending.append(old)
                    continue
                key = (
                    scene,
                    str(old.get("receiver")),
                    str(old.get("source")),
                    str(old.get("receiver_target_id")),
                    round(float(old.get("horizon", 0.0)), 3),
                )
                history[key].append((old_send, float(old["peer_realized_error"])))
            pending = still_pending
            current_rows = [item for item in scene_rows if abs(float(item.get("send_time", 0.0)) - send_time) <= 1e-9]
            for row in current_rows:
                long_h = round(float(row.get("horizon", 0.0)), 3)
                if long_h not in {round(float(v), 3) for v in long_horizons}:
                    continue
                base = (
                    scene,
                    str(row.get("receiver")),
                    str(row.get("source")),
                    str(row.get("receiver_target_id")),
                )
                label = float(row["peer_realized_error"]) > float(row["ego_only_error"]) + 0.10
                for short_horizon in short_horizons:
                    short_h = round(float(short_horizon), 3)
                    entries = sorted(history.get(base + (short_h,), []), key=lambda item: item[0])
                    if not entries:
                        continue
                    values = [value for _, value in entries]
                    pair = {
                        "sequence_id": scene,
                        "long_horizon": long_h,
                        "short_horizon": short_h,
                        "latest": values[-1],
                        "ewma": float(np.mean(values[-5:])),
                        "harm": label,
                    }
                    all_pairs[(short_h, long_h)].append(pair)
        # (The scene field is retained in all_pairs, so no second replay is needed.)

    for (short_h, long_h), pairs in sorted(all_pairs.items()):
        key = "%s_to_%s" % (short_h, long_h)
        result["pooled"][key] = {
            "short_horizon": short_h,
            "long_horizon": long_h,
            "rows": len(pairs),
            "harm_rate": float(np.mean([bool(p["harm"]) for p in pairs])) if pairs else None,
            "auc_latest": _auc([p["latest"] for p in pairs], [bool(p["harm"]) for p in pairs]),
            "auc_ewma": _auc([p["ewma"] for p in pairs], [bool(p["harm"]) for p in pairs]),
            "spearman_latest": _corr([p["latest"] for p in pairs], [float(p["harm"]) for p in pairs]),
            "spearman_ewma": _corr([p["ewma"] for p in pairs], [float(p["harm"]) for p in pairs]),
        }
        by_scene_pairs: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
        for pair in pairs:
            by_scene_pairs[str(pair["sequence_id"])].append(pair)
        result["by_scene"][key] = {
            scene: {
                "short_horizon": short_h,
                "long_horizon": long_h,
                "rows": len(scene_pairs),
                "harm_rate": float(np.mean([bool(p["harm"]) for p in scene_pairs])) if scene_pairs else None,
                "auc_latest": _auc([p["latest"] for p in scene_pairs], [bool(p["harm"]) for p in scene_pairs]),
                "auc_ewma": _auc([p["ewma"] for p in scene_pairs], [bool(p["harm"]) for p in scene_pairs]),
            }
            for scene, scene_pairs in sorted(by_scene_pairs.items())
        }
    return result


def _corr(values_x: Sequence[float], values_y: Sequence[float]) -> Optional[float]:
    if len(values_x) < 3 or len(set(values_x)) < 2 or len(set(values_y)) < 2:
        return None
    result = spearmanr(values_x, values_y)
    value = float(result.statistic if hasattr(result, "statistic") else result[0])
    return value if math.isfinite(value) else None


def _group_summary(rows: Sequence[Mapping[str, Any]], group_name: str, group_value: Any, eps: float) -> Dict[str, Any]:
    if not rows:
        return {"group": group_name, "value": group_value, "rows": 0}
    latest = [float(row["history_latest_diff"]) for row in rows]
    ewma = [float(row["history_ewma_diff"]) for row in rows]
    current_error = [float(row["current_peer_error"]) for row in rows]
    current_diff = [float(row["current_diff"]) for row in rows]
    labels = [value > eps for value in current_diff]
    return {
        "group": group_name,
        "value": group_value,
        "rows": len(rows),
        "spearman_latest_vs_current_peer_error": _corr(latest, current_error),
        "spearman_ewma_vs_current_peer_error": _corr(ewma, current_error),
        "spearman_latest_vs_current_regret": _corr(latest, current_diff),
        "spearman_ewma_vs_current_regret": _corr(ewma, current_diff),
        "auc_latest_for_harm": _auc(latest, labels),
        "auc_ewma_for_harm": _auc(ewma, labels),
        "harm_rate_eps": float(np.mean(labels)) if labels else None,
    }


def _stable(summary: Mapping[str, Any], min_auc: float, min_abs_rho: float, min_rows: int) -> bool:
    if int(summary.get("rows", 0)) < min_rows:
        return False
    auc = summary.get("auc_latest_for_harm")
    rho = summary.get("spearman_latest_vs_current_regret")
    if auc is None or rho is None:
        return False
    return float(auc) >= min_auc and abs(float(rho)) >= min_abs_rho


def _threshold_stats(scores: Sequence[float], labels: Sequence[bool], threshold: float) -> Dict[str, Any]:
    """Evaluate a fixed threshold calibrated on non-held-out scenes."""
    predicted = [float(score) > threshold for score in scores]
    tp = sum(pred and label for pred, label in zip(predicted, labels))
    fp = sum(pred and not label for pred, label in zip(predicted, labels))
    fn = sum((not pred) and label for pred, label in zip(predicted, labels))
    tn = sum((not pred) and (not label) for pred, label in zip(predicted, labels))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2.0 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else None
    accuracy = (tp + tn) / len(labels) if labels else None
    return {
        "threshold": float(threshold),
        "predicted_harm_rate": float(np.mean(predicted)) if predicted else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def _heldout_feature_eval(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    feature: str,
    eps: float,
) -> Dict[str, Any]:
    """Score one causal feature on a scene excluded from calibration."""
    if not train_rows or not test_rows:
        return {
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "status": "INSUFFICIENT_TRAIN_OR_TEST_ROWS",
        }
    train_scores = [float(row[feature]) for row in train_rows]
    test_scores = [float(row[feature]) for row in test_rows]
    test_regret = [float(row["current_diff"]) for row in test_rows]
    labels = [value > eps for value in test_regret]
    threshold = float(np.median(train_scores))
    result = {
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "spearman_vs_current_regret": _corr(test_scores, test_regret),
        "spearman_vs_current_peer_error": _corr(
            test_scores, [float(row["current_peer_error"]) for row in test_rows]
        ),
        "auc_for_harm": _auc(test_scores, labels),
        "test_harm_rate_eps": float(np.mean(labels)) if labels else None,
    }
    result.update(_threshold_stats(test_scores, labels, threshold))
    result["status"] = "EVALUATED"
    return result


def _heldout_scope_eval(
    all_rows: Sequence[Mapping[str, Any]],
    heldout_scene: str,
    eps: float,
) -> Dict[str, Any]:
    """Leave-one-scene-out evaluation for a source or target feature stream.

    The only calibrated quantity is a median threshold from the other scenes;
    no controller or learned model is fitted.  All test features are causal
    receiver-observation histories produced before the current forecast.
    """
    train_rows = [row for row in all_rows if str(row.get("sequence_id")) != heldout_scene]
    test_rows = [row for row in all_rows if str(row.get("sequence_id")) == heldout_scene]
    train_scenes = sorted({str(row.get("sequence_id")) for row in train_rows})
    horizons = sorted({round(float(row.get("horizon", 0.0)), 3) for row in test_rows})
    result: Dict[str, Any] = {
        "heldout_scene": heldout_scene,
        "train_scenes": train_scenes,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "status": "EVALUATED" if train_rows and test_rows else "INSUFFICIENT_TRAIN_OR_TEST_ROWS",
        "overall": {
            "latest": _heldout_feature_eval(train_rows, test_rows, "history_latest_diff", eps),
            "ewma": _heldout_feature_eval(train_rows, test_rows, "history_ewma_diff", eps),
        },
        "by_horizon": {},
    }
    for horizon in horizons:
        train_h = [row for row in train_rows if round(float(row.get("horizon", 0.0)), 3) == horizon]
        test_h = [row for row in test_rows if round(float(row.get("horizon", 0.0)), 3) == horizon]
        result["by_horizon"][str(horizon)] = {
            "latest": _heldout_feature_eval(train_h, test_h, "history_latest_diff", eps),
            "ewma": _heldout_feature_eval(train_h, test_h, "history_ewma_diff", eps),
        }
    return result


def analyze(
    rows: Sequence[Mapping[str, Any]],
    min_rows: int = 20,
    min_auc: float = 0.55,
    min_abs_rho: float = 0.10,
    eps: float = 0.10,
) -> Dict[str, Any]:
    # ``read_jsonl`` is a streaming generator; materialize only the slim
    # metadata records, never the multimodal forecast arrays.
    if not isinstance(rows, Sequence):
        rows = list(rows)
    mtr_rows = [
        row
        for row in rows
        if row.get("target_scope") == "common"
        and row.get("valid_mask")
        and row.get("peer_realized_error") is not None
        and row.get("ego_only_error") is not None
        and _is_real_mtr(row.get("forecast"))
        and _is_real_mtr(row.get("ego_forecast"))
    ]
    if not mtr_rows:
        return {
            "status": "BLOCKED_NO_REAL_MTR",
            "reason": "ledger contains no valid common rows whose forecast.model identifies a real MTR/CMP prediction",
            "input_rows": len(rows),
            "mtr_rows": 0,
        }

    # Replay one sequence at a time.  A matured row is first placed in a
    # pending queue ordered by receiver-observation time.  Rows issued at the
    # current send time stay pending until the *next* send time, enforcing the
    # strict causal condition send_time(history) < send_time(current).
    by_scene: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in mtr_rows:
        by_scene[str(row.get("sequence_id"))].append(row)
    histories_target: MutableMapping[Tuple[str, str, str, str, float], List[Tuple[float, float]]] = defaultdict(list)
    histories_source: MutableMapping[Tuple[str, str, str, float], List[Tuple[float, float]]] = defaultdict(list)
    # Separate ego/peer streams are used by the requested A/B/C probe.  The
    # legacy ``histories_*`` maps above retain the peer-minus-ego descriptive
    # diagnostic for backwards-compatible output.
    error_histories_target: MutableMapping[Tuple[str, str, str, str, float], Dict[str, List[Tuple[float, float]]]] = defaultdict(lambda: {"ego": [], "peer": []})
    error_histories_source: MutableMapping[Tuple[str, str, str, float], Dict[str, List[Tuple[float, float]]]] = defaultdict(lambda: {"ego": [], "peer": []})
    paired_target: MutableMapping[Tuple[str, str, str, str, float], List[Dict[str, Any]]] = defaultdict(list)
    paired_source: MutableMapping[Tuple[str, str, str, float], List[Dict[str, Any]]] = defaultdict(list)
    causal_rows: List[Dict[str, Any]] = []

    for scene, scene_rows in sorted(by_scene.items()):
        scene_rows = sorted(scene_rows, key=lambda row: float(row.get("send_time", 0.0)))
        send_times = sorted({float(row.get("send_time", 0.0)) for row in scene_rows})
        matured_candidates = sorted(
            [row for row in scene_rows if row.get("observation_time") is not None],
            key=lambda row: float(row["observation_time"]),
        )
        pending: List[Mapping[str, Any]] = []
        pointer = 0
        for send_time in send_times:
            while pointer < len(matured_candidates) and float(matured_candidates[pointer]["observation_time"]) <= send_time + 1e-9:
                pending.append(matured_candidates[pointer])
                pointer += 1
            still_pending: List[Mapping[str, Any]] = []
            for history_row in pending:
                history_send = float(history_row.get("send_time", 0.0))
                if history_send >= send_time - 1e-9:
                    still_pending.append(history_row)
                    continue
                h_receiver = str(history_row.get("receiver"))
                h_source = str(history_row.get("source"))
                h_target = str(history_row.get("receiver_target_id"))
                h_horizon = round(float(history_row.get("horizon", 0.0)), 3)
                h_peer = float(history_row["peer_realized_error"])
                h_ego = float(history_row["ego_only_error"])
                h_diff = h_peer - h_ego
                histories_target[(scene, h_receiver, h_source, h_target, h_horizon)].append((history_send, h_diff))
                histories_source[(scene, h_receiver, h_source, h_horizon)].append((history_send, h_diff))
                target_error_key = (scene, h_receiver, h_source, h_target, h_horizon)
                source_error_key = (scene, h_receiver, h_source, h_horizon)
                error_histories_target[target_error_key]["ego"].append((history_send, h_ego))
                error_histories_target[target_error_key]["peer"].append((history_send, h_peer))
                error_histories_source[source_error_key]["ego"].append((history_send, h_ego))
                error_histories_source[source_error_key]["peer"].append((history_send, h_peer))
            pending = still_pending

            for row in [item for item in scene_rows if abs(float(item.get("send_time", 0.0)) - send_time) <= 1e-9]:
                receiver = str(row.get("receiver"))
                source = str(row.get("source"))
                target = str(row.get("receiver_target_id"))
                horizon = round(float(row.get("horizon", 0.0)), 3)
                target_key = (scene, receiver, source, target, horizon)
                source_key = (scene, receiver, source, horizon)
                target_history = sorted(histories_target[target_key], key=lambda item: item[0])
                source_history = sorted(histories_source[source_key], key=lambda item: item[0])
                current_error = float(row["peer_realized_error"])
                current_ego_error = float(row["ego_only_error"])
                current_diff = current_error - current_ego_error
                if target_history:
                    latest = target_history[-1][1]
                    target_ewma = sum(value for _, value in target_history[-5:]) / min(5, len(target_history))
                    paired_target[target_key].append({
                        "sequence_id": scene, "receiver": receiver, "source": source,
                        "receiver_target_id": target, "horizon": horizon,
                        "history_latest_diff": latest, "history_ewma_diff": target_ewma,
                        "current_peer_error": current_error, "current_diff": current_diff,
                    })
                if source_history:
                    latest = source_history[-1][1]
                    source_ewma = sum(value for _, value in source_history[-5:]) / min(5, len(source_history))
                    paired_source[source_key].append({
                        "sequence_id": scene, "receiver": receiver, "source": source,
                        "receiver_target_id": target, "horizon": horizon,
                        "history_latest_diff": latest, "history_ewma_diff": source_ewma,
                        "current_peer_error": current_error, "current_diff": current_diff,
                    })

                # Causal A/B/C feature record.  No GT field is read here: the
                # two realized-error values were obtained from the receiver's
                # ego-only observation path and are available only for past,
                # matured rows.  The current row contributes only its harm
                # label for this offline diagnostic.
                target_errors = error_histories_target[target_key]
                source_errors = error_histories_source[source_key]
                def scope_record(scope_name: str, histories: Mapping[str, Sequence[Tuple[float, float]]]) -> Dict[str, float]:
                    ego_features = _history_features(histories.get("ego", []))
                    peer_features = _history_features(histories.get("peer", []))
                    return {
                        scope_name + "_ego_latest": ego_features["latest"],
                        scope_name + "_ego_ewma": ego_features["ewma"],
                        scope_name + "_ego_count": ego_features["count"],
                        scope_name + "_ego_available": ego_features["available"],
                        scope_name + "_peer_latest": peer_features["latest"],
                        scope_name + "_peer_ewma": peer_features["ewma"],
                        scope_name + "_peer_count": peer_features["count"],
                        scope_name + "_peer_available": peer_features["available"],
                    }
                causal = {
                    "sequence_id": scene,
                    "receiver": receiver,
                    "source": source,
                    "receiver_target_id": target,
                    "horizon": horizon,
                    "harm": bool(current_error > current_ego_error + eps),
                    "peer_error": current_error,
                    "ego_error": current_ego_error,
                    "meta_assoc_confidence": _finite_float(row.get("association_confidence")),
                    "meta_track_age": _finite_float(row.get("track_age")),
                    "meta_miss_count": _finite_float(row.get("miss_count")),
                    "meta_delay_s": _finite_float(row.get("arrival_time")) - _finite_float(row.get("send_time")),
                }
                causal.update(scope_record("target", target_errors))
                causal.update(scope_record("source", source_errors))
                causal_rows.append(causal)

    target_summaries = []
    for key, group_rows in sorted(paired_target.items()):
        scene, receiver, source, target, horizon = key
        item = _group_summary(group_rows, "target", {
            "sequence_id": scene,
            "receiver": receiver,
            "source": source,
            "receiver_target_id": target,
            "horizon": horizon,
        }, eps)
        target_summaries.append(item)
    source_summaries = []
    for key, group_rows in sorted(paired_source.items()):
        scene, receiver, source, horizon = key
        item = _group_summary(group_rows, "source", {
            "sequence_id": scene,
            "receiver": receiver,
            "source": source,
            "horizon": horizon,
        }, eps)
        source_summaries.append(item)

    heldout: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for group in target_summaries + source_summaries:
        value = group.get("value", {})
        scene = str(value.get("sequence_id")) if isinstance(value, Mapping) else ""
        heldout[scene].append(group)
    scene_summary = {
        scene: {
            "groups": len(groups),
            "stable_groups": sum(_stable(group, min_auc, min_abs_rho, min_rows) for group in groups),
            "groups_with_min_rows": sum(int(group.get("rows", 0)) >= min_rows for group in groups),
        }
        for scene, groups in sorted(heldout.items())
    }
    # Explicit leave-one-scene-out diagnostic.  This is separate from the
    # descriptive per-scene group table above: the threshold used for a test
    # scene is calibrated only from the other scenes, while rank metrics are
    # computed on the held-out scene itself.
    paired_target_all = [item for group_rows in paired_target.values() for item in group_rows]
    paired_source_all = [item for group_rows in paired_source.values() for item in group_rows]
    heldout_scene_evaluation = {
        scene: {
            "source": _heldout_scope_eval(paired_source_all, scene, eps),
            "target": _heldout_scope_eval(paired_target_all, scene, eps),
        }
        for scene in sorted(by_scene)
    }
    horizons = sorted({round(float(row.get("horizon", 0.0)), 3) for row in causal_rows})
    harm_loso = _harm_loso(causal_rows, sorted(by_scene), horizons)
    short_to_long_harm = _short_to_long_harm(mtr_rows)
    stable_groups = [
        group
        for group in target_summaries + source_summaries
        if _stable(group, min_auc, min_abs_rho, min_rows)
    ]
    if not stable_groups:
        status = "NO_GO_HISTORICAL_ERROR_SIGNAL" if target_summaries or source_summaries else "INSUFFICIENT_HISTORY"
    else:
        status = "SIGNAL_EXISTS_NO_CONTROLLER_TRAINED"
    return {
        "status": status,
        "input_rows": len(rows),
        "mtr_rows": len(mtr_rows),
        "paired_target_rows": sum(len(value) for value in paired_target.values()),
        "paired_source_rows": sum(len(value) for value in paired_source.values()),
        "thresholds": {
            "min_rows_per_group": min_rows,
            "min_auc_for_harm": min_auc,
            "min_abs_spearman_regret": min_abs_rho,
            "harm_eps_m": eps,
        },
        "heldout_scene": scene_summary,
        "heldout_scene_evaluation": heldout_scene_evaluation,
        "harm_loso": harm_loso,
        "short_to_long_harm": short_to_long_harm,
        "target_groups": target_summaries,
        "source_groups": source_summaries,
        "interpretation": "all histories are strict causal receiver-observation prefixes; no GT or learned controller is used",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path, help="prediction_ledger.jsonl")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-rows", type=int, default=20)
    parser.add_argument("--min-auc", type=float, default=0.55)
    parser.add_argument("--min-abs-rho", type=float, default=0.10)
    parser.add_argument("--eps", type=float, default=0.10)
    args = parser.parse_args()
    result = analyze(read_jsonl(args.ledger), args.min_rows, args.min_auc, args.min_abs_rho, args.eps)
    output = args.output or args.ledger.with_name("mtr_predictiveness.json")
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print("output=%s" % output)
    for key, value in result.items():
        if key not in {"target_groups", "source_groups"}:
            print("%s=%s" % (key, value))


if __name__ == "__main__":
    main()
