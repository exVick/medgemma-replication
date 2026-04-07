import json
import os
import time
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_CONDITIONS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema", 
    "Pleural Effusion"
]

DEFAULT_SAMPLE_SIZES = [64, 512, 4096]
DEFAULT_MIN_OVERAGE_RATIO = 1.5
DEFAULT_C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]


def get_memory_usage_mb() -> float:
    """Return memory usage of the current process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def format_labels(df: pd.DataFrame, condition: str) -> np.ndarray:
    """Map CheXpert labels to binary 1/0 (U-Zero approach)."""
    return df[condition].fillna(0.0).replace(-1.0, 0.0).astype(int).values


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC safely; returns NaN when only one class is present."""
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _pick_target_sizes(
    total_available: int,
    sample_sizes: List[int],
    min_overage_ratio: float,
) -> List[int]:
    target_sizes = sorted({int(s) for s in sample_sizes if int(s) > 0 and int(s) <= total_available})

    if target_sizes:
        last_size = target_sizes[-1]
        if total_available >= last_size * min_overage_ratio and total_available not in target_sizes:
            target_sizes.append(total_available)
    else:
        target_sizes = [total_available]

    return target_sizes


def run_cxr_emb_linear_probing_experiment(args) -> List[Dict[str, object]]:
    run_start_time = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading datasets...")
    df_train = pd.read_parquet(args.train_parquet_path)
    df_val = pd.read_parquet(args.val_parquet_path)

    emb_cols = [col for col in df_train.columns if col.startswith("emb_")]
    if not emb_cols:
        raise ValueError("No embedding columns found. Expected columns that start with 'emb_'.")

    missing_emb_in_val = [c for c in emb_cols if c not in df_val.columns]
    if missing_emb_in_val:
        raise ValueError(
            "Validation parquet is missing embedding columns present in train parquet. "
            f"Missing count: {len(missing_emb_in_val)}"
        )

    X_train_full = df_train[emb_cols].values
    X_val_raw = df_val[emb_cols].values

    conditions = args.conditions if args.conditions else DEFAULT_CONDITIONS
    sample_sizes = args.sample_sizes if args.sample_sizes else DEFAULT_SAMPLE_SIZES
    c_values = args.c_values if args.c_values else DEFAULT_C_VALUES

    run_summaries: List[Dict[str, object]] = []

    for condition in conditions:
        if condition not in df_train.columns or condition not in df_val.columns:
            print(f"[SKIP] Missing condition column in train/val parquet: {condition}")
            continue

        print(f"\n{'=' * 40}\nEvaluating Condition: {condition}\n{'=' * 40}")

        y_train_full = format_labels(df_train, condition)
        y_val = format_labels(df_val, condition)

        total_available = len(y_train_full)
        if total_available == 0:
            print(f"[SKIP] No training rows for condition: {condition}")
            continue

        target_sizes = _pick_target_sizes(
            total_available=total_available,
            sample_sizes=sample_sizes,
            min_overage_ratio=args.min_overage_ratio,
        )

        print(f"Planned sample sizes: {target_sizes}")

        for size in target_sizes:
            print(f"\n--- Training {condition} with N={size} ---")
            start_time = time.time()
            start_mem = get_memory_usage_mb()

            if size == total_available:
                X_train_sub, y_train_sub = X_train_full, y_train_full
            else:
                class_counts = np.bincount(y_train_full)
                can_stratify = len(class_counts) > 1 and np.min(class_counts[class_counts > 0]) >= 2
                X_train_sub, _, y_train_sub, _ = train_test_split(
                    X_train_full,
                    y_train_full,
                    train_size=size,
                    stratify=y_train_full if can_stratify else None,
                    random_state=args.random_state,
                )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_sub)
            X_val_scaled = scaler.transform(X_val_raw)

            best_auc = -1.0
            best_c = None
            best_model = None

            for c_val in c_values:
                clf = LogisticRegression(
                    solver="saga",
                    C=float(c_val),
                    class_weight="balanced",
                    max_iter=args.max_iter,
                    random_state=args.random_state,
                )
                clf.fit(X_train_scaled, y_train_sub)

                y_val_pred_proba = clf.predict_proba(X_val_scaled)[:, 1]
                auc = _safe_auc(y_val, y_val_pred_proba)

                score_for_compare = -1.0 if np.isnan(auc) else auc
                if score_for_compare > best_auc:
                    best_auc = score_for_compare
                    best_c = float(c_val)
                    best_model = clf

            if best_model is None:
                print(f"[SKIP] Could not fit a valid model for {condition}, N={size}")
                continue

            best_auc_out = None if best_auc < 0 else round(float(best_auc), 4)
            print(f"Best C selected: {best_c} (Val AUC: {best_auc_out})")

            end_time = time.time()
            end_mem = get_memory_usage_mb()

            model_name = f"{condition}_{size}"
            model_path = os.path.join(args.output_dir, f"{model_name}.joblib")
            json_path = os.path.join(args.output_dir, f"{model_name}.json")

            joblib.dump({"model": best_model, "scaler": scaler}, model_path)

            metadata = {
                "condition": condition,
                "sample_size": int(size),
                "best_hyperparameter_C": best_c,
                "validation_auc": best_auc_out,
                "runtime_seconds": round(end_time - start_time, 2),
                "memory_used_mb": round(end_mem - start_mem, 2),
                "positive_samples_in_train": int(np.sum(y_train_sub == 1)),
                "negative_samples_in_train": int(np.sum(y_train_sub == 0)),
                "train_parquet_path": args.train_parquet_path,
                "val_parquet_path": args.val_parquet_path,
                "num_embedding_dims": len(emb_cols),
            }

            with open(json_path, "w") as f:
                json.dump(metadata, f, indent=4)

            run_summaries.append(metadata)
            print(f"Saved: {model_name}.joblib and .json")

    total_runtime_seconds = round(time.time() - run_start_time, 2)

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "conditions": conditions,
                "sample_sizes": sample_sizes,
                "min_overage_ratio": args.min_overage_ratio,
                "c_values": c_values,
                "total_runtime_seconds": total_runtime_seconds,
                "runs": run_summaries,
            },
            f,
            indent=2,
        )
    print(f"Saved run summary: {summary_path}")

    return run_summaries
