import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _format_labels(series: pd.Series) -> np.ndarray:
    """Map CheXpert labels to binary 1/0 (U-Zero approach)."""
    return series.fillna(0.0).replace(-1.0, 0.0).astype(int).values


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Return AUC or None when undefined (single-class target)."""
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


def _extract_training_size_from_stem(stem: str) -> Optional[int]:
    if "_" not in stem:
        return None
    maybe_size = stem.rsplit("_", 1)[-1]
    try:
        return int(maybe_size)
    except ValueError:
        return None


def _extract_condition_from_stem(stem: str) -> str:
    if "_" not in stem:
        return stem
    return stem.rsplit("_", 1)[0]


def _prepare_labels_df(labels_csv_path: str) -> pd.DataFrame:
    labels_df = pd.read_csv(labels_csv_path)
    if "Path" not in labels_df.columns:
        raise ValueError("Labels CSV must contain a 'Path' column.")

    labels_df = labels_df.copy()
    labels_df["Path_short"] = labels_df["Path"].astype(str).str.replace(
        r"^(?:[^/]+/){2}", "", regex=True
    )
    return labels_df


def _pick_best_join(
    emb_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    join_candidates = [
        ("Path", "Path"),
        ("source_image_path", "Path"),
        ("Path_short", "Path_short"),
    ]

    best = None
    for left_key, right_key in join_candidates:
        if left_key not in emb_df.columns or right_key not in labels_df.columns:
            continue

        merged = emb_df.merge(
            labels_df,
            left_on=left_key,
            right_on=right_key,
            how="inner",
            suffixes=("", "_label"),
        )

        matched = len(merged)
        if best is None or matched > best[0]:
            best = (matched, left_key, right_key, merged)

    if best is None or best[0] == 0:
        raise ValueError(
            "Could not match test embeddings to labels. "
            "Expected one of these key mappings: "
            "Path->Path, source_image_path->Path, Path_short->Path_short."
        )

    matched, left_key, right_key, merged = best
    join_info = {
        "matched_rows": int(matched),
        "embedding_join_key": left_key,
        "label_join_key": right_key,
        "embedding_rows_total": int(len(emb_df)),
        "label_rows_total": int(len(labels_df)),
    }
    return merged, join_info


def _load_probe_meta(meta_json_path: Path) -> Dict[str, object]:
    if not meta_json_path.exists():
        return {}
    with open(meta_json_path, "r") as f:
        return json.load(f)


def run_cxr_test_linear_probing_experiment(args) -> List[Dict[str, object]]:
    os.makedirs(args.output_dir, exist_ok=True)

    run_start = time.time()

    print("Loading test embeddings and labels...")
    emb_df = pd.read_parquet(args.test_parquet_path)
    labels_df = _prepare_labels_df(args.csv_file)

    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError("Test parquet has no embedding columns (expected 'emb_*').")

    eval_df, join_info = _pick_best_join(emb_df, labels_df)

    X_test = eval_df[emb_cols].values

    model_dir = Path(args.linear_probes_path)
    if not model_dir.exists() or not model_dir.is_dir():
        raise ValueError(f"Invalid linear probes directory: {model_dir}")

    model_paths = sorted(model_dir.glob("*.joblib"))
    if not model_paths:
        raise ValueError(f"No .joblib models found in {model_dir}")

    results: List[Dict[str, object]] = []

    for model_path in model_paths:
        stem = model_path.stem
        per_model_start = time.time()

        probe_meta_path = model_path.with_suffix(".json")
        probe_meta = _load_probe_meta(probe_meta_path)

        condition = str(probe_meta.get("condition") or _extract_condition_from_stem(stem))
        train_size = probe_meta.get("sample_size")
        if train_size is None:
            train_size = _extract_training_size_from_stem(stem)

        best_c = probe_meta.get("best_hyperparameter_C")

        if condition not in eval_df.columns:
            result = {
                "model_name": stem,
                "condition": condition,
                "training_set_size": train_size,
                "best_hyperparameter_C": best_c,
                "auc": None,
                "runtime_seconds": round(time.time() - per_model_start, 4),
                "num_test_samples": 0,
                "status": "skipped_missing_condition_column",
            }
            results.append(result)
            continue

        y_test = _format_labels(eval_df[condition])

        bundle = joblib.load(model_path)
        if "model" not in bundle or "scaler" not in bundle:
            raise ValueError(
                f"Model artifact {model_path.name} does not contain required keys: 'model', 'scaler'."
            )

        model = bundle["model"]
        scaler = bundle["scaler"]

        X_test_scaled = scaler.transform(X_test)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        auc = _safe_auc(y_test, y_pred_proba)

        result = {
            "model_name": stem,
            "condition": condition,
            "training_set_size": int(train_size) if train_size is not None else None,
            "best_hyperparameter_C": best_c,
            "auc": None if auc is None else round(auc, 4),
            "runtime_seconds": round(time.time() - per_model_start, 4),
            "num_test_samples": int(len(y_test)),
            "num_positive_test_samples": int(np.sum(y_test == 1)),
            "num_negative_test_samples": int(np.sum(y_test == 0)),
            "status": "ok" if auc is not None else "auc_undefined_single_class",
        }
        results.append(result)

        print(
            f"[{stem}] condition={condition} train_size={train_size} "
            f"best_C={best_c} auc={result['auc']}"
        )

    total_runtime = round(time.time() - run_start, 2)

    payload = {
        "evaluation_meta": {
            "run_date": datetime.now().isoformat(),
            "test_parquet_path": args.test_parquet_path,
            "csv_file": args.csv_file,
            "linear_probes_path": args.linear_probes_path,
            "output_dir": args.output_dir,
            "num_models": len(model_paths),
            "num_embedding_dims": len(emb_cols),
            "join_info": join_info,
            "total_runtime_seconds": total_runtime,
        },
        "results": results,
    }

    output_name = args.output_file if args.output_file else "cxr_test_linear_probing_results.json"
    output_path = Path(args.output_dir) / output_name
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved evaluation JSON: {output_path}")

    return results
