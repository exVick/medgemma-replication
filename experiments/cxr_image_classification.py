"""
CXR image classification (CheXpert-style) experiment.

Keeps:
- condition-wise yes/no prompting
- rule-based parsing
- per-condition F1 + macro-F1 evaluation (in this same file)

Future extensions can be added here:
- uncertainty calibration
- threshold tuning
- PR/AUC metrics
- subgroup analysis
"""

import os
import time
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score

from core.utils import init_experiment_meta, save_results_with_meta
from config.prompts import PROMPTS


GENERATION_CONFIG = dict(
    max_new_tokens=64,
    do_sample=False,  # greedy deterministic decoding
)

# Default 5-condition setup (can be overridden via --conditions)
DEFAULT_CONDITIONS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]


def add_cxr_classification_args(parser):
    parser.add_argument("--csv_file", type=str, required=True, help="CheXpert-style CSV")
    parser.add_argument("--image_dir", type=str, required=True, help="Root image directory")
    parser.add_argument("--output_file", type=str, default="results_cxr_classification.csv")
    parser.add_argument("--model_id", type=str, default="google/medgemma-4b-it")
    parser.add_argument(
        "--float_type",
        type=str,
        default="float32",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--max_samples", type=int, default=-1, help="-1 = all")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=DEFAULT_CONDITIONS,
        help="Condition columns to evaluate",
    )
    parser.add_argument(
        "--path_col",
        type=str,
        default="Path",
        help="CSV column containing image relative path",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=50,
        help="Checkpoint save frequency",
    )


def build_messages(condition: str):
    prompt = PROMPTS["classify_condition"].format(condition=condition.lower())
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def run_inference_single(model, processor, image: Image.Image, condition: str) -> str:
    """Run inference for one image and one condition. Returns raw generated text."""
    messages = build_messages(condition)

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    ).to(model.device)

    # Keep your explicit float32 cast for stability in your setup
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            **GENERATION_CONFIG,
        )

    generated_ids = output_ids[0][input_len:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


def parse_yes_no(text: str) -> int:
    """
    Parse model output into CheXpert-style label:
      1 = positive
      0 = negative
     -1 = ambiguous/uncertain
    """
    t = (text or "").strip().lower()

    if t.startswith("yes"):
        return 1
    if t.startswith("no"):
        return 0

    positive_keywords = ["yes", "present", "positive", "consistent with", "suggests", "shows"]
    negative_keywords = ["no", "absent", "negative", "not present", "no evidence", "unremarkable"]

    has_positive = any(kw in t for kw in positive_keywords)
    has_negative = any(kw in t for kw in negative_keywords)

    if has_positive and not has_negative:
        return 1
    if has_negative and not has_positive:
        return 0

    return -1


def compute_metrics(results_df: pd.DataFrame, conditions: List[str]) -> Tuple[Dict[str, float], float]:
    """Compute per-condition F1 and macro F1."""
    f1_scores = {}

    print("\n" + "=" * 60)
    print("Classification metrics")
    print("=" * 60)

    for cond in conditions:
        gt_col = f"gt_{cond}"
        pred_col = f"pred_{cond}"

        y_true = results_df[gt_col].values
        y_pred = results_df[pred_col].values

        # Keep uncertain (-1) predictions as-is to match your current logic;
        # if you later want strict binary F1, map -1 -> 0 before this call.
        f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
        f1_scores[cond] = float(f1)

        n_pos_gt = int((y_true == 1).sum())
        n_pos_pred = int((y_pred == 1).sum())
        print(f"  {cond:20s} | F1={f1:.4f} | GT pos: {n_pos_gt}/{len(y_true)} | Pred pos: {n_pos_pred}/{len(y_pred)}")

    macro_f1 = float(np.mean(list(f1_scores.values()))) if f1_scores else 0.0
    print(f"\n  {'MACRO F1':20s} | {macro_f1:.4f}")
    print("=" * 60)

    return f1_scores, macro_f1


def load_classification_dataset(csv_file: str, max_samples: int = -1) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    print(f"Loaded classification CSV: {len(df)} rows")
    if max_samples > 0:
        df = df.head(max_samples).reset_index(drop=True)
        print(f"Limiting to {max_samples} samples")
    return df


def run_cxr_classification_experiment(args, model, processor, experiment_meta):
    df = load_classification_dataset(args.csv_file, args.max_samples)
    meta = init_experiment_meta(experiment_meta)

    # Validate schema
    required_cols = [args.path_col] + list(args.conditions)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    all_results = []

    print(f"\nRunning inference on {len(df)} images and {len(args.conditions)} conditions...")
    print("-" * 60)

    for idx, row in df.iterrows():
        img_path = os.path.join(args.image_dir, row[args.path_col])
        if not os.path.exists(img_path):
            print(f"[SKIP] Image not found: {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")

        result_row = {
            "path": row[args.path_col],
            "image_size": f"{image.size[0]}x{image.size[1]}",
        }

        print(f"\n[{idx+1}/{len(df)}] {row[args.path_col]} (size: {image.size})")

        for cond in args.conditions:
            gt = int(row[cond])
            result_row[f"gt_{cond}"] = gt

            t0 = time.time()
            response = run_inference_single(model, processor, image, cond)
            elapsed = time.time() - t0

            pred = parse_yes_no(response)

            result_row[f"pred_{cond}"] = pred
            result_row[f"raw_{cond}"] = response
            result_row[f"time_{cond}_s"] = round(elapsed, 3)

        all_results.append(result_row)

        if len(all_results) % args.save_every == 0:
            save_results_with_meta(all_results, args.output_file, meta)

    # final save
    save_results_with_meta(all_results, args.output_file, meta)

    results_df = pd.DataFrame(all_results)
    f1_scores, macro_f1 = compute_metrics(results_df, args.conditions)

    # Store final metrics in metadata JSON on final save
    meta["f1_scores"] = f1_scores
    meta["macro_f1"] = macro_f1
    save_results_with_meta(all_results, args.output_file, meta)

    return results_df, f1_scores, macro_f1