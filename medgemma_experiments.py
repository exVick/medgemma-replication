"""
MedGemma Experiments — CheXpert Classification & MIMIC-CXR Report Generation
=============================================================================

Unified script for two MedGemma evaluation tasks:
  1. CheXpert chest X-ray classification (per-condition yes/no)
  2. MIMIC-CXR report generation (findings + impression)

Both tasks share model loading, image processing, and inference infrastructure.
Report generation results are saved for later RadGraph F1 evaluation in a
separate environment.

Usage (report generation):
    python medgemma_experiments.py --gpu 3 report_gen \
        --parquet_file /path/to/mimic_cxr_test.parquet \
        --output_file results_report_gen.csv \
        --model_id google/medgemma-4b-it \
        --float_type bfloat16 \
        --max_samples 10

Usage (classification — placeholder, your existing code):
    python medgemma_experiments.py classify \
        --csv_file /path/to/test_labels.csv \
        --output_file results_classify.csv \
        ...
"""

import os
import sys

def _extract_gpu_arg() -> str:
    """
    Scan sys.argv for --gpu <ID>, remove both tokens, return the ID.
    Returns "0" if --gpu is not present.
    """
    gpu_id = "0"                       # default: first visible GPU
    if "--gpu" in sys.argv:
        idx = sys.argv.index("--gpu")
        if idx + 1 < len(sys.argv):
            gpu_id = sys.argv[idx + 1]
            sys.argv.pop(idx)          # remove '--gpu'
            sys.argv.pop(idx)          # remove the value (now at same index)
        else:
            print("ERROR: --gpu requires a value (e.g. --gpu 3)")
            sys.exit(1)
    return gpu_id

_GPU_ID = _extract_gpu_arg()
os.environ["CUDA_VISIBLE_DEVICES"] = _GPU_ID

import torch
print(f"CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Count:      {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU Name:       {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No CUDA device available — inference will be very slow or fail.")
print()

import time
from datetime import datetime
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional, Callable

import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

EXPERIMENT_META = {
    "gpu_id": _GPU_ID,
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    "model_id": None,
    "float_type": None,
    "use_8bit": None,
    "vram_after_load_gb": None,
    "vram_peak_gb": None,
    "total_runtime_s": None,
    "start_time": None,
}

# ============================================================
# 1. MODEL LOADING — Shared across all experiments
# ============================================================

def load_model(
    model_id: str = "google/medgemma-4b-it",
    float_type: Optional[str] = "bfloat16",
    use_8bit: bool = False,
):
    """
    Load MedGemma model and processor.

    Args:
        model_id:    HuggingFace model identifier.
        float_type:  One of "bfloat16", "float16", "float32", or None.
        use_8bit:    If True, load with 8-bit quantization (overrides float_type).

    Returns:
        (model, processor) tuple, model already in eval mode on CUDA.
    """
    quant_config = None
    if use_8bit:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        float_type = None          # dtype is handled by bitsandbytes

    # Reset peak tracking so it only reflects this load + inference
    torch.cuda.reset_peak_memory_stats()

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=float_type,
        quantization_config=quant_config,
        device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()

    vram_after_load = torch.cuda.max_memory_allocated() / 1e9

    # Record in global metadata
    EXPERIMENT_META["model_id"]            = model_id
    EXPERIMENT_META["float_type"]          = str(float_type)
    EXPERIMENT_META["use_8bit"]            = use_8bit
    EXPERIMENT_META["vram_after_load_gb"]  = round(vram_after_load, 3)

    print(f"Loaded '{model_id}' | dtype={float_type} | 8-bit={use_8bit} | VRAM after load: {vram_after_load:.2f} GB")
    return model, processor


# ============================================================
# 2. GENERIC SINGLE-IMAGE INFERENCE
# ============================================================
# This is the *one* inference function that every experiment calls.
# It takes a PIL image + a prompt string, runs the model, and
# returns the raw generated text.  All experiment-specific logic
# (prompt construction, output parsing) lives *outside* this
# function, so you can reuse it for any future task (VQA,
# grounding, etc.) just by changing the prompt.
# ============================================================

def run_inference(
    model,
    processor,
    image: Image.Image,
    prompt_text: str,
    max_new_tokens: int = 512,
    do_sample: bool = False,
) -> str:
    """
    Run single-image inference with an arbitrary text prompt.

    Args:
        model:          Loaded MedGemma model.
        processor:      Loaded processor.
        image:          PIL Image (RGB).
        prompt_text:    The user-turn text (image placeholder is added automatically).
        max_new_tokens: Maximum tokens to generate.
        do_sample:      False → greedy (deterministic); True → sampling.

    Returns:
        Generated text string (prompt tokens stripped).
    """
    # Build chat-format messages — image placeholder + text
    messages = [
        # {
        # "role": "system",
        # "content": [{"type": "text", "text": "You are an expert radiologist."}]
        # },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Step 1: chat template → raw prompt string (not yet tokenized)
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    # Step 2: tokenize text + preprocess image in one call
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    ).to(model.device)

    # Step 3: match pixel_values dtype to model
    # With 8-bit quantization the vision encoder runs in float32;
    # with bfloat16/float16 it expects the matching half type.
    # Safest: cast to whatever the model's embedding layer uses.
    model_dtype = next(model.parameters()).dtype
    inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

    generated_ids = output_ids[0][input_len:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


# ============================================================
# 3. PROMPT REGISTRY — all prompts in one place
# ============================================================
# Every experiment registers its prompts here so they are easy
# to inspect, modify, and cite.  New experiments just add a key.
#
# WHY THESE PROMPTS?
# ------------------------------------------------------------------
# MedGemma was trained on CheXinstruct-style data.  The training
# templates (Stanford-AIMI/CheXagent, data_chexinstruct/templates.py)
# include prompts like:
#   • "Write an example findings section for the CXR"
#   • "Describe the findings in this chest X-ray."
# So the model responds well to this phrasing at inference time.
#
# We use TWO separate calls (findings, then impression) because:
#   1. RadGraph F1 is computed per-section — we need clean text
#      for each section independently.
#   2. A single "write the full report" prompt would require us
#      to *parse* where findings ends and impression begins in
#      free text, which is fragile and error-prone.
#   3. Separate calls let the model focus on one section at a time,
#      matching how it was trained (section-level QA pairs).
#   4. This is the standard approach in CXR report-gen evaluation
#      (LLaVA-Rad, RadVLM, MAIRA-2 all do this).
# ------------------------------------------------------------------

PROMPTS = {
    # --- Report generation (Section 3.5 of the MedGemma tech report) ---
    "findings": (
        "You are provided with a chest X-ray image. "
        "Write the findings section of the radiology report for this chest X-ray."
    ),
    "impression": (
        "You are provided with a chest X-ray image. "
        "Write the impression section of the radiology report for this chest X-ray."
    ),
    # --- Classification (Section 3.4 — per-condition binary) ---
    # The {condition} placeholder is filled at call time.
    "classify_condition": (
        "Does this chest X-ray show {condition}? Answer yes or no."
    ),
}


# ============================================================
# 4. REPORT GENERATION EXPERIMENT
# ============================================================
#
# Input : your preprocessed parquet with columns
#         ['study_id', 'subject_id', 'raw_text', 'impression',
#          'findings', 'comparison', 'indication', 'view_positions',
#          'dicom_id', 'local_image_path']
#
# Output: CSV (+ JSON mirror) with columns
#         study_id, dicom_id, subject_id, local_image_path,
#         findings_gt, impression_gt,
#         findings_gen, impression_gen,
#         time_findings_s, time_impression_s
#
# The output file is the input for RadGraph F1 evaluation.
# ============================================================

def load_report_gen_dataset(
    parquet_file: str,
    max_samples: int = -1,
) -> pd.DataFrame:
    """
    Load the preprocessed MIMIC-CXR test parquet.

    Expects columns:
        study_id, subject_id, raw_text, impression, findings,
        comparison, indication, view_positions, dicom_id,
        local_image_path

    We treat 'findings' and 'impression' as ground truth.
    Rows with *both* empty are dropped (nothing to evaluate).
    """
    df = pd.read_parquet(parquet_file)
    print(f"Loaded parquet: {len(df)} rows, columns: {list(df.columns)}")

    # Clean GT text
    df["findings"]   = df["findings"].fillna("").astype(str).str.strip()
    df["impression"] = df["impression"].fillna("").astype(str).str.strip()

    # Drop rows where BOTH are empty — no GT to compare against
    mask = (df["findings"].str.len() > 0) | (df["impression"].str.len() > 0)
    n_before = len(df)
    df = df[mask].reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} rows with empty findings AND impression")

    if max_samples > 0:
        df = df.head(max_samples).reset_index(drop=True)
        print(f"Limiting to {max_samples} samples")

    print(f"Final dataset: {len(df)} studies")
    return df


def run_report_generation(
    model,
    processor,
    df: pd.DataFrame,
    output_file: str,
    sections: List[str] = None,
    max_new_tokens: int = 512,
    save_every: int = 50,
):
    """
    Generate findings and/or impression for every row in df.

    Args:
        model, processor:  loaded MedGemma.
        df:                DataFrame from load_report_gen_dataset().
        output_file:       where to write the results CSV.
        sections:          which sections to generate, default ["findings", "impression"].
        max_new_tokens:    generation budget per section.
        save_every:        flush to disk every N rows.
    """
    if sections is None:
        sections = ["findings", "impression"]

    EXPERIMENT_META["start_time"] = time.time() 

    all_results = []
    total = len(df)

    print(f"\n{'='*70}")
    print(f"Report generation: {total} studies x {len(sections)} sections")
    print(f"Sections: {sections}")
    print(f"Output: {output_file}")
    print(f"{'='*70}\n")

    for idx, row in df.iterrows():
        img_path = row["local_image_path"]

        if not os.path.exists(img_path):
            print(f"  [SKIP] Image not found: {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")

        result = {
            "study_id":         row["study_id"],
            "dicom_id":         row["dicom_id"],
            "subject_id":       row["subject_id"],
            "local_image_path": img_path,
        }

        for section in sections:
            gt_col  = section                      # 'findings' or 'impression'
            gen_col = f"{section}_gen"
            time_col = f"time_{section}_s"

            result[f"{section}_gt"] = row.get(gt_col, "")

            t0 = time.time()
            gen_text = run_inference(
                model, processor, image,
                prompt_text=PROMPTS[section],
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            elapsed = time.time() - t0

            result[gen_col]  = gen_text
            result[time_col] = round(elapsed, 2)

        all_results.append(result)
        n_done = len(all_results)

        # ---- progress log ----
        if n_done == 1 or n_done % 50 == 0 or n_done == total:
            print(f"[{n_done}/{total}]  study_id={row['study_id']}")
            for section in sections:
                gt_snip  = result[f"{section}_gt"][:90]
                gen_snip = result[f"{section}_gen"][:90]
                secs     = result[f"time_{section}_s"]
                print(f"  {section:12s} GT:  {gt_snip}...")
                print(f"  {section:12s} GEN: {gen_snip}...  ({secs:.1f}s)")

        # ---- incremental save ----
        if n_done % save_every == 0:
            _save_results(all_results, output_file)
            print(f"  >> saved {n_done} rows to {output_file}")

    # ---- final save ----
    _save_results(all_results, output_file)
    results_df = pd.DataFrame(all_results)

    print(f"\n{'='*70}")
    print(f"DONE — {len(results_df)} studies")
    print(f"Results: {output_file}")
    _print_length_stats(results_df, sections)
    print(f"{'='*70}")

    return results_df


# ============================================================
# 5. UTILITIES
# ============================================================

def _check_path(path, file, i=1):
    """
    Prevent from overwriting files
    """
    file_path = Path(path, file)
    if os.path.exists(file_path):
        new_file = f"{file.rsplit('.', 1)[0]}_{i}.{file.rsplit('.', 1)[1]}"
        print(f"The file with path {file_path} already exists -> creating a new name: {new_file}")
        return _check_path(path, new_file, i + 1)
    return file_path

def _save_results(results: list, output_file: str):
    """Write results as CSV + JSON, both including experiment metadata."""

    # Snapshot peak VRAM and runtime at every save
    EXPERIMENT_META["vram_peak_gb"] = round(
        torch.cuda.max_memory_allocated() / 1e9, 3
    ) if torch.cuda.is_available() else None

    if EXPERIMENT_META["start_time"] is not None:
        EXPERIMENT_META["total_runtime_s"] = round(
            time.time() - EXPERIMENT_META["start_time"], 2
        )

    # ---- CSV: metadata as comment header + data rows ----
    df = pd.DataFrame(results)
    current_month_day = datetime.now().strftime("%m-%d")

    path, file = os.path.dirname(output_file), os.path.basename(output_file)
    file = current_month_day +  "_" + file

    file_path = _check_path(path, file)
    df.to_csv(file_path, index=False)
    print(f"Saved to {file_path}")

    # ---- JSON: metadata as top-level key alongside rows ----
    json_path = output_file.rsplit(".", 1)[0] + ".json"
    payload = {
        "experiment_meta": {
            k: v for k, v in EXPERIMENT_META.items() if k != "start_time"
        },
        "results": results,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _print_length_stats(df: pd.DataFrame, sections: List[str]):
    """Quick sanity-check summary."""
    for s in sections:
        gt_len  = df[f"{s}_gt"].str.len()
        gen_len = df[f"{s}_gen"].str.len()
        t       = df[f"time_{s}_s"]
        print(f"  {s:12s}  GT  chars: mean={gt_len.mean():.0f}  median={gt_len.median():.0f}")
        print(f"  {s:12s}  GEN chars: mean={gen_len.mean():.0f}  median={gen_len.median():.0f}")
        print(f"  {s:12s}  time:      mean={t.mean():.1f}s  total={t.sum():.0f}s")


# ============================================================
# 6. CLI
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="MedGemma experiments: classification & report generation"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- report_gen sub-command ----
    rg = sub.add_parser("report_gen", help="MIMIC-CXR report generation")
    rg.add_argument("--parquet_file", type=str, required=True,
                     help="Preprocessed MIMIC-CXR test parquet")
    rg.add_argument("--output_file", type=str, default="results_report_gen.csv")
    rg.add_argument("--model_id", type=str, default="google/medgemma-4b-it")
    rg.add_argument("--float_type", type=str, default="bfloat16",
                     choices=["bfloat16", "float16", "float32"])
    rg.add_argument("--use_8bit", action="store_true")
    rg.add_argument("--max_new_tokens", type=int, default=512)
    rg.add_argument("--max_samples", type=int, default=-1,
                     help="-1 = all")
    rg.add_argument("--save_every", type=int, default=50)
    rg.add_argument("--sections", nargs="+", default=["findings", "impression"],
                     help="Which sections to generate")

    # ---- classify sub-command (placeholder for your existing code) ----
    cl = sub.add_parser("classify", help="CheXpert classification")
    cl.add_argument("--csv_file", type=str, required=True)
    cl.add_argument("--image_dir", type=str, required=True)
    cl.add_argument("--output_file", type=str, default="results_classify.csv")
    cl.add_argument("--model_id", type=str, default="google/medgemma-4b-it")
    cl.add_argument("--float_type", type=str, default="float32",
                     choices=["bfloat16", "float16", "float32"])
    cl.add_argument("--use_8bit", action="store_true")
    cl.add_argument("--max_samples", type=int, default=-1)
    cl.add_argument("--only_frontal", action="store_true")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "report_gen":
        # Load data
        df = load_report_gen_dataset(
            parquet_file=args.parquet_file,
            max_samples=args.max_samples,
        )

        # Load model
        model, processor = load_model(
            model_id=args.model_id,
            float_type=args.float_type if not args.use_8bit else None,
            use_8bit=args.use_8bit,
        )

        # Run
        run_report_generation(
            model=model,
            processor=processor,
            df=df,
            output_file=args.output_file,
            sections=args.sections,
            max_new_tokens=args.max_new_tokens,
            save_every=args.save_every,
        )

    elif args.command == "classify":
        # ----- Your existing classification code goes here -----
        # For now, just a placeholder showing how it fits.
        print("Classification sub-command: integrate your existing code here.")
        print("The shared pieces are: load_model(), run_inference(), PROMPTS.")
        # Example of how classification uses the shared run_inference():
        #
        #   for condition in conditions:
        #       prompt = PROMPTS["classify_condition"].format(condition=condition.lower())
        #       response = run_inference(model, processor, image, prompt,
        #                                max_new_tokens=64, do_sample=False)
        #       pred = parse_yes_no(response)


if __name__ == "__main__":
    main()