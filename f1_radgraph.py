import os
import json
import argparse
import re
from datetime import datetime


def _extract_gpu_arg_early(default: str = "0") -> str:
    """
    Parse --gpu from raw argv *before* importing torch or torch-dependent modules.
    Supports:
      --gpu 3
      --gpu=3
    """
    import sys

    gpu_id = default
    argv = sys.argv[1:]

    for i, tok in enumerate(argv):
        if tok == "--gpu" and i + 1 < len(argv):
            gpu_id = argv[i + 1]
            break
        if tok.startswith("--gpu="):
            gpu_id = tok.split("=", 1)[1]
            break

    return gpu_id


_EARLY_GPU_ID = _extract_gpu_arg_early()
os.environ["CUDA_VISIBLE_DEVICES"] = _EARLY_GPU_ID

from core.utils import print_cuda_info
from radgraph import F1RadGraph
import pandas as pd
import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="F1-RadGraph score for CXR report generation evaluation"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=_EARLY_GPU_ID,
        help="Physical CUDA GPU ID to expose via CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="CSV with either (full_gt/full_gen) or (findings/impression gt/gen)",
    )
    return parser


def clean_generated_text(text):
    """
    Strip markdown headers like **Findings:** or **Impression:** that
    MedGemma can add to output.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = re.sub(r"\*\*(Findings|Impression|Report):\*\*\s*", "", text)
    return text.strip()


def clean_gt_text(text):
    """Clean ground truth text (handle NaN, strip whitespace)."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return text.strip()


def update_existing_json_with_evaluation(csv_path: str, evaluation_payload: dict):
    """
    Update experiment_meta.evaluation in existing JSON with same basename as CSV.
    Will NOT create a new JSON file.
    """
    json_path = os.path.splitext(csv_path)[0] + ".json"

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Expected existing JSON not found: {json_path}\n"
            f"This script updates existing JSON only."
        )

    with open(json_path, "r") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object/dict: {json_path}")

    if "experiment_meta" not in payload or not isinstance(payload["experiment_meta"], dict):
        payload["experiment_meta"] = {}

    payload["experiment_meta"]["evaluation"] = evaluation_payload

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Updated existing JSON: {json_path}")


def has_cols(df: pd.DataFrame, cols):
    return all(c in df.columns for c in cols)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.gpu != _EARLY_GPU_ID:
        print(
            f"[WARN] --gpu parsed late as {args.gpu}, but CUDA was initialized with {_EARLY_GPU_ID}. "
            "Use --gpu at launch time; changing after startup is not supported."
        )

    print_cuda_info()

    df = pd.read_csv(args.input)

    # Use RG_ER only
    f1radgraph = F1RadGraph(reward_level="partial", model_type="radgraph-xl")

    rg_er_findings = None
    rg_er_impr = None
    rg_er_full = None
    schema_mode = None

    # -------- NEW schema preferred: full_gt/full_gen --------
    if has_cols(df, ["full_gt", "full_gen"]):
        schema_mode = "combined_full"

        df["full_gt_clean"] = df["full_gt"].apply(clean_gt_text)
        df["full_gen_clean"] = df["full_gen"].apply(clean_generated_text)

        rg_er_full, _, _, _ = f1radgraph(
            hyps=df["full_gen_clean"].tolist(),
            refs=df["full_gt_clean"].tolist(),
        )
        print(f"RG_ER (Full report): {rg_er_full:.4f}")

        # Optional: if old columns also exist, compute section-level too
        if has_cols(df, ["findings_gt", "findings_gen", "impression_gt", "impression_gen"]):
            df["findings_gt_clean"] = df["findings_gt"].apply(clean_gt_text)
            df["findings_gen_clean"] = df["findings_gen"].apply(clean_generated_text)
            df["impression_gt_clean"] = df["impression_gt"].apply(clean_gt_text)
            df["impression_gen_clean"] = df["impression_gen"].apply(clean_generated_text)

            rg_er_findings, _, _, _ = f1radgraph(
                hyps=df["findings_gen_clean"].tolist(),
                refs=df["findings_gt_clean"].tolist(),
            )
            rg_er_impr, _, _, _ = f1radgraph(
                hyps=df["impression_gen_clean"].tolist(),
                refs=df["impression_gt_clean"].tolist(),
            )
            print(f"RG_ER (Findings): {rg_er_findings:.4f}")
            print(f"RG_ER (Impression): {rg_er_impr:.4f}")

    # -------- OLD schema fallback: findings/impression --------
    elif has_cols(df, ["findings_gt", "findings_gen", "impression_gt", "impression_gen"]):
        schema_mode = "separate_sections"

        df["findings_gt_clean"] = df["findings_gt"].apply(clean_gt_text)
        df["findings_gen_clean"] = df["findings_gen"].apply(clean_generated_text)
        df["impression_gt_clean"] = df["impression_gt"].apply(clean_gt_text)
        df["impression_gen_clean"] = df["impression_gen"].apply(clean_generated_text)

        rg_er_findings, _, _, _ = f1radgraph(
            hyps=df["findings_gen_clean"].tolist(),
            refs=df["findings_gt_clean"].tolist(),
        )
        rg_er_impr, _, _, _ = f1radgraph(
            hyps=df["impression_gen_clean"].tolist(),
            refs=df["impression_gt_clean"].tolist(),
        )
        print(f"RG_ER (Findings): {rg_er_findings:.4f}")
        print(f"RG_ER (Impression): {rg_er_impr:.4f}")

        # Construct full text from separate sections
        df["full_gt_clean"] = (df["findings_gt_clean"] + " " + df["impression_gt_clean"]).str.strip()
        df["full_gen_clean"] = (df["findings_gen_clean"] + " " + df["impression_gen_clean"]).str.strip()

        rg_er_full, _, _, _ = f1radgraph(
            hyps=df["full_gen_clean"].tolist(),
            refs=df["full_gt_clean"].tolist(),
        )
        print(f"RG_ER (Findings + Impression): {rg_er_full:.4f}")

    else:
        raise ValueError(
            "Unsupported CSV schema. Need either:\n"
            "  1) full_gt, full_gen\n"
            "  2) findings_gt, findings_gen, impression_gt, impression_gen"
        )

    vram_gb = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
    print(f"VRAM used: {vram_gb:.2f} GB")

    evaluation_payload = {
        "task": "radgraph_report_generation",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_csv": args.input,
        "schema_mode": schema_mode,
        "n_samples": int(len(df)),
        "model_type": "radgraph-xl",
        "reward_level": "partial",
        "scores": {
            "rg_er_findings": float(rg_er_findings) if rg_er_findings is not None else None,
            "rg_er_impression": float(rg_er_impr) if rg_er_impr is not None else None,
            "rg_er_full_report": float(rg_er_full) if rg_er_full is not None else None,
        },
        "runtime": {
            "gpu_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "N/A"),
            "vram_peak_gb": float(round(vram_gb, 4)),
        },
    }

    update_existing_json_with_evaluation(
        csv_path=args.input,
        evaluation_payload=evaluation_payload,
    )


if __name__ == "__main__":
    main()