import os
import argparse

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
import re


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="F1-RadGraph score for CXR report generation evaluation"
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default=_EARLY_GPU_ID,  # reflects what was already applied
        help="Physical CUDA GPU ID to expose via CUDA_VISIBLE_DEVICES",
    )

    parser.add_argument("--csv_file", type=str, required=True, help="CSV with findings and impression for gt and gen")

    return parser

def clean_generated_text(text):
    """
    Strip markdown headers like **Findings:**\n\n or **Impression:**\n\n
    that MedGemma adds to its output.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    # Remove markdown-style section headers
    text = re.sub(r'\*\*(Findings|Impression|Report):\*\*\s*', '', text)
    text = text.strip()
    return text

def clean_gt_text(text):
    """Clean ground truth text (handle NaN, strip whitespace)."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return text.strip()


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.gpu != _EARLY_GPU_ID:
        print(
            f"[WARN] --gpu parsed late as {args.gpu}, but CUDA was initialized with {_EARLY_GPU_ID}. "
            "Use --gpu at launch time; changing after startup is not supported."
        )
    
    print_cuda_info()

    df = pd.read_csv(args.csv_file)

    df['findings_gt_clean'] = df['findings_gt'].apply(clean_gt_text)
    df['findings_gen_clean'] = df['findings_gen'].apply(clean_generated_text)
    df['impression_gt_clean'] = df['impression_gt'].apply(clean_gt_text)
    df['impression_gen_clean'] = df['impression_gen'].apply(clean_generated_text)

    # Initialize F1RadGraph with RadGraph-XL (the model used in modern evaluations)
    # reward_level="all" returns all three levels: (simple, partial, complete)
    f1radgraph = F1RadGraph(reward_level="all", model_type="radgraph-xl")

    # Evaluate on FINDINGS section
    refs_findings = df['findings_gt_clean'].tolist()
    hyps_findings = df['findings_gen_clean'].tolist()

    mean_reward_f, _, _, _ = f1radgraph(hyps=hyps_findings, refs=refs_findings)
    _, rg_er_findings, _ = mean_reward_f

    print(f"RG_ER (Findings): {rg_er_findings:.4f}")

    # Evaluate on IMPRESSION section 
    refs_impression = df['impression_gt_clean'].tolist()
    hyps_impression = df['impression_gen_clean'].tolist()

    mean_reward_i, _, _, _ = f1radgraph(hyps=hyps_impression, refs=refs_impression)
    _, rg_er_impr, _ = mean_reward_i

    print(f"RG_ER (Impression): {rg_er_impr:.4f}")

    # Evaluate on concatenated FINDINGS + IMPRESSION (full report) 
    df['full_gt'] = (df['findings_gt_clean'] + " " + df['impression_gt_clean']).str.strip()
    df['full_gen'] = (df['findings_gen_clean'] + " " + df['impression_gen_clean']).str.strip()

    mean_reward_full, _, _, _ = f1radgraph(
        hyps=df['full_gen'].tolist(),
        refs=df['full_gt'].tolist()
    )
    _, rg_er_full, _ = mean_reward_full

    print(f"RG_ER (Findings + Impression): {rg_er_full:.4f}")

    vram_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"VRAM used: {vram_gb:.2f} GB")

if __name__ == "__main__":
    main()