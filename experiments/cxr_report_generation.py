import os
import time
from typing import List

import pandas as pd
from PIL import Image

from core.model import run_inference
from core.utils import init_experiment_meta, save_results_with_meta
from config.prompts import PROMPTS

# TODO:
# - trigger automatic RadGraph eval pipeline hook here
# - fix the printouts


def add_report_gen_args(parser):
    parser.add_argument("--parquet_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="results_report_gen.csv")
    parser.add_argument("--model_id", type=str, default="google/medgemma-4b-it")
    parser.add_argument(
        "--float_type",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--sections", nargs="+", default=["findings", "impression"])


def load_report_gen_dataset(parquet_file: str, max_samples: int = -1) -> pd.DataFrame:
    df = pd.read_parquet(parquet_file)
    print(f"Loaded parquet: {len(df)} rows, columns: {list(df.columns)}")

    df["findings"] = df["findings"].fillna("").astype(str).str.strip()
    df["impression"] = df["impression"].fillna("").astype(str).str.strip()

    mask = (df["findings"].str.len() > 0) | (df["impression"].str.len() > 0)
    n_before = len(df)
    df = df[mask].reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} rows with empty findings AND impression")

    if max_samples > 0:
        df = df.head(max_samples).reset_index(drop=True)
        print(f"Limiting to {max_samples} samples")

    print(f"Final dataset: {len(df)} studies")
    return df


def _print_length_stats(df: pd.DataFrame, sections: List[str]):
    for s in sections:
        gt_len = df[f"{s}_gt"].str.len()
        gen_len = df[f"{s}_gen"].str.len()
        t = df[f"time_{s}_s"]
        print(f"  {s:12s}  GT  chars: mean={gt_len.mean():.0f}  median={gt_len.median():.0f}")
        print(f"  {s:12s}  GEN chars: mean={gen_len.mean():.0f}  median={gen_len.median():.0f}")
        print(f"  {s:12s}  time:      mean={t.mean():.1f}s  total={t.sum():.0f}s")


def run_report_generation_experiment(
    args, model, processor, experiment_meta
    ):
    df = load_report_gen_dataset(args.parquet_file, args.max_samples)
    meta = init_experiment_meta(experiment_meta)

    sections = args.sections
    all_results = []
    total = len(df)

    print(f"\n{'='*70}")
    print(f"Report generation: {total} studies x {len(sections)} sections")
    print(f"Sections: {sections}")
    print(f"{'='*70}\n")

    for _, row in df.iterrows():
        img_path = row["local_image_path"]
        if not os.path.exists(img_path):
            print(f"  [SKIP] Image not found: {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")

        result = {
            "study_id": row["study_id"],
            "dicom_id": row["dicom_id"],
            "subject_id": row["subject_id"],
            "local_image_path": img_path,
        }

        for section in sections:
            result[f"{section}_gt"] = row.get(section, "")

            t0 = time.time()
            gen_text = run_inference(
                model=model,
                processor=processor,
                image=image,
                prompt_text=PROMPTS[section],
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            elapsed = time.time() - t0

            result[f"{section}_gen"] = gen_text
            result[f"time_{section}_s"] = round(elapsed, 2)

        all_results.append(result)
        n_done = len(all_results)

        if n_done == 1 or n_done % 50 == 0 or n_done == total:
            print(f"[{n_done}/{total}] study_id={row['study_id']}")

        if n_done % args.save_every == 0:
            save_results_with_meta(all_results, args.output_file, meta)
            print(f"  >> checkpoint save at {n_done} rows")

    save_results_with_meta(all_results, args.output_file, meta)
    results_df = pd.DataFrame(all_results)

    print(f"\n{'='*70}")
    print(f"DONE — {len(results_df)} studies")
    _print_length_stats(results_df, sections)
    print(f"{'='*70}")

