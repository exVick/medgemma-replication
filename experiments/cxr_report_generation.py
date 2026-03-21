import os
import time
from typing import List
from tqdm import tqdm

import pandas as pd
from PIL import Image

from core.model import run_inference
from core.utils import init_experiment_meta, save_results_with_meta
from config.prompts import PROMPTS


def load_report_gen_dataset(parquet_file: str, max_samples: int = -1) -> pd.DataFrame:
    df = pd.read_parquet(parquet_file)
    print(f"Loaded parquet: {len(df)} rows")

    # normalize text cols
    for c in ["findings", "impression", "indication"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip()
        else:
            df[c] = ""

    # Keep rows with any GT report text
    mask = (df["findings"].str.len() > 0) | (df["impression"].str.len() > 0)
    df = df[mask].reset_index(drop=True)

    if max_samples > 0:
        df = df.head(max_samples).reset_index(drop=True)

    print(f"Final dataset: {len(df)} studies")
    return df


def _build_prompt_from_indication(indication: str) -> str:
    indication = (indication or "").strip()
    # Avoid empty leading prompt
    if not indication:
        indication = "Chest x-ray"
    return PROMPTS["findings_and_impression"].format(indication=indication)

    
def _print_length_stats(df: pd.DataFrame, sections: List[str]):
    for s in sections:
        gt_len = df[f"{s}_gt"].str.len()
        gen_len = df[f"{s}_gen"].str.len()
        t = df[f"time_{s}_s"]
        print(f"  {s:12s}  GT  chars: mean={gt_len.mean():.0f}  median={gt_len.median():.0f}")
        print(f"  {s:12s}  GEN chars: mean={gen_len.mean():.0f} median={gen_len.median():.0f}")
        print(f"  {s:12s}  time:      mean={t.mean():.1f}s      total={t.sum():.0f}s")


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

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Doctor MedGemma is generating radiology reports...",
        unit="img",
    ):
        img_path = row["local_image_path"]
        if not os.path.exists(img_path):
            print(f"  [SKIP] Image not found: {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")

        indication = row.get("indication", "")
        prompt = _build_prompt_from_indication(indication)

        findings_gt = row.get("findings", "")
        impression_gt = row.get("impression", "")
        full_gt = f"{findings_gt} {impression_gt}".strip()

        t0 = time.time()
        full_gen = run_inference(
            model=model,
            processor=processor,
            image=image,
            prompt_text=prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        elapsed = time.time() - t0

        result = {
            "study_id": row.get("study_id", ""),
            "dicom_id": row.get("dicom_id", ""),
            "subject_id": row.get("subject_id", ""),
            "local_image_path": img_path,
            "indication": indication,
            "prompt_used": prompt,

            # keep original GT columns (needed for downstream scripts if any)
            "findings_gt": findings_gt,
            "impression_gt": impression_gt,

            # new combined output
            "full_gt": full_gt,
            "full_gen": full_gen,
            "time_full_s": round(elapsed, 2),
        }
        
        all_results.append(result)
        n_done = len(all_results)

        # if n_done == 1 or n_done % 50 == 0 or n_done == total:
        #     print(f"[{n_done}/{total}] study_id={row['study_id']}")

        if n_done % args.save_every == 0:
            save_results_with_meta(all_results, args.output_file, meta)

    csv_path, json_path = save_results_with_meta(all_results, args.output_file, meta)
    print(f"\nSaved CSV to {csv_path}")
    print(f"\nSaved JSON to {json_path}")

    results_df = pd.DataFrame(all_results)

    print(f"\n{'='*70}")
    print(f"DONE — {len(results_df)} studies")
    _print_length_stats(results_df, sections)
    print(f"{'='*70}")

