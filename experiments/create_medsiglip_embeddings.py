import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


def _load_and_prepare_dataset(csv_file: str, max_patients: int = -1) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    print(f"Loaded CSV: {len(df)} rows")

    if "Path" not in df.columns:
        raise ValueError("Missing required column 'Path' in CSV")

    # Keep only canonical patient IDs and support patient-limited slicing.
    df["patient_num"] = pd.to_numeric(
        df["Path"].str.extract(r"patient(\d+)")[0], errors="coerce"
    )
    df = df[df["patient_num"].between(1, 8528)].copy()

    # Strip first two folders from source paths before joining with image_dir.
    df["Path_short"] = df["Path"].str.replace(r"^(?:[^/]+/){2}", "", regex=True)

    if max_patients > 0:
        selected_patients = (
            df["patient_num"].drop_duplicates().head(max_patients).tolist()
        )
        df = df[df["patient_num"].isin(selected_patients)].copy()
        print(
            f"Applied max_patients={max_patients}: "
            f"{len(selected_patients)} patients, {len(df)} rows"
        )

    df = df.reset_index(drop=True)
    print(
        f"Prepared dataset: {len(df)} rows across "
        f"{df['patient_num'].nunique()} patients"
    )
    return df


def _build_output_paths(output_file: str) -> Tuple[Path, Path]:
    out_path = Path(output_file)

    if out_path.suffix.lower() != ".parquet":
        out_path = out_path.with_suffix(".parquet")

    if str(out_path.parent) != "":
        out_path.parent.mkdir(parents=True, exist_ok=True)

    json_path = out_path.with_suffix(".json")
    return out_path, json_path


def _serialize_for_parquet(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if "embedding" not in df.columns:
        return df

    emb_series = df.pop("embedding")
    if emb_series.empty:
        return df

    emb_dim = len(emb_series.iloc[0]) if len(emb_series.iloc[0]) > 0 else 0
    emb_cols = [f"emb_{i:04d}" for i in range(emb_dim)]
    emb_df = pd.DataFrame(emb_series.tolist(), columns=emb_cols)

    return pd.concat([df.reset_index(drop=True), emb_df], axis=1)


def _save_progress(
    successful_records: List[Dict[str, Any]],
    unsuccessful_records: List[Dict[str, Any]],
    output_parquet: Path,
    output_json: Path,
    meta: Dict[str, Any],
) -> Tuple[Path, Path]:
    if meta.get("start_time") is not None:
        meta["total_runtime_s"] = round(time.time() - meta["start_time"], 2)

    if torch.cuda.is_available():
        meta["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)

    success_patients = {
        int(r["patient_num"])
        for r in successful_records
        if pd.notna(r.get("patient_num"))
    }

    df_out = _serialize_for_parquet(successful_records)
    df_out.to_parquet(output_parquet, index=False)

    payload = {
        "experiment_meta": {k: v for k, v in meta.items() if k != "start_time"},
        "results": {
            "successful_embeddings": {
                "num_images": len(successful_records),
                "num_patients": len(success_patients),
            },
            "unsuccessful_embeddings": unsuccessful_records,
            "output_parquet": str(output_parquet),
        },
    }

    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    return output_parquet, output_json


def _prepare_batch_records(
    batch_df: pd.DataFrame, image_dir: str
) -> Tuple[List[Dict[str, Any]], List[Image.Image], List[Dict[str, Any]]]:
    valid_records: List[Dict[str, Any]] = []
    valid_images: List[Image.Image] = []
    failed_records: List[Dict[str, Any]] = []

    for _, row in batch_df.iterrows():
        source_path = str(row["Path"])
        short_path = str(row["Path_short"])
        full_path = os.path.join(image_dir, short_path)

        row_dict = row.to_dict()
        row_dict["source_image_path"] = source_path
        row_dict["resolved_image_path"] = full_path

        if not os.path.exists(full_path):
            failed_records.append(
                {
                    "path": source_path,
                    "path_short": short_path,
                    "patient_num": int(row["patient_num"]),
                    "reason": "file_not_found",
                }
            )
            continue

        try:
            with Image.open(full_path) as im:
                image = im.convert("RGB")
            valid_records.append(row_dict)
            valid_images.append(image)
        except Exception as exc:
            failed_records.append(
                {
                    "path": source_path,
                    "path_short": short_path,
                    "patient_num": int(row["patient_num"]),
                    "reason": "image_open_error",
                    "error": str(exc),
                }
            )

    return valid_records, valid_images, failed_records


def _embed_batch(processor, model, device, images: List[Image.Image]) -> torch.Tensor:
    inputs = processor(images=images, return_tensors="pt").to(device)
    outputs = model.vision_model(**inputs)
    image_embeds = outputs.pooler_output
    embeddings = F.normalize(image_embeds, p=2, dim=-1)
    return embeddings.cpu()


def run_medsiglip_embeddings_experiment(args):
    output_parquet, output_json = _build_output_paths(args.output_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"Loading MedSigLIP model and processor: {args.model_id}")
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id).to(device)
    model.eval()

    meta = {
        "gpu_id": os.environ.get("CUDA_VISIBLE_DEVICES", "N/A"),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "model_id": args.model_id,
        "batch_size": args.batch_size,
        "max_patients": args.max_patients,
        "save_every_batches": args.save_every,
        "csv_file": args.csv_file,
        "image_dir": args.image_dir,
        "vram_after_load_gb": (
            round(torch.cuda.max_memory_allocated() / 1e9, 3)
            if torch.cuda.is_available()
            else None
        ),
        "vram_peak_gb": None,
        "total_runtime_s": None,
        "run_date": datetime.now().isoformat(),
        "start_time": time.time(),
    }

    df = _load_and_prepare_dataset(args.csv_file, args.max_patients)

    successful_records: List[Dict[str, Any]] = []
    unsuccessful_records: List[Dict[str, Any]] = []

    total_rows = len(df)
    if total_rows == 0:
        print("No rows to process after filtering. Writing empty outputs.")
        parquet_path, json_path = _save_progress(
            successful_records, unsuccessful_records, output_parquet, output_json, meta
        )
        print(f"Saved parquet to {parquet_path}")
        print(f"Saved metadata JSON to {json_path}")
        return

    batch_size = max(1, args.batch_size)
    save_every_batches = max(1, args.save_every)
    num_batches = (total_rows + batch_size - 1) // batch_size

    print(
        f"Starting embedding extraction: {total_rows} rows in {num_batches} batches "
        f"(batch_size={batch_size}, save_every={save_every_batches} batches)"
    )

    with torch.no_grad():
        for batch_idx, start in enumerate(
            tqdm(range(0, total_rows, batch_size), desc="Extracting MedSigLIP embeddings", unit="batch"),
            start=1,
        ):
            end = min(start + batch_size, total_rows)
            batch_df = df.iloc[start:end]

            valid_records, valid_images, failed_records = _prepare_batch_records(
                batch_df, args.image_dir
            )
            unsuccessful_records.extend(failed_records)

            if valid_records:
                try:
                    embeddings_cpu = _embed_batch(processor, model, device, valid_images)

                    for rec, emb in zip(valid_records, embeddings_cpu):
                        rec["embedding"] = emb.tolist()
                        successful_records.append(rec)
                except Exception as batch_exc:
                    # Retry one-by-one to salvage partial progress from a failed batch.
                    for rec, img in zip(valid_records, valid_images):
                        try:
                            emb_cpu = _embed_batch(processor, model, device, [img])[0]
                            rec["embedding"] = emb_cpu.tolist()
                            successful_records.append(rec)
                        except Exception as item_exc:
                            unsuccessful_records.append(
                                {
                                    "path": rec["Path"],
                                    "path_short": rec["Path_short"],
                                    "patient_num": int(rec["patient_num"]),
                                    "reason": "embedding_error",
                                    "error": f"batch_error={batch_exc}; item_error={item_exc}",
                                }
                            )

            for img in valid_images:
                img.close()

            should_save = (batch_idx % save_every_batches == 0) or (batch_idx == num_batches)
            if should_save:
                _save_progress(
                    successful_records,
                    unsuccessful_records,
                    output_parquet,
                    output_json,
                    meta,
                )

    parquet_path, json_path = _save_progress(
        successful_records,
        unsuccessful_records,
        output_parquet,
        output_json,
        meta,
    )

    success_patients = pd.Series([r["patient_num"] for r in successful_records]).nunique() if successful_records else 0

    print("\nExtraction complete.")
    print(f"Successful image embeddings: {len(successful_records)}")
    print(f"Patients with successful embeddings: {int(success_patients)}")
    print(f"Unsuccessful embeddings: {len(unsuccessful_records)}")
    print(f"Saved parquet to {parquet_path}")
    print(f"Saved metadata JSON to {json_path}")
