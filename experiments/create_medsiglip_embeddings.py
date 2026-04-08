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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel


def _load_and_prepare_dataset(
    csv_file: str,
    patient_lower: int = 1,
    patient_upper: int = 8528,
) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    print(f"Loaded CSV: {len(df)} rows")

    if "Path" not in df.columns:
        raise ValueError("Missing required column 'Path' in CSV")

    # Keep only canonical patient IDs and support patient-limited slicing.
    df["patient_num"] = pd.to_numeric(
        df["Path"].str.extract(r"patient(\d+)")[0], errors="coerce"
    )
    if patient_lower > patient_upper:
        raise ValueError(
            f"Invalid patient range: lower={patient_lower} > upper={patient_upper}"
        )

    df = df[df["patient_num"].between(patient_lower, patient_upper)].copy()
    filter_note = f"between({patient_lower}, {patient_upper})"

    # Strip first two folders from source paths before joining with image_dir.
    df["Path_short"] = df["Path"].str.extract(r"(patient.*)", expand=False)

    print(f"Applied patient_num filter: {filter_note} -> {len(df)} rows")

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
    write_parquet: bool,
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

    if write_parquet:
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
            "parquet_written": write_parquet,
        },
    }

    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    return output_parquet, output_json


class _ImagePathDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], image_dir: str):
        self.rows = rows
        self.image_dir = image_dir

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row_dict = dict(self.rows[idx])
        source_path = str(row_dict["Path"])
        short_path = str(row_dict["Path_short"])
        full_path = os.path.join(self.image_dir, short_path)

        row_dict["source_image_path"] = source_path
        row_dict["resolved_image_path"] = full_path

        if not os.path.exists(full_path):
            return {
                "ok": False,
                "failure": {
                    "path": source_path,
                    "path_short": short_path,
                    "patient_num": int(row_dict["patient_num"]),
                    "reason": "file_not_found",
                },
            }

        try:
            with Image.open(full_path) as im:
                image = im.convert("RGB")
            return {"ok": True, "row": row_dict, "image": image}
        except Exception as exc:
            return {
                "ok": False,
                "failure": {
                    "path": source_path,
                    "path_short": short_path,
                    "patient_num": int(row_dict["patient_num"]),
                    "reason": "image_open_error",
                    "error": str(exc),
                },
            }


def _collate_loaded_items(
    batch: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Image.Image], List[Dict[str, Any]]]:
    valid_records: List[Dict[str, Any]] = []
    valid_images: List[Image.Image] = []
    failed_records: List[Dict[str, Any]] = []

    for item in batch:
        if item.get("ok"):
            valid_records.append(item["row"])
            valid_images.append(item["image"])
        else:
            failed_records.append(item["failure"])

    return valid_records, valid_images, failed_records


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
    cpu_inputs = processor(images=images, return_tensors="pt")
    inputs = {
        name: tensor.to(device, non_blocking=True)
        for name, tensor in cpu_inputs.items()
    }
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

    patient_lower = int(getattr(args, "patient_lower", 1))
    patient_upper = int(
        getattr(args, "patient_upper", getattr(args, "max_patients", 8528))
    )

    meta = {
        "gpu_id": os.environ.get("CUDA_VISIBLE_DEVICES", "N/A"),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "model_id": args.model_id,
        "batch_size": args.batch_size,
        "patient_lower": patient_lower,
        "patient_upper": patient_upper,
        "save_every_batches": args.save_every,
        "num_workers": None,
        "prefetch_factor": None,
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

    df = _load_and_prepare_dataset(
        args.csv_file,
        patient_lower=patient_lower,
        patient_upper=patient_upper,
    )

    successful_records: List[Dict[str, Any]] = []
    unsuccessful_records: List[Dict[str, Any]] = []

    total_rows = len(df)
    if total_rows == 0:
        print("No rows to process after filtering. Writing empty outputs.")
        parquet_path, json_path = _save_progress(
            successful_records,
            unsuccessful_records,
            output_parquet,
            output_json,
            meta,
            write_parquet=True,
        )
        print(f"Saved parquet to {parquet_path}")
        print(f"Saved metadata JSON to {json_path}")
        return

    batch_size = max(1, args.batch_size)
    save_every_batches = max(1, args.save_every)

    default_workers = max(1, min(8, (os.cpu_count() or 4) // 2))
    num_workers = max(0, int(getattr(args, "num_workers", default_workers)))
    prefetch_factor = max(2, int(getattr(args, "prefetch_factor", 4)))
    pin_memory = bool(getattr(args, "pin_memory", torch.cuda.is_available()))

    meta["num_workers"] = num_workers
    meta["prefetch_factor"] = prefetch_factor if num_workers > 0 else None

    dataset = _ImagePathDataset(df.to_dict("records"), args.image_dir)
    loader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": _collate_loaded_items,
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = True

    dataloader = DataLoader(**loader_kwargs)
    num_batches = len(dataloader)

    print(
        f"Starting embedding extraction: {total_rows} rows in {num_batches} batches "
        f"(batch_size={batch_size}, save_every={save_every_batches} batches, "
        f"num_workers={num_workers}, pin_memory={pin_memory})"
    )

    current_month_day = datetime.now().strftime("%m-%d")
    
    path, file = os.path.dirname(output_parquet), os.path.basename(output_parquet)
    parquet_name = current_month_day + "_" + file
    output_parquet = Path(path, parquet_name)

    path, file = os.path.dirname(output_json), os.path.basename(output_json)
    json_name = current_month_day + "_" + file
    output_json = Path(path, json_name)

    with torch.inference_mode():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc="Extracting MedSigLIP embeddings", unit="batch"),
            start=1,
        ):
            valid_records, valid_images, failed_records = batch
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
                    write_parquet=False,
                )

    parquet_path, json_path = _save_progress(
        successful_records,
        unsuccessful_records,
        output_parquet,
        output_json,
        meta,
        write_parquet=True,
    )

    success_patients = pd.Series([r["patient_num"] for r in successful_records]).nunique() if successful_records else 0

    print("\nExtraction complete.")
    print(f"Successful image embeddings: {len(successful_records)}")
    print(f"Patients with successful embeddings: {int(success_patients)}")
    print(f"Unsuccessful embeddings: {len(unsuccessful_records)}")
    print(f"Saved parquet to {parquet_path}")
    print(f"Saved metadata JSON to {json_path}")
