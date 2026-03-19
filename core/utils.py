import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import torch


def print_cuda_info() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "N/A")
    print(f"Using GPU with ID:\t{visible}")
    print(f"CUDA Available:\t\t{torch.cuda.is_available()}")

    if torch.cuda.is_available():
        cur = torch.cuda.current_device()
        print(f"GPU name:\t\t{torch.cuda.get_device_name(cur)}")
    else:
        print("WARNING: No CUDA device available — inference will be very slow or fail.")
    print()


def init_experiment_meta(base_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize run-level metadata shared by all experiments.
    """
    meta = {
        "gpu_id": os.environ.get("CUDA_VISIBLE_DEVICES", "N/A"),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "model_id": base_meta.get("model_id"),
        "float_type": base_meta.get("float_type"),
        "use_8bit": base_meta.get("use_8bit"),
        "vram_after_load_gb": base_meta.get("vram_after_load_gb"),
        "vram_peak_gb": None,
        "total_runtime_s": None,
        "start_time": time.time(),
    }
    return meta

#### cool feature but made too much unnecessary copies during checkpoint safety ####
#### turning it off for now, but keeping it just in case ###########################
# def check_path(path: str, file: str, i: int = 1) -> Path:
#     """
#     Prevent overwriting existing files by appending suffixes.
#     """
#     file_path = Path(path, file)
#     if os.path.exists(file_path):
#         new_file = f"{file.rsplit(f'_{i}', 1)[0]}_{i+1}.{file.rsplit('.', 1)[1]}"
#         print(f"File exists: {file_path} -> creating new name: {new_file}")
#         return check_path(path, new_file, i+1)
#     return file_path
####################################################################################


def save_results_with_meta(
    results: List[dict],
    output_file: str,
    experiment_meta: Dict[str, Any],
) -> None:
    """
    Save CSV + JSON payload with metadata.
    """
    experiment_meta["vram_peak_gb"] = (
        round(torch.cuda.max_memory_allocated() / 1e9, 3)
        if torch.cuda.is_available()
        else None
    )

    if experiment_meta.get("start_time") is not None:
        experiment_meta["total_runtime_s"] = round(
            time.time() - experiment_meta["start_time"], 2
        )

    df = pd.DataFrame(results)
    current_month_day = datetime.now().strftime("%m-%d")
    path, file = os.path.dirname(output_file), os.path.basename(output_file)
    
    csv_name = current_month_day + "_" + file
    # csv_path = check_path(path, csv_name)
    csv_path = Path(path, csv_name)
    df.to_csv(csv_path, index=False)

    json_name = current_month_day + "_" + file.rsplit(".", 1)[0] + ".json"
    # json_path = check_path(path, json_name)
    json_path = Path(path, json_name)
    payload = {
        "experiment_meta": {
            k: v for k, v in experiment_meta.items() if k != "start_time"
        },
        "results": results,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    return csv_path, json_path