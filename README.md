# medgemma-replication
Local reproductions and benchmark sanity checks using of the MedGemma collection. This repository tracks the preliminary learning phase of my Master’s Thesis at the Johannes Kepler University Linz, establishing a baseline and model familiarity before moving into my primary research experiments.

# MedGemma Reproduction Study

In this project I reproduce the following MedGemma evaluation experiments from *MedGemma Technical Report* (2025) by Sellergren et al.:

1. **MIMIC-CXR report generation** (findings + impression)
2. **CheXpert classification** (per-condition yes/no)

## Project Structure

```text
*medgemma-thesis/
├── main.py                     # CLI router
├── f1_radgraph.py              # Evaluation for report gen (Requires RadGraph Env)
├── core/                       # Shared backbone
│   ├── model.py                # model loading + generic inference
│   └── utils.py                # GPU setup, metadata, save utils
├── experiments/                # Task-specific logic
│   ├── report_generation.py
│   └── classification.py
└── config/
    └── prompts.py              # centralized prompt registry
```

## Environments & Dependencies

This project requires two separate Conda environments due to conflicting dependency requirements for the RadGraphF1 metric:

`medgemma-main` - use for all general tasks

`medgemma-radgraph` - use only for RadGraphF1 evaluation (f1_radgraph.py)


## Quick Start

### Report generation

```bash
python main.py --gpu 3 report_gen \
  --parquet_file /path/to/mimic_cxr_test.parquet \
  --output_file results_report_gen.csv \
  --model_id google/medgemma-4b-it \
  --float_type bfloat16 \
  --max_samples 10
```

### Classification 

```bash
python main.py --gpu 42 classify \
  --csv_file /path/to/test_labels.csv \
  --image_dir /path/to/images \
  --output_file results_classify.csv
```

### RadGraph F1 Evaluation

```bash
python f1_radgraph.py --gpu 8 --input results_report_gen.csv
```

## Notes

- `--gpu` is parsed before CUDA initialization and mapped to `CUDA_VISIBLE_DEVICES` to facilitate work on shared servers with multiple GPUs.
