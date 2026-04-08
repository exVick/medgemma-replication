import os
import sys
import argparse

def _extract_gpu_arg_early(default: str = "0") -> str:
    """
    Extract GPU early, to limit cuda to only one physical ID (for shared servers with multiple GPUs)
    """
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

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MedGemma thesis experiments: report generation + CXR classification + MedSigLIP embeddings"
    )

    parser.add_argument("--gpu", type=str, default=_EARLY_GPU_ID, help="Physical GPU ID")

    sub = parser.add_subparsers(dest="command", required=True)

    rg = sub.add_parser("report_gen", help="MIMIC-CXR report generation")
    rg.add_argument("--parquet_file", type=str, required=True)
    rg.add_argument("--output_file", type=str, default="results_report_gen.csv")
    rg.add_argument("--model_id", type=str, default="google/medgemma-4b-it")
    rg.add_argument("--float_type", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    rg.add_argument("--use_8bit", action="store_true")
    rg.add_argument("--max_new_tokens", type=int, default=512)
    rg.add_argument("--max_samples", type=int, default=-1, help="-1 = all")
    rg.add_argument("--save_every", type=int, default=50)
    rg.add_argument("--sections", nargs="+", default=["findings", "impression"])

    cxr = sub.add_parser("cxr_classify", help="CheXpert/CXR image classification")
    cxr.add_argument("--csv_file", type=str, required=True)
    cxr.add_argument("--image_dir", type=str, required=True)
    cxr.add_argument("--output_file", type=str, default="results_cxr_classification.csv")
    cxr.add_argument("--model_id", type=str, default="google/medgemma-4b-it")
    cxr.add_argument("--float_type", type=str, default="float32", choices=["bfloat16", "float16", "float32"])
    cxr.add_argument("--use_8bit", type=bool, default=False, help="True if you want 8-bit model")
    cxr.add_argument("--max_samples", type=int, default=-1, help="-1 = all")
    cxr.add_argument("--save_every", type=int, default=50)

    medsiglip = sub.add_parser("medsiglip_emb", help="Generate MedSigLIP embeddings and save to parquet")
    medsiglip.add_argument("--csv-file", dest="csv_file", type=str, required=True)
    medsiglip.add_argument("--image-dir", dest="image_dir", type=str, required=True)
    medsiglip.add_argument("--output-file", dest="output_file", type=str, default="results_medsiglip_emb.parquet")
    medsiglip.add_argument("--model-id", dest="model_id", type=str, default="google/medsiglip-448")
    medsiglip.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    medsiglip.add_argument("--patient-lower", dest="patient_lower", type=int, default=1, help="Lower bound for patient_num filter")
    medsiglip.add_argument("--patient-upper", "--max-patients", dest="patient_upper", type=int, default=8528, help="Upper bound for patient_num filter")
    medsiglip.add_argument("--save-every", dest="save_every", type=int, default=1, help="Save every N batches")

    probe = sub.add_parser("cxr_emb_probe", help="CXR embedding linear probing with logistic regression")
    probe.add_argument("--output-dir", dest="output_dir", type=str, required=True)
    probe.add_argument("--train-parquet-path", dest="train_parquet_path", type=str, required=True)
    probe.add_argument("--val-parquet-path", dest="val_parquet_path", type=str, required=True)
    probe.add_argument("--conditions", nargs="+", default=["Atelectasis", "Cardiomegaly", "Consolidation"])
    probe.add_argument("--sample-sizes", nargs="+", type=int, default=[64])
    probe.add_argument("--min-overage-ratio", dest="min_overage_ratio", type=float, default=1.5)
    probe.add_argument("--c-values", nargs="+", type=float, default=[0.001, 0.01, 0.1, 1.0, 10.0])
    probe.add_argument("--max-iter", dest="max_iter", type=int, default=5000)
    probe.add_argument("--random-state", dest="random_state", type=int, default=42)

    test_probe = sub.add_parser("cxr_test_probe", help="Evaluate trained CXR embedding linear probes on test set")
    test_probe.add_argument("--test-parquet-path", dest="test_parquet_path", type=str, required=True)
    test_probe.add_argument("--csv-file", dest="csv_file", type=str, required=True)
    test_probe.add_argument("--linear-probes-path", dest="linear_probes_path", type=str, required=True)
    test_probe.add_argument("--output-dir", dest="output_dir", type=str, required=True)
    test_probe.add_argument("--output-file", dest="output_file", type=str, default="cxr_test_linear_probing_results.json")

    return parser


def main():
    parser = build_parser()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        print("\nExample:")
        print("  python main.py --gpu 3 report_gen --parquet_file /path/to/test.parquet")
        print("  python main.py --gpu 3 cxr_classify --csv_file /path/to/test.csv --image_dir /path/to/images")
        print("  python main.py --gpu 3 medsiglip_emb --csv-file /path/to/test.csv --image-dir /path/to/images")
        print("  python main.py --gpu 3 cxr_emb_probe --output-dir /path/to/out --train-parquet-path /path/to/train.parquet --val-parquet-path /path/to/val.parquet")
        print("  python main.py --gpu 3 cxr_test_probe --test-parquet-path /path/to/test_emb.parquet --csv-file /path/to/test_labels.csv --linear-probes-path /path/to/probes --output-dir /path/to/out")
        sys.exit(2)

    args = parser.parse_args()

    from core.utils import print_cuda_info
    from core.model import load_model
    from experiments.cxr_report_generation import run_report_generation_experiment
    from experiments.cxr_image_classification import run_cxr_classification_experiment
    from experiments.create_medsiglip_embeddings import run_medsiglip_embeddings_experiment
    from experiments.cxr_emb_linear_probing import run_cxr_emb_linear_probing_experiment
    from experiments.cxr_test_linear_probing import run_cxr_test_linear_probing_experiment

    print_cuda_info()

    if args.command == "medsiglip_emb":
        run_medsiglip_embeddings_experiment(args)
        return
    if args.command == "cxr_emb_probe":
        run_cxr_emb_linear_probing_experiment(args)
        return
    if args.command == "cxr_test_probe":
        run_cxr_test_linear_probing_experiment(args)
        return

    model, processor, meta = load_model(
        model_id=args.model_id,
        float_type=args.float_type if not args.use_8bit else None,
        use_8bit=args.use_8bit,
    )

    if args.command == "report_gen":
        run_report_generation_experiment(args, model, processor, meta)
    elif args.command == "cxr_classify":
        run_cxr_classification_experiment(args, model, processor, meta)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()