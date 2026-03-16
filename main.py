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
from core.model import load_model
from experiments.cxr_report_generation import (
    add_report_gen_args,
    run_report_generation_experiment,
)
from experiments.cxr_image_classification import (
    add_cxr_classification_args,
    run_cxr_classification_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MedGemma thesis experiments: report generation + CXR classification"
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default=_EARLY_GPU_ID,  # reflects what was already applied
        help="Physical CUDA GPU ID to expose via CUDA_VISIBLE_DEVICES",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    rg = sub.add_parser("report_gen", help="MIMIC-CXR report generation")
    add_report_gen_args(rg)

    cxr = sub.add_parser("cxr_classify", help="CheXpert/CXR image classification")
    add_cxr_classification_args(cxr)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.gpu != _EARLY_GPU_ID:
        print(
            f"[WARN] --gpu parsed late as {args.gpu}, but CUDA was initialized with {_EARLY_GPU_ID}. "
            "Use --gpu at launch time; changing after startup is not supported."
        )

    print_cuda_info()

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