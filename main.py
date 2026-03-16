import argparse

from core.utils import extract_and_set_gpu, print_cuda_info
from core.model import load_model
from experiments.report_generation import (
    add_report_gen_args,
    run_report_generation_experiment,
)
from experiments.classification import (
    add_classification_args,
    run_classification_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MedGemma evaluation reproduction"
    )

    # Global/shared args (available to all subcommands)
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="CUDA device ID to expose as CUDA_VISIBLE_DEVICES (default: 0)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # Register task-specific parsers
    rg = sub.add_parser("report_gen", help="Chest X-ray report generation: MIMIC-CXR")
    add_report_gen_args(rg)

    cl = sub.add_parser("classify", help="Medical image classification: CheXpert")
    add_classification_args(cl)

    # Future experiments:
    # vqa = sub.add_parser("vqa", help="Visual question answering")
    # add_vqa_args(vqa)
    
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Must happen before model loading
    extract_and_set_gpu(args.gpu)
    print_cuda_info()

    # Shared model load for each command
    model, processor, meta = load_model(
        model_id=args.model_id,
        float_type=args.float_type if not args.use_8bit else None,
        use_8bit=args.use_8bit,
    )

    if args.command == "report_gen":
        run_report_generation_experiment(
            args=args,
            model=model,
            processor=processor,
            experiment_meta=meta,
        )
    elif args.command == "classify":
        run_classification_experiment(
            args=args,
            model=model,
            processor=processor,
            experiment_meta=meta,
        )
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()