"""Validate local reconstruction datasets."""

import argparse
import sys

from imaged.data.validate import DatasetValidationError, validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="imaged-dataset", description="Validate local imaged datasets"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser(
        "validate", help="Validate a local dataset directory"
    )
    validate_parser.add_argument("dataset_dir")
    args = parser.parse_args()

    try:
        dataset = validate_dataset(args.dataset_dir)
    except DatasetValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except KeyboardInterrupt as exc:
        print("cancelled", file=sys.stderr)
        raise SystemExit(130) from exc

    print(
        f"Valid dataset: {dataset.root} "
        f"({len(dataset.capture_paths)} captures, {dataset.total_bytes:,} bytes)"
    )


if __name__ == "__main__":
    main()
