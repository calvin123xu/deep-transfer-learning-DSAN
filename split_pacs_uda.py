#!/usr/bin/env python3
"""Split PACS-style source/target domains into UDA train/val/test folders.

Expected input structure:
    root/domain/class_name/image.xxx

This script creates:
    root/<src>_train
    root/<src>_val
    root/<tar>_train
    root/<tar>_test
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable, Sequence

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class SplitError(RuntimeError):
    """Raised when dataset split preconditions are not met."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split PACS-style source/target domains for UDA experiments."
    )
    parser.add_argument(
        "--root_path",
        type=Path,
        required=True,
        help="Dataset root containing PACS domain folders.",
    )
    parser.add_argument("--src", type=str, required=True, help="Source domain name.")
    parser.add_argument("--tar", type=str, required=True, help="Target domain name.")
    parser.add_argument(
        "--src_train_ratio",
        type=float,
        default=0.8,
        help="Train ratio for the source split (default: 0.8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2021,
        help="Random seed for deterministic per-class shuffling (default: 2021).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing output split folders before creating new ones.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.root_path.exists() or not args.root_path.is_dir():
        raise SplitError(f"root_path does not exist or is not a directory: {args.root_path}")

    if args.src == args.tar:
        raise SplitError("--src and --tar must be different domains.")

    if not (0.0 < args.src_train_ratio < 1.0):
        raise SplitError("--src_train_ratio must be between 0 and 1 (exclusive).")


def get_class_dirs(domain_dir: Path) -> list[Path]:
    class_dirs = sorted([p for p in domain_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise SplitError(f"No class folders found under domain: {domain_dir}")
    return class_dirs


def list_images(class_dir: Path) -> list[Path]:
    images = sorted(
        [
            p
            for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )
    return images


def ensure_clean_output_dirs(output_dirs: Sequence[Path], overwrite: bool) -> None:
    existing = [d for d in output_dirs if d.exists()]
    if existing and not overwrite:
        existing_str = "\n  - ".join(str(p) for p in existing)
        raise SplitError(
            "Output split folders already exist. Use --overwrite to remove and recreate them."
            f"\n  - {existing_str}"
        )

    if overwrite:
        for d in existing:
            shutil.rmtree(d)

    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)


def split_indices(total: int, train_ratio: float) -> tuple[int, int]:
    train_count = int(total * train_ratio)
    train_count = max(0, min(train_count, total))
    second_count = total - train_count
    return train_count, second_count


def copy_split(
    class_name: str,
    train_files: Iterable[Path],
    second_files: Iterable[Path],
    train_root: Path,
    second_root: Path,
) -> None:
    train_class_dir = train_root / class_name
    second_class_dir = second_root / class_name
    train_class_dir.mkdir(parents=True, exist_ok=True)
    second_class_dir.mkdir(parents=True, exist_ok=True)

    for src_file in train_files:
        shutil.copy2(src_file, train_class_dir / src_file.name)

    for src_file in second_files:
        shutil.copy2(src_file, second_class_dir / src_file.name)


def split_domain(
    domain_dir: Path,
    train_root: Path,
    second_root: Path,
    train_ratio: float,
    seed: int,
    second_label: str,
) -> None:
    class_dirs = get_class_dirs(domain_dir)

    for class_dir in class_dirs:
        class_name = class_dir.name
        images = list_images(class_dir)

        if not images:
            print(f"[WARN] {domain_dir.name}/{class_name}: no supported image files, skipped")
            continue

        rng = random.Random(seed)
        shuffled = images[:]
        rng.shuffle(shuffled)

        train_count, second_count = split_indices(len(shuffled), train_ratio)
        train_files = shuffled[:train_count]
        second_files = shuffled[train_count:]

        copy_split(class_name, train_files, second_files, train_root, second_root)

        print(
            f"{domain_dir.name}/{class_name}: total={len(shuffled)}, "
            f"train_count={train_count}, {second_label}_count={second_count}"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    root = args.root_path
    src_dir = root / args.src
    tar_dir = root / args.tar

    if not src_dir.exists() or not src_dir.is_dir():
        raise SplitError(f"Source domain folder does not exist: {src_dir}")
    if not tar_dir.exists() or not tar_dir.is_dir():
        raise SplitError(f"Target domain folder does not exist: {tar_dir}")

    src_train_dir = root / f"{args.src}_train"
    src_val_dir = root / f"{args.src}_val"
    tar_train_dir = root / f"{args.tar}_train"
    tar_test_dir = root / f"{args.tar}_test"

    ensure_clean_output_dirs(
        [src_train_dir, src_val_dir, tar_train_dir, tar_test_dir],
        overwrite=args.overwrite,
    )

    print("[Source split]")
    split_domain(
        domain_dir=src_dir,
        train_root=src_train_dir,
        second_root=src_val_dir,
        train_ratio=args.src_train_ratio,
        seed=args.seed,
        second_label="val",
    )

    print("\n[Target split]")
    split_domain(
        domain_dir=tar_dir,
        train_root=tar_train_dir,
        second_root=tar_test_dir,
        train_ratio=0.5,
        seed=args.seed,
        second_label="test",
    )

    print("\nDone. Created split folders:")
    print(f"  - {src_train_dir}")
    print(f"  - {src_val_dir}")
    print(f"  - {tar_train_dir}")
    print(f"  - {tar_test_dir}")


if __name__ == "__main__":
    try:
        main()
    except SplitError as err:
        raise SystemExit(f"Error: {err}")
