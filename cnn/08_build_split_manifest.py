#!/usr/bin/env python3
"""Construye manifest + split train/val/test para la CNN en una sola ejecución."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera manifest y split para CNN")
    parser.add_argument("--labels-csv", default="dataset/kelp_photos_filtered.csv")
    parser.add_argument("--images-dir", default="out_img_norm")
    parser.add_argument("--pattern", default="*_yolo.png")
    parser.add_argument("--manifest-output", default="cnn/manifest.csv")
    parser.add_argument("--split-output-dir", default="cnn/splits")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def photo_code(image_path: Path) -> str:
    name = image_path.stem
    return name[:-5] if name.endswith("_yolo") else name


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_int_score(value: str, field_name: str, photo_code: str) -> int:
    try:
        numeric = float(value)
    except ValueError as exc:
        raise ValueError(f"Valor no numerico en {field_name} para {photo_code}: {value!r}") from exc

    if not numeric.is_integer():
        raise ValueError(f"Valor no entero en {field_name} para {photo_code}: {numeric}")
    return int(numeric)


def build_manifest_rows(labels_csv: Path, images_dir: Path, pattern: str) -> list[dict[str, object]]:
    image_by_code: dict[str, Path] = {}
    for image_path in sorted(images_dir.glob(pattern)):
        image_by_code.setdefault(photo_code(image_path), image_path)

    rows: list[dict[str, object]] = []
    with labels_csv.open(newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        for row in reader:
            code = row["Photo_cod"].strip()
            image_path = image_by_code.get(code)
            if image_path is None:
                continue
            rows.append(
                {
                    "photo_cod": code,
                    "image_path": image_path.as_posix(),
                    "hpi": parse_int_score(row["HPI"], "HPI", code),
                    "ivr": parse_int_score(row["IVR"], "IVR", code),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError("train-ratio + val-ratio + test-ratio debe sumar 1.0")

    labels_csv = Path(args.labels_csv)
    images_dir = Path(args.images_dir)
    manifest_output = Path(args.manifest_output)
    split_output_dir = Path(args.split_output_dir)

    rows = build_manifest_rows(labels_csv, images_dir, args.pattern)
    write_rows(manifest_output, ["photo_cod", "image_path", "hpi", "ivr"], rows)

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    n = len(rows)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    train_rows = rows[:n_train]
    val_rows = rows[n_train : n_train + n_val]
    test_rows = rows[n_train + n_val :]

    write_rows(split_output_dir / "train.csv", ["photo_cod", "image_path", "hpi", "ivr"], train_rows)
    write_rows(split_output_dir / "val.csv", ["photo_cod", "image_path", "hpi", "ivr"], val_rows)
    write_rows(split_output_dir / "test.csv", ["photo_cod", "image_path", "hpi", "ivr"], test_rows)

    print(f"Manifest escrito en {manifest_output} ({n} filas)")
    print(f"Split escrito en {split_output_dir}: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")


if __name__ == "__main__":
    main()
