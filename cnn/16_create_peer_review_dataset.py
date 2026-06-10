#!/usr/bin/env python3
"""Crea un dataset anonimizado para revision por pares a partir de /out."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


HPI_LEVELS = tuple(range(7))
IVR_LEVELS = tuple(range(1, 8))


@dataclass(frozen=True)
class Sample:
    photo_cod: str
    image_path: Path
    hpi: int
    ivr: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Selecciona imagenes de /out, balancea HPI/IVR y anonimiza nombres."
    )
    parser.add_argument("--labels-csv", default="dataset/kelp_photos_filtered.csv")
    parser.add_argument("--images-dir", default="out")
    parser.add_argument("--pattern", default="*_yolo.*")
    parser.add_argument("--output-dir", default="peer_review_dataset")
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--min-hpi-fraction", type=float, default=0.10)
    parser.add_argument("--min-ivr-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Borra el directorio de salida si ya existe.",
    )
    return parser.parse_args()


def photo_code_from_path(image_path: Path) -> str:
    stem = image_path.stem
    return stem[:-5] if stem.endswith("_yolo") else stem


def parse_int_label(raw: str, field_name: str, photo_cod: str) -> int:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"Valor no numerico en {field_name} para {photo_cod}: {raw!r}") from exc
    if not value.is_integer():
        raise ValueError(f"Valor no entero en {field_name} para {photo_cod}: {value}")
    return int(value)


def build_sample_pool(labels_csv: Path, images_dir: Path, pattern: str) -> list[Sample]:
    image_by_code: dict[str, Path] = {}
    for image_path in sorted(images_dir.glob(pattern)):
        if image_path.is_file():
            image_by_code.setdefault(photo_code_from_path(image_path), image_path)

    samples: list[Sample] = []
    with labels_csv.open(newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        for row in reader:
            photo_cod = row["Photo_cod"].strip()
            image_path = image_by_code.get(photo_cod)
            if image_path is None:
                continue
            samples.append(
                Sample(
                    photo_cod=photo_cod,
                    image_path=image_path,
                    hpi=parse_int_label(row["HPI"], "HPI", photo_cod),
                    ivr=parse_int_label(row["IVR"], "IVR", photo_cod),
                )
            )

    if not samples:
        raise ValueError("No se encontraron muestras validas al cruzar CSV e imagenes.")
    return samples


def largest_remainder_distribution(total: int, capacities: dict[int, int], rng: random.Random) -> dict[int, int]:
    if total < 0:
        raise ValueError("total no puede ser negativo")

    keys = list(capacities.keys())
    distribution = {key: 0 for key in keys}
    capacity_sum = sum(max(capacity, 0) for capacity in capacities.values())
    if total == 0 or capacity_sum == 0:
        return distribution

    raw_shares: dict[int, float] = {}
    for key in keys:
        raw_shares[key] = total * max(capacities[key], 0) / capacity_sum

    assigned = 0
    remainders: list[tuple[float, float, int]] = []
    for key in keys:
        floor_share = min(int(math.floor(raw_shares[key])), max(capacities[key], 0))
        distribution[key] = floor_share
        assigned += floor_share
        remainders.append((raw_shares[key] - floor_share, rng.random(), key))

    leftover = total - assigned
    for _, _, key in sorted(remainders, reverse=True):
        if leftover <= 0:
            break
        if distribution[key] >= max(capacities[key], 0):
            continue
        distribution[key] += 1
        leftover -= 1

    if leftover > 0:
        expandable = [key for key in keys if distribution[key] < max(capacities[key], 0)]
        while leftover > 0 and expandable:
            for key in sorted(expandable, key=lambda item: rng.random()):
                if leftover <= 0:
                    break
                if distribution[key] >= max(capacities[key], 0):
                    continue
                distribution[key] += 1
                leftover -= 1
            expandable = [key for key in keys if distribution[key] < max(capacities[key], 0)]

    if sum(distribution.values()) != min(total, capacity_sum):
        raise RuntimeError("No se pudo completar la distribucion solicitada.")
    return distribution


def allocate_hpi_targets(
    samples: list[Sample],
    total_images: int,
    min_hpi_fraction: float,
    rng: random.Random,
) -> dict[int, int]:
    counts = Counter(sample.hpi for sample in samples)
    missing = [level for level in HPI_LEVELS if counts[level] <= 0]
    if missing:
        raise ValueError(f"Faltan muestras para niveles HPI obligatorios: {missing}")

    min_per_hpi = math.ceil(total_images * min_hpi_fraction)
    minimum_required = min_per_hpi * len(HPI_LEVELS)
    if minimum_required > total_images:
        raise ValueError(
            "No es posible cumplir la cuota minima de HPI con el total pedido. "
            f"Necesario={minimum_required}, total={total_images}."
        )

    targets = {level: min_per_hpi for level in HPI_LEVELS}
    for level in HPI_LEVELS:
        if counts[level] < targets[level]:
            raise ValueError(
                f"No hay suficientes muestras para HPI={level}: disponibles={counts[level]}, "
                f"minimo requerido={targets[level]}."
            )

    remaining = total_images - sum(targets.values())
    extra_capacities = {level: counts[level] - targets[level] for level in HPI_LEVELS}
    extra_distribution = largest_remainder_distribution(remaining, extra_capacities, rng)
    for level, extra in extra_distribution.items():
        targets[level] += extra
    return targets


def allocate_ivr_targets_for_hpi(
    samples_for_hpi: list[Sample],
    target_count: int,
    min_ivr_fraction: float,
    rng: random.Random,
) -> tuple[dict[int, int], list[str]]:
    by_ivr: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples_for_hpi:
        by_ivr[sample.ivr].append(sample)

    warnings: list[str] = []
    positive_levels = [level for level in IVR_LEVELS if by_ivr.get(level)]
    if not positive_levels:
        return {}, warnings

    desired_min = math.ceil(target_count * min_ivr_fraction)
    targets = {level: 0 for level in positive_levels}

    for level in positive_levels:
        available = len(by_ivr[level])
        targets[level] = min(desired_min, available)
        if available < desired_min:
            warnings.append(
                f"HPI={samples_for_hpi[0].hpi}, IVR={level}: disponibles={available}, "
                f"minimo objetivo={desired_min}. Se usa best effort."
            )

    assigned = sum(targets.values())
    if assigned > target_count:
        for level in sorted(targets, key=lambda item: (targets[item], rng.random()), reverse=True):
            while assigned > target_count and targets[level] > 0:
                targets[level] -= 1
                assigned -= 1

    remaining = target_count - assigned
    capacities = {
        level: len(by_ivr[level]) - targets[level]
        for level in positive_levels
    }
    extras = largest_remainder_distribution(remaining, capacities, rng)
    for level, extra in extras.items():
        targets[level] += extra

    if sum(targets.values()) != target_count:
        raise RuntimeError("La asignacion IVR no suma el objetivo del estrato HPI.")
    return targets, warnings


def stratified_select(
    pool: list[Sample],
    hpi_targets: dict[int, int],
    min_ivr_fraction: float,
    rng: random.Random,
) -> tuple[list[Sample], list[str]]:
    by_hpi: dict[int, list[Sample]] = defaultdict(list)
    for sample in pool:
        by_hpi[sample.hpi].append(sample)

    selected: list[Sample] = []
    warnings: list[str] = []

    for hpi_level in HPI_LEVELS:
        hpi_samples = list(by_hpi[hpi_level])
        rng.shuffle(hpi_samples)
        target_count = hpi_targets[hpi_level]

        if hpi_level in {0, 5, 6}:
            selected.extend(rng.sample(hpi_samples, target_count))
            continue

        ivr_targets, ivr_warnings = allocate_ivr_targets_for_hpi(
            hpi_samples, target_count, min_ivr_fraction, rng
        )
        warnings.extend(ivr_warnings)

        by_ivr: dict[int, list[Sample]] = defaultdict(list)
        for sample in hpi_samples:
            by_ivr[sample.ivr].append(sample)

        chosen_for_hpi: list[Sample] = []
        for ivr_level, ivr_target in ivr_targets.items():
            bucket = by_ivr[ivr_level]
            if len(bucket) < ivr_target:
                raise RuntimeError(
                    f"Asignacion invalida para HPI={hpi_level}, IVR={ivr_level}: "
                    f"objetivo={ivr_target}, disponibles={len(bucket)}."
                )
            chosen_for_hpi.extend(rng.sample(bucket, ivr_target))

        if len(chosen_for_hpi) != target_count:
            raise RuntimeError(f"Seleccion HPI={hpi_level} incompleta: {len(chosen_for_hpi)} != {target_count}")

        selected.extend(chosen_for_hpi)

    if len(selected) != sum(hpi_targets.values()):
        raise RuntimeError("La seleccion final no coincide con el total objetivo.")
    return selected, warnings


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"El directorio de salida ya existe: {output_dir}. Usa --overwrite para reemplazarlo."
            )
        shutil.rmtree(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(selected: list[Sample], hpi_targets: dict[int, int], warnings: list[str]) -> dict[str, object]:
    hpi_counts = Counter(sample.hpi for sample in selected)
    by_hpi_ivr: dict[str, dict[str, int]] = {}
    for hpi_level in HPI_LEVELS:
        counts = Counter(sample.ivr for sample in selected if sample.hpi == hpi_level)
        by_hpi_ivr[str(hpi_level)] = {str(ivr): counts[ivr] for ivr in sorted(counts)}

    return {
        "num_selected": len(selected),
        "hpi_targets": {str(level): hpi_targets[level] for level in HPI_LEVELS},
        "hpi_selected": {str(level): hpi_counts[level] for level in HPI_LEVELS},
        "ivr_selected_within_hpi": by_hpi_ivr,
        "warnings": warnings,
    }


def export_dataset(selected: list[Sample], output_dir: Path, rng: random.Random) -> dict[str, object]:
    images_dir = output_dir / "images"
    selected_shuffled = list(selected)
    rng.shuffle(selected_shuffled)

    public_rows: list[dict[str, object]] = []
    private_rows: list[dict[str, object]] = []

    for index, sample in enumerate(selected_shuffled, start=1):
        anon_name = f"peer_{index:03d}{sample.image_path.suffix.lower()}"
        target_path = images_dir / anon_name
        shutil.copy2(sample.image_path, target_path)

        public_rows.append(
            {
                "anon_filename": anon_name,
                "hpi_review": "",
                "ivr_review": "",
                "comments": "",
            }
        )
        private_rows.append(
            {
                "anon_filename": anon_name,
                "photo_cod": sample.photo_cod,
                "original_filename": sample.image_path.name,
                "original_path": sample.image_path.as_posix(),
                "hpi": sample.hpi,
                "ivr": sample.ivr,
            }
        )

    write_csv(
        output_dir / "review_template.csv",
        ["anon_filename", "hpi_review", "ivr_review", "comments"],
        public_rows,
    )
    write_csv(
        output_dir / "answer_key_private.csv",
        ["anon_filename", "photo_cod", "original_filename", "original_path", "hpi", "ivr"],
        private_rows,
    )

    return {
        "review_template": (output_dir / "review_template.csv").as_posix(),
        "answer_key_private": (output_dir / "answer_key_private.csv").as_posix(),
        "images_dir": images_dir.as_posix(),
    }


def main() -> None:
    args = parse_args()
    if args.num_images <= 0:
        raise ValueError("--num-images debe ser > 0")
    if not (0.0 < args.min_hpi_fraction <= 1.0):
        raise ValueError("--min-hpi-fraction debe estar en (0, 1]")
    if not (0.0 < args.min_ivr_fraction <= 1.0):
        raise ValueError("--min-ivr-fraction debe estar en (0, 1]")

    labels_csv = Path(args.labels_csv)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    rng = random.Random(args.seed)
    pool = build_sample_pool(labels_csv, images_dir, args.pattern)
    if len(pool) < args.num_images:
        raise ValueError(
            f"No hay suficientes muestras en el pool: disponibles={len(pool)}, pedidas={args.num_images}."
        )

    hpi_targets = allocate_hpi_targets(pool, args.num_images, args.min_hpi_fraction, rng)
    selected, warnings = stratified_select(pool, hpi_targets, args.min_ivr_fraction, rng)

    prepare_output_dir(output_dir, args.overwrite)
    exported_paths = export_dataset(selected, output_dir, rng)

    summary = build_summary(selected, hpi_targets, warnings)
    summary.update(
        {
            "config": {
                "labels_csv": labels_csv.as_posix(),
                "images_dir": images_dir.as_posix(),
                "output_dir": output_dir.as_posix(),
                "num_images": args.num_images,
                "min_hpi_fraction": args.min_hpi_fraction,
                "min_ivr_fraction": args.min_ivr_fraction,
                "seed": args.seed,
            },
            "exports": exported_paths,
        }
    )
    summary_path = output_dir / "selection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Pool disponible: {len(pool)} imagenes con etiqueta y archivo en {images_dir}")
    print(f"Dataset creado en: {output_dir}")
    print(f"Imagenes copiadas: {len(selected)}")
    print(f"Plantilla publica: {exported_paths['review_template']}")
    print(f"Respuesta privada: {exported_paths['answer_key_private']}")
    print(f"Resumen: {summary_path.as_posix()}")
    if warnings:
        print("Avisos de representatividad:")
        for warning in warnings:
            print(f" - {warning}")


if __name__ == "__main__":
    main()
