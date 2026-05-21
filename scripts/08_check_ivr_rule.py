#!/usr/bin/env python3
"""Valida la regla HPI in {0,5,6} => IVR = 0 en un CSV de etiquetas."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comprueba la regla IVR=0 para HPI 0,5,6")
    parser.add_argument("csv_path", help="Ruta al CSV a revisar")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Corrige las filas incumplidas poniendo IVR=0 y sobrescribe el archivo",
    )
    return parser.parse_args()


def detect_column(fieldnames: list[str], *candidates: str) -> str:
    for name in candidates:
        if name in fieldnames:
            return name
    raise ValueError(f"No se encontró ninguna de las columnas esperadas: {candidates}")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"CSV sin cabecera: {csv_path}")
        rows = list(reader)

    photo_key = detect_column(fieldnames, "Photo_cod", "photo_cod")
    hpi_key = detect_column(fieldnames, "HPI", "hpi")
    ivr_key = detect_column(fieldnames, "IVR", "ivr")

    invalid_rows: list[dict[str, str]] = []
    for row in rows:
        hpi = int(float(row[hpi_key]))
        ivr = int(float(row[ivr_key]))
        if hpi in {0, 5, 6} and ivr != 0:
            invalid_rows.append(row)
            if args.fix:
                row[ivr_key] = "0"

    print(f"Archivo: {csv_path}")
    print(f"Filas totales: {len(rows)}")
    print(f"Incumplimientos: {len(invalid_rows)}")

    for row in invalid_rows[:20]:
        print(f"- {row[photo_key]}: {hpi_key}={row[hpi_key]}, {ivr_key}={row[ivr_key]}")

    if args.fix:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print("Archivo corregido sobrescribiendo IVR=0 en las filas incumplidas.")


if __name__ == "__main__":
    main()
