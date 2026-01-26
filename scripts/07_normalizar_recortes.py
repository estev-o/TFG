#!/usr/bin/env python3
"""
Normaliza recortes del directorio /out:
- Filtra solo imágenes con código presente en el CSV filtrado.
- Calcula el tamaño máximo y normaliza todas al mismo tamaño sin perder proporción.
- Guarda resultados en /out_img_norm.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    from pi_heif import register_heif_opener

    register_heif_opener()
except Exception:
    # Si no hay soporte HEIC, seguimos con los formatos estándar.
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "out"
DEFAULT_CSV = REPO_ROOT / "dataset" / "kelp_photos_filtered.csv"
DEFAULT_OUTPUT = REPO_ROOT / "out_img_norm"
IMAGE_PATTERNS = [
    "*.jfif", "*.JFIF", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG",
    "*.png", "*.PNG", "*.heic", "*.HEIC", "*.tif", "*.TIF", "*.tiff", "*.TIFF",
]
SUFIJOS_CONOCIDOS = ("_yolo", "_alga")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normaliza recortes usando el tamaño máximo y filtra por CSV."
    )
    parser.add_argument("--input_dir", default=str(DEFAULT_INPUT), help="Directorio con recortes (default: out)")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="CSV filtrado con Photo_cod")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT), help="Directorio de salida normalizada")
    parser.add_argument("--bg", type=int, default=127, help="Color de fondo para padding (0-255)")
    return parser.parse_args()


def normalizar_codigo(valor):
    if valor is None:
        return None
    texto = str(valor).strip()
    if not texto:
        return None
    return Path(texto).stem.lower()


def encontrar_columna(fieldnames):
    if not fieldnames:
        return None
    def norm(col):
        return col.strip().lower().replace("_", "").replace(" ", "")

    for col in fieldnames:
        if norm(col) == "photocod":
            return col
    for col in fieldnames:
        n = norm(col)
        if "photo" in n and ("cod" in n or "code" in n):
            return col
    return None


def cargar_codigos(csv_path: Path):
    if not csv_path.exists():
        print(f"ERROR: No existe el CSV: {csv_path}")
        sys.exit(1)

    codigos = set()
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        col = encontrar_columna(reader.fieldnames)
        if col is None:
            print("ERROR: No se encontró columna Photo_cod en el CSV.")
            print(f"Columnas: {reader.fieldnames}")
            sys.exit(1)
        for row in reader:
            code = normalizar_codigo(row.get(col))
            if code:
                codigos.add(code)
    return codigos


def extraer_codigo_imagen(path: Path):
    stem = path.stem
    lower = stem.lower()
    for suf in SUFIJOS_CONOCIDOS:
        if lower.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return stem.strip().lower()


def recolectar_imagenes(input_dir: Path):
    imagenes = []
    for pattern in IMAGE_PATTERNS:
        imagenes.extend(sorted(input_dir.glob(pattern)))
    return imagenes


def leer_tamaño(path: Path):
    """Lee tamaño sin decodificar la imagen completa (menos memoria)."""
    try:
        with Image.open(path) as img:
            return img.width, img.height
    except Exception:
        return None


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    csv_path = Path(args.csv)

    if not input_dir.exists():
        print(f"ERROR: No existe el directorio de entrada: {input_dir}")
        sys.exit(1)

    codigos_validos = cargar_codigos(csv_path)
    imagenes = recolectar_imagenes(input_dir)
    if not imagenes:
        print(f"ERROR: No se encontraron imágenes en {input_dir}")
        sys.exit(1)

    sin_match = 0
    ilegibles = 0
    max_side = 0
    seleccionadas = 0

    # Paso 1 + 2: filtrar por CSV y encontrar el tamaño máximo sin cargar todo en memoria
    for img_path in imagenes:
        code = extraer_codigo_imagen(img_path)
        if code not in codigos_validos:
            sin_match += 1
            continue
        size = leer_tamaño(img_path)
        if size is None:
            ilegibles += 1
            continue
        w, h = size
        max_side = max(max_side, h, w)
        seleccionadas += 1

    if seleccionadas == 0 or max_side == 0:
        print("ERROR: No hay imágenes seleccionadas tras filtrar por CSV.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    guardadas = 0

    # Paso 3: normalizar con el tamaño máximo
    for img_path in imagenes:
        code = extraer_codigo_imagen(img_path)
        if code not in codigos_validos:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            ilegibles += 1
            continue
        h, w = img.shape[:2]
        side = max(h, w)
        canvas = np.full((side, side, 3), args.bg, dtype=np.uint8)
        offset_y = (side - h) // 2
        offset_x = (side - w) // 2
        canvas[offset_y:offset_y + h, offset_x:offset_x + w] = img
        if side != max_side:
            canvas = cv2.resize(canvas, (max_side, max_side), interpolation=cv2.INTER_LANCZOS4)

        out_path = output_dir / img_path.name
        if cv2.imwrite(str(out_path), canvas):
            guardadas += 1

    print("\nResumen:")
    print(f"  Imágenes encontradas: {len(imagenes)}")
    print(f"  Con match CSV: {seleccionadas}")
    print(f"  Sin match CSV: {sin_match}")
    print(f"  Ilegibles: {ilegibles}")
    print(f"  Tamaño objetivo: {max_side}x{max_side}")
    print(f"  Guardadas: {guardadas}")
    print(f"  Salida: {output_dir}")


if __name__ == "__main__":
    main()
