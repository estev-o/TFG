#!/usr/bin/env python3
"""
Normaliza recortes del directorio /out:
- Filtra solo imágenes con código presente en el CSV filtrado.
- Segmenta alga con Otsu, conserva solo el componente más grande y mantiene huecos internos.
- Normaliza todas a tamaño cuadrado común con padding negro (binario 0/255).
- Guarda máscaras finales en /out_img_norm y paneles de depuración en /out_img_norm_debug.
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
DEFAULT_DEBUG = REPO_ROOT / "out_img_norm_debug"
IMAGE_PATTERNS = [
    "*.jfif", "*.JFIF", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG",
    "*.png", "*.PNG", "*.heic", "*.HEIC", "*.tif", "*.TIF", "*.tiff", "*.TIFF",
]
SUFIJOS_CONOCIDOS = ("_yolo", "_alga")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normaliza recortes binarios (alga=255, fondo=0) y filtra por CSV."
    )
    parser.add_argument("--input_dir", default=str(DEFAULT_INPUT), help="Directorio con recortes (default: out)")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="CSV filtrado con Photo_cod")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT), help="Directorio de salida (máscaras binarias)")
    parser.add_argument("--debug_dir", default=str(DEFAULT_DEBUG), help="Directorio de paneles de depuración")
    parser.add_argument(
        "--open_kernel",
        type=int,
        default=3,
        help="Tamaño de kernel para apertura morfológica (0 o 1 para desactivar)",
    )
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


def leer_imagen_color(path: Path):
    """Lee imagen en BGR con fallback a PIL para formatos no soportados por OpenCV."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        with Image.open(path) as pil_img:
            rgb = np.array(pil_img.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def mayor_componente(mask_binaria: np.ndarray):
    """
    Devuelve la máscara del componente conexo más grande (255 foreground),
    su área y su bounding box. Mantiene huecos internos de forma natural.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binaria, connectivity=8)
    if num_labels <= 1:
        return None, 0, None

    h, w = mask_binaria.shape[:2]
    mejor_label = -1
    mejor_area = 0
    mejor_bbox = None

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])

        # Ignora componentes que abarcan toda la imagen (típico fondo de polaridad incorrecta).
        cubre_toda_imagen = (x == 0 and y == 0 and (x + bw) == w and (y + bh) == h)
        if cubre_toda_imagen:
            continue

        if area > mejor_area:
            mejor_area = area
            mejor_label = label
            mejor_bbox = (x, y, bw, bh)

    if mejor_label == -1:
        # Fallback: si todos cubren toda la imagen, usa igualmente el mayor.
        mejor_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mejor_area = int(stats[mejor_label, cv2.CC_STAT_AREA])
        x = int(stats[mejor_label, cv2.CC_STAT_LEFT])
        y = int(stats[mejor_label, cv2.CC_STAT_TOP])
        bw = int(stats[mejor_label, cv2.CC_STAT_WIDTH])
        bh = int(stats[mejor_label, cv2.CC_STAT_HEIGHT])
        mejor_bbox = (x, y, bw, bh)

    componente = np.zeros_like(mask_binaria, dtype=np.uint8)
    componente[labels == mejor_label] = 255
    return componente, mejor_area, mejor_bbox


def construir_mascara_alga(img_bgr: np.ndarray, open_kernel: int):
    """
    Segmenta con Otsu en ambas polaridades y elige la opción con componente mayor.
    Retorna: máscara final, máscara otsu normal, máscara otsu invertida, polaridad elegida, área.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_otsu_inv = cv2.bitwise_not(mask_otsu)

    if open_kernel and open_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        mask_otsu = cv2.morphologyEx(mask_otsu, cv2.MORPH_OPEN, kernel)
        mask_otsu_inv = cv2.morphologyEx(mask_otsu_inv, cv2.MORPH_OPEN, kernel)

    comp_bin, area_bin, _ = mayor_componente(mask_otsu)
    comp_inv, area_inv, _ = mayor_componente(mask_otsu_inv)

    if area_bin == 0 and area_inv == 0:
        return None, mask_otsu, mask_otsu_inv, "none", 0
    if area_inv > area_bin:
        return comp_inv, mask_otsu, mask_otsu_inv, "inv", area_inv
    return comp_bin, mask_otsu, mask_otsu_inv, "bin", area_bin


def guardar_panel_debug(
    debug_dir: Path,
    img_path: Path,
    img_bgr: np.ndarray,
    mask_otsu: np.ndarray,
    mask_otsu_inv: np.ndarray,
    mask_final: np.ndarray,
    polaridad: str,
    area: int,
):
    h, w = img_bgr.shape[:2]
    canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

    otsu_bgr = cv2.cvtColor(mask_otsu, cv2.COLOR_GRAY2BGR)
    otsu_inv_bgr = cv2.cvtColor(mask_otsu_inv, cv2.COLOR_GRAY2BGR)
    final_bgr = cv2.cvtColor(mask_final, cv2.COLOR_GRAY2BGR)

    canvas[0:h, 0:w] = img_bgr
    canvas[0:h, w:2 * w] = otsu_bgr
    canvas[h:2 * h, 0:w] = otsu_inv_bgr
    canvas[h:2 * h, w:2 * w] = final_bgr

    cv2.putText(canvas, "Original", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Otsu bin", (w + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Otsu inv", (10, h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"Final ({polaridad}, area={area})",
        (w + 10, h + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    out_debug = debug_dir / f"{img_path.stem}_debug.png"
    return cv2.imwrite(str(out_debug), canvas)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    debug_dir = Path(args.debug_dir)
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
    debug_dir.mkdir(parents=True, exist_ok=True)
    guardadas = 0
    sin_mascara = 0
    debug_guardadas = 0

    # Paso 3: segmentar alga y normalizar máscara binaria con el tamaño máximo
    for img_path in imagenes:
        code = extraer_codigo_imagen(img_path)
        if code not in codigos_validos:
            continue
        img = leer_imagen_color(img_path)
        if img is None:
            ilegibles += 1
            continue
        mask, mask_otsu, mask_otsu_inv, polaridad, area = construir_mascara_alga(img, args.open_kernel)
        if mask is None:
            sin_mascara += 1
            continue

        h, w = mask.shape[:2]
        side = max(h, w)
        canvas = np.zeros((side, side), dtype=np.uint8)
        offset_y = (side - h) // 2
        offset_x = (side - w) // 2
        canvas[offset_y:offset_y + h, offset_x:offset_x + w] = mask
        if side != max_side:
            canvas = cv2.resize(canvas, (max_side, max_side), interpolation=cv2.INTER_NEAREST)
        canvas = np.where(canvas > 0, 255, 0).astype(np.uint8)

        out_path = output_dir / f"{img_path.stem}.png"
        if cv2.imwrite(str(out_path), canvas):
            guardadas += 1
            if guardar_panel_debug(
                debug_dir=debug_dir,
                img_path=img_path,
                img_bgr=img,
                mask_otsu=mask_otsu,
                mask_otsu_inv=mask_otsu_inv,
                mask_final=mask,
                polaridad=polaridad,
                area=area,
            ):
                debug_guardadas += 1

    print("\nResumen:")
    print(f"  Imágenes encontradas: {len(imagenes)}")
    print(f"  Con match CSV: {seleccionadas}")
    print(f"  Sin match CSV: {sin_match}")
    print(f"  Ilegibles: {ilegibles}")
    print(f"  Sin máscara válida: {sin_mascara}")
    print(f"  Tamaño objetivo: {max_side}x{max_side}")
    print(f"  Guardadas: {guardadas}")
    print(f"  Debug guardadas: {debug_guardadas}")
    print(f"  Salida: {output_dir}")
    print(f"  Debug: {debug_dir}")


if __name__ == "__main__":
    main()
