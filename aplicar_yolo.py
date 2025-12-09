#!/usr/bin/env python3
"""
Aplica un modelo YOLO entrenado para recortar algas de imágenes aleatorias del dataset.
- Selecciona N imágenes al azar del dataset.
- Predice bbox con YOLO y recorta el bbox de mayor confianza.
- Guarda los recortes en /out con sufijo _yolo.jpg.
"""

import argparse
import random
from pathlib import Path
from typing import Optional

import cv2
import torch
from ultralytics import YOLO

import recortar_algas as detector

BASE_DIR = Path(__file__).parent
DEFAULT_DATASET = BASE_DIR / 'dataset' / 'Kelps_database_photos' / 'Photos_kelps_database'
DEFAULT_MODEL = BASE_DIR / 'runs_yolo' / 'kelp' / 'weights' / 'best.pt'
DEFAULT_OUTPUT = BASE_DIR / 'out'
YOLO_SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.pfm', '.webp', '.dng', '.heic', '.mpo'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Recorta algas con un modelo YOLO sobre imágenes aleatorias del dataset.'
    )
    parser.add_argument('--model', type=str, default=str(DEFAULT_MODEL), help='Ruta al modelo YOLO (.pt)')
    parser.add_argument('--dataset', type=str, default=str(DEFAULT_DATASET), help='Directorio de imágenes')
    parser.add_argument('--output_dir', type=str, default=str(DEFAULT_OUTPUT), help='Directorio de salida')
    parser.add_argument('--num_images', type=int, default=10, help='Número de imágenes aleatorias a procesar')
    parser.add_argument('--seed', type=int, default=42, help='Semilla para la selección aleatoria')
    parser.add_argument('--imgsz', type=int, default=640, help='Tamaño de entrada para YOLO')
    parser.add_argument(
        '--device', default=None, help="Dispositivo (ej. 0, 0,1 o cpu). Si no se indica, autodetecta."
    )
    parser.add_argument(
        '--conf', type=float, default=0.25, help="Confianza mínima YOLO (baja a 0.1 si devuelve muchas 'sin detección')."
    )
    return parser.parse_args()


def seleccionar_imagenes(dataset_dir: Path, num_images: int, seed: int):
    imagenes = detector.recolectar_imagenes(dataset_dir)
    imagenes = [p for p in imagenes if p.suffix.lower() in YOLO_SUPPORTED_EXTS]
    if not imagenes:
        raise SystemExit(f"No se encontraron imágenes en {dataset_dir}")
    random.seed(seed)
    if num_images and num_images < len(imagenes):
        imagenes = random.sample(imagenes, num_images)
    return imagenes


def mejor_caja(pred) -> Optional[tuple[int, int, int, int]]:
    """Devuelve bbox xyxy entero de la predicción con mayor confianza."""
    boxes = pred.boxes
    if boxes is None or len(boxes) == 0:
        return None
    best_idx = boxes.conf.argmax().item()
    xyxy = boxes.xyxy[best_idx].int().tolist()
    x1, y1, x2, y2 = xyxy
    return x1, y1, x2, y2


def recortar_y_guardar(img_path: Path, bbox_xyxy, output_dir: Path):
    imagen = cv2.imread(str(img_path))
    if imagen is None:
        return False
    x1, y1, x2, y2 = bbox_xyxy
    h, w = imagen.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return False
    recorte = imagen[y1:y2, x1:x2]
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / f"{img_path.stem}_yolo.jpg"
    cv2.imwrite(str(out_path), recorte)
    return True


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output_dir)

    if not dataset_dir.exists():
        raise SystemExit(f"El dataset no existe: {dataset_dir}")
    if not Path(args.model).exists():
        raise SystemExit(f"No se encontró el modelo: {args.model}")

    imagenes = seleccionar_imagenes(dataset_dir, args.num_images, args.seed)
    model = YOLO(args.model)

    print(f"Procesando {len(imagenes)} imágenes aleatorias desde {dataset_dir}")
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    ok = 0
    fallidas = 0
    sin_deteccion = 0

    missing = 0
    for idx, img_path in enumerate(imagenes, 1):
        print(f"[{idx}/{len(imagenes)}] {img_path.name}... ", end='', flush=True)
        if not img_path.exists():
            missing += 1
            print("archivo no encontrado, se omite")
            continue
        try:
            preds = model.predict(
                source=str(img_path),
                imgsz=args.imgsz,
                device=device,
                conf=args.conf,
                verbose=False,
            )
        except Exception as e:
            fallidas += 1
            print(f"error de predicción: {e}")
            continue
        if not preds:
            sin_deteccion += 1
            print("sin predicción")
            continue
        bbox = mejor_caja(preds[0])
        if bbox is None:
            sin_deteccion += 1
            print("sin detección")
            continue
        if recortar_y_guardar(img_path, bbox, output_dir):
            ok += 1
            print("OK")
        else:
            fallidas += 1
            print("error al recortar")

    print("\nResumen:")
    print(f"  Recortes guardados: {ok}")
    if missing:
        print(f"  Archivos faltantes: {missing}")
    print(f"  Sin detección: {sin_deteccion}")
    print(f"  Errores: {fallidas}")
    print(f"  Salida: {output_dir}")


if __name__ == '__main__':
    main()
