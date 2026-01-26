#!/usr/bin/env python3
"""
Genera un dataset en formato YOLO usando el detector heurístico de `recortar_algas`.
- Copia las imágenes a yolo/images/{train,val}
- Crea etiquetas YOLO (x_center y_center width height normalizados) en yolo/labels/{train,val}
- Escribe un data.yaml listo para entrenar con una única clase: alga
"""

import argparse
import random
import shutil
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / 'dataset' / 'Kelps_database_photos' / 'Photos_kelps_database'
DEFAULT_OUTPUT = REPO_ROOT / 'yolo'
ALGAS_PATH = REPO_ROOT / 'scripts' / '02_recortar_algas.py'
CLASS_ID = 0
CLASS_NAME = 'alga'
DEFAULT_SCREEN_FALLBACK = (1920, 1080)


def cargar_detector():
    if not ALGAS_PATH.exists():
        print(f"ERROR: No se encontró el script de detección: {ALGAS_PATH}")
        sys.exit(1)
    spec = spec_from_file_location("recortar_algas", ALGAS_PATH)
    if spec is None or spec.loader is None:
        print("ERROR: No se pudo cargar recortar_algas.")
        sys.exit(1)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


detector = cargar_detector()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Genera dataset YOLO con bounding boxes de algas usando recortar_algas.'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default=str(DEFAULT_INPUT),
        help='Directorio con imágenes de entrada'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(DEFAULT_OUTPUT),
        help='Directorio base para el dataset YOLO (se crearán images/ y labels/)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Limitar el número de imágenes a procesar'
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.2,
        help='Porcentaje de imágenes para validación (0-1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semilla para el split aleatorio train/val'
    )
    parser.add_argument(
        '--review',
        action='store_true',
        help='Modo interactivo: muestra la imagen con bbox y solo guarda si confirmas (Enter=guardar, otra tecla=descartar, q/Esc=salir).'
    )
    parser.add_argument(
        '--review_max_size',
        type=int,
        default=900,
        help='Tamaño máximo (px) para ancho/alto de la ventana en modo review.'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Eliminar la carpeta de salida si ya existe'
    )
    return parser.parse_args()


def preparar_directorios(base_dir: Path, overwrite: bool) -> Dict[str, Dict[str, Path]]:
    """Crea la estructura yolo/{images,labels}/{train,val}."""
    if base_dir.exists():
        if any(base_dir.iterdir()):
            if not overwrite:
                print(f"ERROR: {base_dir} ya existe y no está vacío. Usa --overwrite para regenerarlo.")
                sys.exit(1)
            shutil.rmtree(base_dir)

    images_train = base_dir / 'images' / 'train'
    images_val = base_dir / 'images' / 'val'
    labels_train = base_dir / 'labels' / 'train'
    labels_val = base_dir / 'labels' / 'val'

    for path in (images_train, images_val, labels_train, labels_val):
        path.mkdir(parents=True, exist_ok=True)

    return {
        'images': {'train': images_train, 'val': images_val},
        'labels': {'train': labels_train, 'val': labels_val},
    }


def bbox_a_formato_yolo(
    bbox: Tuple[int, int, int, int], img_shape
) -> Optional[Tuple[float, float, float, float]]:
    """Convierte (x, y, w, h) absoluto a formato YOLO normalizado."""
    x, y, w, h = bbox
    img_h, img_w = img_shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)

    ancho = max(0, x2 - x1)
    alto = max(0, y2 - y1)

    if ancho <= 1 or alto <= 1 or img_w == 0 or img_h == 0:
        return None

    x_center = (x1 + ancho / 2) / img_w
    y_center = (y1 + alto / 2) / img_h

    return x_center, y_center, ancho / img_w, alto / img_h


def escribir_label(label_path: Path, bbox_norm: Tuple[float, float, float, float]):
    """Escribe un único bbox en formato YOLO para la clase alga."""
    x_c, y_c, w_n, h_n = bbox_norm
    label_path.write_text(f"{CLASS_ID} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")


def escribir_data_yaml(base_dir: Path) -> Path:
    """Genera el fichero data.yaml que usan los entrenadores YOLO."""
    yaml_path = base_dir / 'data.yaml'
    contenido = "\n".join(
        [
            "# Dataset auto-generado con scripts/04_recortar_yolo.py",
            f"path: {base_dir.resolve()}",
            "train: images/train",
            "val: images/val",
            "names:",
            f"  {CLASS_ID}: {CLASS_NAME}",
            "",
        ]
    )
    yaml_path.write_text(contenido)
    return yaml_path


def obtener_tamaño_pantalla():
    """Devuelve (ancho, alto) de la pantalla principal o None si no se pudo."""
    try:
        import tkinter as tk  # stdlib

        root = tk.Tk()
        root.withdraw()
        ancho = root.winfo_screenwidth()
        alto = root.winfo_screenheight()
        root.destroy()
        return ancho, alto
    except Exception:
        return None


def revisar_manual(
    imagen, bbox: Tuple[int, int, int, int], nombre: str, max_dim: int
) -> Optional[bool]:
    """Muestra la imagen con el bbox y devuelve True=guardar, False=descartar, None=salir."""
    vista = imagen.copy()
    x, y, w, h = bbox
    cv2.rectangle(vista, (x, y), (x + w, y + h), (0, 255, 0), 4)
    cv2.putText(vista, "Enter=guardar | cualquier otra=descartar | q/Esc=salir",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    # Reducir ventana si es demasiado grande para que quepa en pantalla
    escala = min(1.0, max_dim / max(vista.shape[:2]))
    if escala < 1.0:
        nueva_w = int(vista.shape[1] * escala)
        nueva_h = int(vista.shape[0] * escala)
        vista = cv2.resize(vista, (nueva_w, nueva_h), interpolation=cv2.INTER_AREA)
    win_name = f"Revision {nombre}"
    cv2.imshow(win_name, vista)

    # Intentar centrar la ventana considerando el tamaño final
    screen_size = obtener_tamaño_pantalla() or DEFAULT_SCREEN_FALLBACK
    if screen_size:
        win_h, win_w = vista.shape[:2]
        scr_w, scr_h = screen_size
        x = max(0, (scr_w - win_w) // 2)
        y = max(0, (scr_h - win_h) // 2)
        cv2.moveWindow(win_name, x, y)

    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow(win_name)

    if key in (13, 10, ord('y'), ord('s'), ord(' ')):  # Enter, espacio o yes/sí
        return True
    if key in (ord('q'), 27):  # q o Esc para abortar
        return None
    return False


def main():
    args = parse_args()

    if not 0 <= args.val_split < 1:
        print("ERROR: --val_split debe estar entre 0 y 1 (ej. 0.2 = 20% val)")
        sys.exit(1)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"ERROR: No existe el directorio de entrada: {input_dir}")
        sys.exit(1)

    imagenes = detector.recolectar_imagenes(input_dir, args.num_samples)
    if not imagenes:
        print("ERROR: No se encontraron imágenes en el directorio indicado.")
        sys.exit(1)

    rutas = preparar_directorios(output_dir, args.overwrite)
    random.seed(args.seed)

    exitosas = 0
    descartadas_manual = 0
    abortado_por_usuario = False
    fallidas = []

    total = len(imagenes)
    print(f"Procesando {total} imágenes desde {input_dir} → {output_dir} ...\n")
    if args.review:
        print("Modo revisión manual activado (Enter=guardar, otra tecla=descartar, q/Esc=salir).\n")

    for idx, img_path in enumerate(imagenes, 1):
        print(f"[{idx}/{total}] {img_path.name}... ", end='', flush=True)
        imagen = cv2.imread(str(img_path))
        if imagen is None:
            fallidas.append((img_path.name, 'no se pudo leer'))
            print("no se pudo leer")
            continue

        _, bbox = detector.detectar_alga_desde_centro(imagen)
        if bbox is None:
            fallidas.append((img_path.name, 'no se detectó alga'))
            print("sin detección")
            continue

        bbox_norm = bbox_a_formato_yolo(bbox, imagen.shape)
        if bbox_norm is None:
            fallidas.append((img_path.name, 'bbox inválido'))
            print("bbox inválido")
            continue

        if args.review:
            decision = revisar_manual(imagen, bbox, img_path.name, args.review_max_size)
            if decision is None:
                abortado_por_usuario = True
                print("abortar")
                break
            if not decision:
                descartadas_manual += 1
                fallidas.append((img_path.name, 'descartada manual'))
                print("descartada manual")
                continue

        split = 'val' if random.random() < args.val_split else 'train'
        destino_img = rutas['images'][split] / img_path.name
        destino_label = rutas['labels'][split] / f"{img_path.stem}.txt"

        shutil.copy2(img_path, destino_img)
        escribir_label(destino_label, bbox_norm)

        exitosas += 1
        print(f"OK → {split}")

    if args.review:
        cv2.destroyAllWindows()

    yaml_path = escribir_data_yaml(output_dir)

    print("\nResumen:")
    print(f"  Imágenes procesadas: {total}")
    print(f"  Con anotación: {exitosas}")
    if descartadas_manual:
        print(f"  Descartadas manualmente: {descartadas_manual}")
    print(f"  Fallidas: {len(fallidas)}")
    if abortado_por_usuario:
        print("  Proceso abortado manualmente, no se revisaron todas las imágenes.")
    print(f"  data.yaml: {yaml_path}")

    if fallidas:
        print("\nAlgunas imágenes sin etiqueta:")
        for nombre, motivo in fallidas[:10]:
            print(f"  - {nombre}: {motivo}")
        restantes = len(fallidas) - 10
        if restantes > 0:
            print(f"  ... y {restantes} más.")


if __name__ == '__main__':
    main()
