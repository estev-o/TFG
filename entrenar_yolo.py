#!/usr/bin/env python3
"""Entrena un modelo YOLOv8 sobre el dataset generado en ./yolo/data.yaml.

Usa CPU automáticamente si no hay GPU disponible (o si pasas --device cpu).
"""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenar YOLOv8 con el dataset yolo/data.yaml")
    parser.add_argument("--data", default="yolo/data.yaml", help="Ruta al data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Checkpoint inicial (p.ej. yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Número de épocas")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de imagen")
    parser.add_argument("--batch", type=int, default=16, help="Tamaño de batch")
    parser.add_argument(
        "--device",
        default=None,
        help="Dispositivo (ej. 0, 0,1 o cpu). Por defecto autodetecta GPU y si no hay usa cpu.",
    )
    parser.add_argument("--project", default="runs_yolo", help="Carpeta base de resultados")
    parser.add_argument("--name", default="kelp", help="Nombre del experimento")
    return parser.parse_args()


def elegir_device(user_device: str | None) -> str:
    if user_device:
        return user_device
    return "0" if torch.cuda.is_available() else "cpu"


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"data.yaml no encontrado en {data_path}")

    device = elegir_device(args.device)
    print(f"Entrenando en dispositivo: {device}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
