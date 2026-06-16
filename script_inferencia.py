#!/usr/bin/env python3
"""Pipeline completa de inferencia: YOLO -> normalizacion -> CNN -> CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import (
    convnext_small,
    convnext_tiny,
    efficientnet_b0,
    resnet18,
)
from ultralytics import YOLO

try:
    from pi_heif import register_heif_opener

    register_heif_opener()
except Exception:
    pass


SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    ".heic",
    ".jfif",
}
IVR_ACTIVE_HPI_CLASSES = {1, 2, 3, 4}


@dataclass(frozen=True)
class ReviewProfile:
    name: str
    hpi_gate_margin: float
    yolo_low_conf: float
    description: str


REVIEW_PROFILES = {
    "conservador": ReviewProfile(
        name="conservador",
        hpi_gate_margin=0.15,
        yolo_low_conf=0.40,
        description="revisa mas casos cerca de HPI 0/1 y 4/5",
    ),
    "solo_dudoso": ReviewProfile(
        name="solo_dudoso",
        hpi_gate_margin=0.05,
        yolo_low_conf=0.25,
        description="revisa solo los casos realmente pegados a la frontera",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline completa de inferencia para puntuacion HPI/IVR."
    )
    parser.add_argument(
        "--model-dir",
        default=str(SCRIPT_DIR),
        help="Directorio donde buscar pesos YOLO, pesos CNN y config.json.",
    )
    parser.add_argument("--yolo-pt", default="", help="Ruta explicita al .pt de YOLO.")
    parser.add_argument("--cnn-pt", default="", help="Ruta explicita al .pt de la CNN.")
    parser.add_argument(
        "--cnn-config",
        default="",
        help="Ruta explicita al config.json de la CNN.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="",
        help="Directorio con imagenes a analizar. Si se omite, se pide por teclado.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directorio de salida. Si se omite, se crea en inferencia_resultados/.",
    )
    parser.add_argument(
        "--review-mode",
        default="",
        choices=["", "conservador", "solo_dudoso"],
        help="Perfil de revision manual. Si se omite, se pide por teclado.",
    )
    parser.add_argument(
        "--hpi-gate-margin",
        type=float,
        default=-1.0,
        help="Sobrescribe el margen de duda para las fronteras HPI 0/1 y 4/5.",
    )
    parser.add_argument(
        "--yolo-low-conf",
        type=float,
        default=-1.0,
        help="Sobrescribe la confianza YOLO minima para considerar la prediccion segura.",
    )
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|0")
    parser.add_argument("--open-kernel", type=int, default=1)
    parser.add_argument("--keep-rel", type=float, default=0.25)
    return parser.parse_args()


def prompt_path(message: str, must_exist: bool = True) -> Path:
    while True:
        raw = input(message).strip().strip('"').strip("'")
        path = Path(raw).expanduser()
        if not raw:
            print("Ruta vacia. Introduce una ruta valida.")
            continue
        if must_exist and not path.exists():
            print(f"No existe: {path}")
            continue
        return path


def prompt_review_profile() -> str:
    print("\nModo de revision manual:")
    print("  1) conservador  - revisa mas casos cerca de HPI 0/1 y 4/5")
    print("  2) solo_dudoso  - revisa solo los casos pegados a la frontera")
    while True:
        raw = input("Elige modo [1/2]: ").strip().lower()
        if raw in {"1", "conservador"}:
            return "conservador"
        if raw in {"2", "solo_dudoso", "dudoso"}:
            return "solo_dudoso"
        print("Opcion no valida. Usa 1 o 2.")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se pidio CUDA pero no esta disponible.")
    if device_arg.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError("Se pidio GPU numerica pero CUDA no esta disponible.")
        return torch.device(f"cuda:{device_arg}")
    return torch.device(device_arg)


def select_file_interactively(candidates: Sequence[Path], label: str) -> Path:
    if not candidates:
        raise FileNotFoundError(f"No se encontraron candidatos para {label}.")
    if len(candidates) == 1:
        return candidates[0]

    print(f"\nSe encontraron varios candidatos para {label}:")
    for idx, path in enumerate(candidates, 1):
        print(f"  {idx}) {path}")
    while True:
        raw = input(f"Elige {label} [1-{len(candidates)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(candidates):
            return candidates[int(raw) - 1]
        print("Opcion no valida.")


def resolve_files(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"No existe --model-dir: {model_dir}")

    config_path = Path(args.cnn_config).expanduser() if args.cnn_config else model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            "No se encontro config.json. Copia el config de la run junto al script "
            "o usa --cnn-config."
        )
    config_path = config_path.resolve()

    candidate_dirs = [model_dir]
    if config_path.parent != model_dir:
        candidate_dirs.append(config_path.parent)
    pt_candidates = sorted(
        {p.resolve() for directory in candidate_dirs for p in directory.glob("*.pt")}
    )
    if args.yolo_pt:
        yolo_pt = Path(args.yolo_pt).expanduser().resolve()
    else:
        yolo_named = [p for p in pt_candidates if "yolo" in p.name.lower()]
        yolo_pt = select_file_interactively(yolo_named or pt_candidates, "YOLO .pt").resolve()
    if not yolo_pt.exists():
        raise FileNotFoundError(f"No existe el YOLO .pt: {yolo_pt}")

    if args.cnn_pt:
        cnn_pt = Path(args.cnn_pt).expanduser().resolve()
    else:
        cnn_candidates = [p for p in pt_candidates if p.resolve() != yolo_pt]
        cnn_named = [
            p
            for p in cnn_candidates
            if any(token in p.name.lower() for token in ("cnn", "best", "last"))
        ]
        cnn_pt = select_file_interactively(cnn_named or cnn_candidates, "CNN .pt").resolve()
    if not cnn_pt.exists():
        raise FileNotFoundError(f"No existe la CNN .pt: {cnn_pt}")

    return yolo_pt, cnn_pt, config_path


def collect_images(dataset_dir: Path) -> list[Path]:
    images = [
        p
        for p in sorted(dataset_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    if not images:
        raise FileNotFoundError(f"No se encontraron imagenes en {dataset_dir}")
    return images


def safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def read_image_bgr(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        with Image.open(path) as pil_img:
            rgb = np.array(pil_img.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def ratio_borde(mask_binaria: np.ndarray) -> float:
    h, w = mask_binaria.shape[:2]
    if h == 0 or w == 0:
        return 1.0
    top = mask_binaria[0, :] > 0
    bottom = mask_binaria[-1, :] > 0
    if h > 2:
        left = mask_binaria[1:-1, 0] > 0
        right = mask_binaria[1:-1, -1] > 0
    else:
        left = np.zeros((0,), dtype=bool)
        right = np.zeros((0,), dtype=bool)
    total = top.size + bottom.size + left.size + right.size
    fg = int(top.sum() + bottom.sum() + left.sum() + right.sum())
    return fg / total if total else 1.0


def componentes_relevantes(
    mask_binaria: np.ndarray, keep_rel: float
) -> tuple[Optional[np.ndarray], int, int, float]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_binaria, connectivity=8
    )
    if num_labels <= 1:
        return None, 0, 0, 1.0

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    mayor = int(areas.max())
    keep_rel = max(0.0, min(1.0, float(keep_rel)))
    umbral_area = max(32, int(mayor * keep_rel))

    labels_keep = np.where(areas >= umbral_area)[0] + 1
    if labels_keep.size == 0:
        labels_keep = np.array([1 + int(np.argmax(areas))], dtype=np.int64)

    componente = np.zeros_like(mask_binaria, dtype=np.uint8)
    for label in labels_keep:
        componente[labels == int(label)] = 255

    area_total = int(np.count_nonzero(componente))
    borde = ratio_borde(componente)
    return componente, area_total, int(labels_keep.size), borde


def construir_mascara_alga(
    img_bgr: np.ndarray, open_kernel: int, keep_rel: float
) -> tuple[Optional[np.ndarray], str, int]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_otsu_inv = cv2.bitwise_not(mask_otsu)

    if open_kernel and open_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_kernel, open_kernel)
        )
        mask_otsu = cv2.morphologyEx(mask_otsu, cv2.MORPH_OPEN, kernel)
        mask_otsu_inv = cv2.morphologyEx(mask_otsu_inv, cv2.MORPH_OPEN, kernel)

    comp_bin, area_bin, n_bin, borde_bin = componentes_relevantes(mask_otsu, keep_rel)
    comp_inv, area_inv, n_inv, borde_inv = componentes_relevantes(
        mask_otsu_inv, keep_rel
    )

    if area_bin == 0 and area_inv == 0:
        return None, "none", 0
    if area_bin == 0:
        return comp_inv, f"inv n={n_inv} b={borde_inv:.2f}", area_inv
    if area_inv == 0:
        return comp_bin, f"bin n={n_bin} b={borde_bin:.2f}", area_bin
    if abs(borde_bin - borde_inv) >= 0.08:
        if borde_bin < borde_inv:
            return comp_bin, f"bin n={n_bin} b={borde_bin:.2f}", area_bin
        return comp_inv, f"inv n={n_inv} b={borde_inv:.2f}", area_inv
    if area_inv > area_bin:
        return comp_inv, f"inv n={n_inv} b={borde_inv:.2f}", area_inv
    return comp_bin, f"bin n={n_bin} b={borde_bin:.2f}", area_bin


def crop_best_detection(
    yolo_model: YOLO,
    img_path: Path,
    crop_path: Path,
    device: torch.device,
    imgsz: int,
    conf: float,
) -> tuple[bool, Optional[float], Optional[tuple[int, int, int, int]], str]:
    img = read_image_bgr(img_path)
    if img is None:
        return False, None, None, "image_unreadable"

    yolo_device = "cpu" if device.type == "cpu" else str(device.index or 0)
    preds = yolo_model.predict(
        source=str(img_path),
        imgsz=imgsz,
        device=yolo_device,
        conf=conf,
        verbose=False,
    )
    if not preds or preds[0].boxes is None or len(preds[0].boxes) == 0:
        return False, None, None, "yolo_no_detection"

    boxes = preds[0].boxes
    best_idx = int(boxes.conf.argmax().item())
    best_conf = float(boxes.conf[best_idx].item())
    x1, y1, x2, y2 = boxes.xyxy[best_idx].int().tolist()
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return False, best_conf, (x1, y1, x2, y2), "yolo_invalid_bbox"

    crop = img[y1:y2, x1:x2]
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(crop_path), crop):
        return False, best_conf, (x1, y1, x2, y2), "crop_write_failed"
    return True, best_conf, (x1, y1, x2, y2), ""


def normalize_crop(
    crop_path: Path,
    norm_path: Path,
    max_side: int,
    open_kernel: int,
    keep_rel: float,
) -> tuple[bool, str, int, int, int]:
    img = read_image_bgr(crop_path)
    if img is None:
        return False, "crop_unreadable", 0, 0, 0

    mask, polarity, area = construir_mascara_alga(img, open_kernel, keep_rel)
    if mask is None:
        return False, "normalization_no_mask", 0, 0, 0

    h, w = mask.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side), dtype=np.uint8)
    offset_y = (side - h) // 2
    offset_x = (side - w) // 2
    canvas[offset_y : offset_y + h, offset_x : offset_x + w] = mask
    if max_side > 0 and side != max_side:
        canvas = cv2.resize(canvas, (max_side, max_side), interpolation=cv2.INTER_NEAREST)
    canvas = np.where(canvas > 0, 255, 0).astype(np.uint8)

    norm_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(norm_path), canvas):
        return False, "normalization_write_failed", area, w, h
    return True, polarity, area, w, h


def _build_feature_backbone(name: str, pretrained: bool = False) -> tuple[nn.Module, int]:
    if name == "resnet18":
        model = resnet18(weights=None)
        feature_dim = int(model.fc.in_features)
        model.fc = nn.Identity()
        return model, feature_dim
    if name == "efficientnet_b0":
        model = efficientnet_b0(weights=None)
        feature_dim = int(model.classifier[1].in_features)
        model.classifier[1] = nn.Identity()
        return model, feature_dim
    if name == "convnext_tiny":
        model = convnext_tiny(weights=None)
        feature_dim = int(model.classifier[2].in_features)
        model.classifier[2] = nn.Identity()
        return model, feature_dim
    if name == "convnext_small":
        model = convnext_small(weights=None)
        feature_dim = int(model.classifier[2].in_features)
        model.classifier[2] = nn.Identity()
        return model, feature_dim
    raise ValueError(f"Modelo no soportado: {name}")


class HpiCoralIvrScoreConditionedModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes_hpi: int,
        ivr_conditioning_source: str = "hpi_probs",
        ivr_hidden_dim: int = 128,
        ivr_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone, feature_dim = _build_feature_backbone(backbone_name)
        self.hpi_dim = num_classes_hpi - 1
        self.ivr_conditioning_source = ivr_conditioning_source
        self.hpi_head = nn.Linear(feature_dim, self.hpi_dim)
        self.ivr_adapter = nn.Sequential(
            nn.Linear(feature_dim + self.hpi_dim, ivr_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=ivr_dropout),
        )
        self.ivr_score_head = nn.Linear(ivr_hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        hpi_logits = self.hpi_head(features)
        if self.ivr_conditioning_source == "hpi_logits":
            hpi_condition = hpi_logits
        else:
            hpi_condition = torch.sigmoid(hpi_logits)
        ivr_hidden = self.ivr_adapter(torch.cat([features, hpi_condition], dim=1))
        ivr_score_logit = self.ivr_score_head(ivr_hidden)
        return torch.cat([hpi_logits, ivr_score_logit], dim=1)


class HpiCoralIvrDualConditionedModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes_hpi: int,
        ivr_conditioning_source: str = "hpi_probs",
        ivr_hidden_dim: int = 128,
        ivr_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone, feature_dim = _build_feature_backbone(backbone_name)
        self.hpi_dim = num_classes_hpi - 1
        self.ivr_conditioning_source = ivr_conditioning_source
        self.hpi_head = nn.Linear(feature_dim, self.hpi_dim)
        self.ivr_adapter = nn.Sequential(
            nn.Linear(feature_dim + self.hpi_dim, ivr_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=ivr_dropout),
        )
        self.ivr_app_head = nn.Linear(ivr_hidden_dim, 1)
        self.ivr_score_head = nn.Linear(ivr_hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        hpi_logits = self.hpi_head(features)
        if self.ivr_conditioning_source == "hpi_logits":
            hpi_condition = hpi_logits
        else:
            hpi_condition = torch.sigmoid(hpi_logits)
        ivr_hidden = self.ivr_adapter(torch.cat([features, hpi_condition], dim=1))
        return torch.cat(
            [
                hpi_logits,
                self.ivr_app_head(ivr_hidden),
                self.ivr_score_head(ivr_hidden),
            ],
            dim=1,
        )


def build_plain_model(name: str, output_dim: int) -> nn.Module:
    if name == "resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, output_dim)
        return model
    if name == "efficientnet_b0":
        model = efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, output_dim)
        return model
    if name == "convnext_tiny":
        model = convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, output_dim)
        return model
    if name == "convnext_small":
        model = convnext_small(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, output_dim)
        return model
    raise ValueError(f"Modelo no soportado: {name}")


def build_cnn_from_config(config: dict[str, Any]) -> nn.Module:
    model_name = str(config.get("model", "convnext_tiny"))
    head_type = str(config.get("head_type", ""))
    num_classes_hpi = int(config.get("num_classes_hpi", 7))
    output_dim = int(config.get("output_dim", 0))

    if head_type == "hpi_coral_ivr_score_conditioned":
        return HpiCoralIvrScoreConditionedModel(
            backbone_name=model_name,
            num_classes_hpi=num_classes_hpi,
            ivr_conditioning_source=str(config.get("ivr_conditioning_source", "hpi_probs")),
            ivr_hidden_dim=int(config.get("ivr_conditioned_hidden_dim", 128)),
            ivr_dropout=float(config.get("ivr_conditioned_dropout", 0.1)),
        )
    if head_type == "hpi_coral_ivr_dual_conditioned":
        return HpiCoralIvrDualConditionedModel(
            backbone_name=model_name,
            num_classes_hpi=num_classes_hpi,
            ivr_conditioning_source=str(config.get("ivr_conditioning_source", "hpi_probs")),
            ivr_hidden_dim=int(config.get("ivr_conditioned_hidden_dim", 128)),
            ivr_dropout=float(config.get("ivr_conditioned_dropout", 0.1)),
        )
    if output_dim <= 0:
        raise ValueError("config.json no contiene output_dim valido.")
    return build_plain_model(model_name, output_dim)


def load_cnn(cnn_pt: Path, config: dict[str, Any], device: torch.device) -> nn.Module:
    model = build_cnn_from_config(config).to(device)
    try:
        checkpoint = torch.load(cnn_pt, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(cnn_pt, map_location=device)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def parse_ivr_score_targets(raw: Any) -> dict[int, float]:
    if isinstance(raw, dict):
        return {int(k): float(v) for k, v in raw.items() if v is not None}
    if isinstance(raw, list):
        return {idx: float(v) for idx, v in enumerate(raw) if v is not None}
    return {
        1: 0.00,
        2: 0.05,
        3: 0.25,
        4: 0.50,
        5: 0.75,
        6: 0.95,
        7: 1.00,
    }


def decode_ordinal_logits(logits: torch.Tensor) -> torch.Tensor:
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


def ordinal_class_probs(logits: torch.Tensor) -> torch.Tensor:
    probs_gt = torch.sigmoid(logits)
    n, km1 = probs_gt.shape
    n_classes = km1 + 1
    out = torch.zeros((n, n_classes), device=logits.device, dtype=logits.dtype)
    out[:, 0] = 1.0 - probs_gt[:, 0]
    if n_classes > 2:
        out[:, 1:-1] = probs_gt[:, :-1] - probs_gt[:, 1:]
    out[:, -1] = probs_gt[:, -1]
    out = out.clamp_min(0.0)
    return out / out.sum(dim=1, keepdim=True).clamp_min(1e-8)


def nearest_ivr_class(score: float, anchors: dict[int, float]) -> tuple[int, float]:
    classes = sorted(k for k in anchors if k > 0)
    best = min(classes, key=lambda cls: abs(score - anchors[cls]))
    return int(best), float(abs(score - anchors[best]))


def nearest_ivr_boundary_distance(score: float, anchors: dict[int, float]) -> float:
    classes = sorted(k for k in anchors if k > 0)
    boundaries = []
    for left, right in zip(classes, classes[1:]):
        boundaries.append((anchors[left] + anchors[right]) / 2.0)
    if not boundaries:
        return 1.0
    return float(min(abs(score - boundary) for boundary in boundaries))


def build_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def infer_batch(
    model: nn.Module,
    image_paths: Sequence[Path],
    config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    img_size = int(config.get("img_size", 224))
    num_classes_hpi = int(config.get("num_classes_hpi", 7))
    hpi_dim = num_classes_hpi - 1
    head_type = str(config.get("head_type", ""))
    anchors = parse_ivr_score_targets(config.get("ivr_score_targets", []))
    transform = build_transform(img_size)

    tensors = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        tensors.append(transform(image))
    x = torch.stack(tensors).to(device)

    with torch.no_grad():
        logits = model(x)

    hpi_logits = logits[:, :hpi_dim]
    hpi_threshold_probs = torch.sigmoid(hpi_logits)
    hpi_probs = ordinal_class_probs(hpi_logits)
    pred_hpi = decode_ordinal_logits(hpi_logits)

    results: list[dict[str, Any]] = []
    for idx in range(len(image_paths)):
        hpi = int(pred_hpi[idx].item())
        hpi_class_prob = float(hpi_probs[idx, hpi].item())
        p_gt_0 = float(hpi_threshold_probs[idx, 0].item())
        p_gt_4 = float(hpi_threshold_probs[idx, 4].item()) if hpi_dim > 4 else None

        if head_type in {"hpi_coral_ivr_score_conditioned", "hpi_coral_ivr_score"}:
            ivr_score = float(torch.sigmoid(logits[idx, hpi_dim]).item())
            raw_ivr, ivr_anchor_distance = nearest_ivr_class(ivr_score, anchors)
            pred_ivr = raw_ivr if hpi in IVR_ACTIVE_HPI_CLASSES else 0
        elif head_type in {"hpi_coral_ivr_dual", "hpi_coral_ivr_dual_conditioned"}:
            ivr_app_prob = float(torch.sigmoid(logits[idx, hpi_dim]).item())
            ivr_score = float(torch.sigmoid(logits[idx, hpi_dim + 1]).item())
            raw_ivr, ivr_anchor_distance = nearest_ivr_class(ivr_score, anchors)
            threshold = float(config.get("ivr_app_threshold", 0.5))
            pred_ivr = raw_ivr if ivr_app_prob >= threshold else 0
        else:
            ivr_score = None
            ivr_anchor_distance = None
            pred_ivr = None

        results.append(
            {
                "pred_hpi": hpi,
                "pred_hpi_prob": hpi_class_prob,
                "hpi_p_gt_0": p_gt_0,
                "hpi_p_gt_4": p_gt_4,
                "hpi_margin_0_1": abs(p_gt_0 - 0.5),
                "hpi_margin_4_5": None if p_gt_4 is None else abs(p_gt_4 - 0.5),
                "pred_ivr": pred_ivr,
                "pred_ivr_score": ivr_score,
                "pred_ivr_anchor_distance": ivr_anchor_distance,
                "pred_ivr_boundary_distance": (
                    None
                    if ivr_score is None
                    else nearest_ivr_boundary_distance(ivr_score, anchors)
                ),
                "ivr_forced_zero_by_hpi": int(hpi not in IVR_ACTIVE_HPI_CLASSES),
            }
        )
    return results


def make_base_row(img_path: Path) -> dict[str, Any]:
    return {
        "photo_cod": img_path.stem,
        "image_path": img_path.as_posix(),
        "status": "pending",
        "crop_path": "",
        "normalized_path": "",
        "bbox_x1": "",
        "bbox_y1": "",
        "bbox_x2": "",
        "bbox_y2": "",
        "yolo_conf": "",
        "normalization_polarity": "",
        "mask_area": "",
        "crop_width": "",
        "crop_height": "",
        "pred_hpi": "",
        "pred_hpi_prob": "",
        "hpi_p_gt_0": "",
        "hpi_p_gt_4": "",
        "hpi_margin_0_1": "",
        "hpi_margin_4_5": "",
        "pred_ivr": "",
        "pred_ivr_score": "",
        "pred_ivr_anchor_distance": "",
        "pred_ivr_boundary_distance": "",
        "ivr_forced_zero_by_hpi": "",
        "review_mode": "",
        "needs_review": 1,
        "review_reasons": "",
    }


def update_review_flags(
    row: dict[str, Any],
    profile: ReviewProfile,
    hpi_gate_margin: float,
    yolo_low_conf: float,
) -> None:
    reasons: list[str] = []
    if row["status"] != "ok":
        reasons.append(str(row["status"]))

    try:
        yolo_conf = float(row["yolo_conf"])
        if yolo_conf < yolo_low_conf:
            reasons.append("yolo_low_confidence")
    except (TypeError, ValueError):
        if row["status"] == "ok":
            reasons.append("yolo_conf_missing")

    for margin_key, reason in (
        ("hpi_margin_0_1", "hpi_boundary_0_1"),
        ("hpi_margin_4_5", "hpi_boundary_4_5"),
    ):
        value = row.get(margin_key, "")
        if value == "" or value is None:
            continue
        if float(value) <= hpi_gate_margin:
            reasons.append(reason)

    row["review_mode"] = profile.name
    row["needs_review"] = int(bool(reasons))
    row["review_reasons"] = ";".join(reasons)


def format_confidence(value: Any) -> str:
    if value == "" or value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def ivr_confidence_for_experts(row: dict[str, Any]) -> str:
    if row.get("status") != "ok":
        return ""
    if int(row.get("ivr_forced_zero_by_hpi") or 0) == 1:
        try:
            pred_hpi = int(row.get("pred_hpi", ""))
        except (TypeError, ValueError):
            return ""
        if pred_hpi == 0:
            return format_confidence(row.get("hpi_margin_0_1", ""))
        if pred_hpi in {5, 6}:
            return format_confidence(row.get("hpi_margin_4_5", ""))
        return ""
    return format_confidence(row.get("pred_ivr_boundary_distance", ""))


def expert_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "imagen": row.get("photo_cod", ""),
        "pred_hpi": row.get("pred_hpi", ""),
        "pred_ivr": row.get("pred_ivr", ""),
        "confianza_hpi": format_confidence(row.get("pred_hpi_prob", "")),
        "confianza_ivr": ivr_confidence_for_experts(row),
        "confianza_yolo": format_confidence(row.get("yolo_conf", "")),
        "necesita_revision": row.get("needs_review", 1),
        "motivo_revision": row.get("review_reasons", ""),
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "imagen",
        "pred_hpi",
        "pred_ivr",
        "confianza_hpi",
        "confianza_ivr",
        "confianza_yolo",
        "necesita_revision",
        "motivo_revision",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(expert_row(row))


def main() -> None:
    args = parse_args()
    yolo_pt, cnn_pt, config_path = resolve_files(args)

    dataset_dir = (
        Path(args.dataset_dir).expanduser()
        if args.dataset_dir
        else prompt_path("\nRuta del directorio de imagenes a analizar: ")
    ).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"No existe el dataset: {dataset_dir}")

    review_mode = args.review_mode or prompt_review_profile()
    profile = REVIEW_PROFILES[review_mode]
    hpi_gate_margin = (
        float(args.hpi_gate_margin)
        if args.hpi_gate_margin >= 0
        else profile.hpi_gate_margin
    )
    yolo_low_conf = (
        float(args.yolo_low_conf) if args.yolo_low_conf >= 0 else profile.yolo_low_conf
    )

    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else SCRIPT_DIR
        / "inferencia_resultados"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    ).resolve()
    crops_dir = output_dir / "recortes_yolo"
    norm_dir = output_dir / "normalizadas"

    print("\nConfiguracion:")
    print(f"  YOLO: {yolo_pt}")
    print(f"  CNN: {cnn_pt}")
    print(f"  config: {config_path}")
    print(f"  dataset: {dataset_dir}")
    print(f"  salida: {output_dir}")
    print(
        f"  revision: {profile.name} "
        f"(margen HPI={hpi_gate_margin:.2f}, yolo_low_conf={yolo_low_conf:.2f})"
    )

    with config_path.open(encoding="utf-8") as f:
        config: dict[str, Any] = json.load(f)

    if str(config.get("target", "both")) != "both":
        raise ValueError("Este script espera una CNN target=both para producir HPI e IVR.")

    images = collect_images(dataset_dir)
    device = resolve_device(args.device)
    yolo_model = YOLO(str(yolo_pt))
    cnn_model = load_cnn(cnn_pt, config, device)

    rows = [make_base_row(path) for path in images]
    successful_crop_indices: list[int] = []
    max_side = 0

    print(f"\n1/3 Deteccion YOLO y recorte: {len(images)} imagenes")
    for idx, img_path in enumerate(images, 1):
        row = rows[idx - 1]
        crop_path = crops_dir / f"{safe_stem(img_path)}_yolo.jpg"
        ok, best_conf, bbox, error = crop_best_detection(
            yolo_model=yolo_model,
            img_path=img_path,
            crop_path=crop_path,
            device=device,
            imgsz=args.yolo_imgsz,
            conf=args.yolo_conf,
        )
        if best_conf is not None:
            row["yolo_conf"] = f"{best_conf:.6f}"
        if bbox is not None:
            row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"] = bbox
        if not ok:
            row["status"] = error
            continue
        row["crop_path"] = crop_path.as_posix()
        crop_img = read_image_bgr(crop_path)
        if crop_img is None:
            row["status"] = "crop_unreadable"
            continue
        h, w = crop_img.shape[:2]
        row["crop_width"] = w
        row["crop_height"] = h
        max_side = max(max_side, h, w)
        successful_crop_indices.append(idx - 1)

    print(f"  Recortes validos: {len(successful_crop_indices)}")
    if not successful_crop_indices:
        for row in rows:
            update_review_flags(row, profile, hpi_gate_margin, yolo_low_conf)
        write_csv(output_dir / "puntuaciones_total.csv", rows)
        write_csv(output_dir / "puntuaciones_seguras.csv", [])
        write_csv(output_dir / "puntuaciones_revision_humana.csv", rows)
        print(f"No hubo recortes validos. CSVs guardados en: {output_dir}")
        return

    print("\n2/3 Normalizacion de recortes")
    infer_indices: list[int] = []
    for row_idx in successful_crop_indices:
        row = rows[row_idx]
        crop_path = Path(str(row["crop_path"]))
        norm_path = norm_dir / f"{crop_path.stem}.png"
        ok, polarity, area, crop_w, crop_h = normalize_crop(
            crop_path=crop_path,
            norm_path=norm_path,
            max_side=max_side,
            open_kernel=args.open_kernel,
            keep_rel=args.keep_rel,
        )
        row["normalization_polarity"] = polarity
        row["mask_area"] = area
        if crop_w:
            row["crop_width"] = crop_w
        if crop_h:
            row["crop_height"] = crop_h
        if not ok:
            row["status"] = polarity
            continue
        row["normalized_path"] = norm_path.as_posix()
        infer_indices.append(row_idx)

    print(f"  Normalizadas validas: {len(infer_indices)}")

    print("\n3/3 Inferencia CNN")
    for start in range(0, len(infer_indices), args.batch_size):
        batch_indices = infer_indices[start : start + args.batch_size]
        batch_paths = [Path(str(rows[i]["normalized_path"])) for i in batch_indices]
        predictions = infer_batch(cnn_model, batch_paths, config, device)
        for row_idx, pred in zip(batch_indices, predictions):
            rows[row_idx].update(
                {
                    key: "" if value is None else value
                    for key, value in pred.items()
                }
            )
            rows[row_idx]["status"] = "ok"

    for row in rows:
        update_review_flags(row, profile, hpi_gate_margin, yolo_low_conf)

    safe_rows = [row for row in rows if int(row["needs_review"]) == 0]
    review_rows = [row for row in rows if int(row["needs_review"]) == 1]

    total_csv = output_dir / "puntuaciones_total.csv"
    safe_csv = output_dir / "puntuaciones_seguras.csv"
    review_csv = output_dir / "puntuaciones_revision_humana.csv"
    write_csv(total_csv, rows)
    write_csv(safe_csv, safe_rows)
    write_csv(review_csv, review_rows)

    print("\nResumen:")
    print(f"  Imagenes totales: {len(rows)}")
    print(f"  Seguras: {len(safe_rows)}")
    print(f"  Requieren revision humana: {len(review_rows)}")
    print(f"  CSV total: {total_csv}")
    print(f"  CSV seguras: {safe_csv}")
    print(f"  CSV revision: {review_csv}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelado por el usuario.")
        sys.exit(130)
