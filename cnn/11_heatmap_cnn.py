#!/usr/bin/env python3
"""Genera heatmaps Grad-CAM para una prediccion de la CNN ordinal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import (
    convnext_small,
    convnext_tiny,
    efficientnet_b0,
    resnet18,
)

from hpi_ivr_dual_conditioned import (
    build_conditioned_dual_model,
    build_conditioned_score_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera heatmaps Grad-CAM para una imagen usando una run CNN"
    )
    parser.add_argument("--run-dir", required=True, help="Directorio de run con config.json")
    parser.add_argument("--image-path", required=True, help="Ruta de la imagen a analizar")
    parser.add_argument("--photo-cod", default="", help="Codigo de foto para nombrar salidas")
    parser.add_argument("--checkpoint", default="best.pt", help="Checkpoint dentro de run-dir")
    parser.add_argument(
        "--model",
        default="",
        choices=["", "resnet18", "efficientnet_b0", "convnext_small", "convnext_tiny"],
    )
    parser.add_argument("--target", default="auto", choices=["auto", "both", "hpi", "ivr"])
    parser.add_argument("--img-size", type=int, default=0, help="0 para usar config.json")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directorio de salida. Si vacio: <run-dir>/heatmaps_single",
    )
    parser.add_argument("--true-hpi", type=int, default=-1)
    parser.add_argument("--true-ivr", type=int, default=-1)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se pidio --device cuda pero CUDA no esta disponible.")
    return torch.device(device_arg)


def resolve_run_dir(run_dir: Path) -> Path:
    if (run_dir / "config.json").exists():
        return run_dir

    children = [p for p in run_dir.iterdir() if p.is_dir()]
    with_config = [p for p in children if (p / "config.json").exists()]
    if len(with_config) == 1:
        return with_config[0]
    raise FileNotFoundError(
        f"No se encontro config.json en {run_dir} ni en un unico subdirectorio."
    )


def output_dim_for_target(
    head_type: str, target: str, num_classes_hpi: int, num_classes_ivr: int
) -> int:
    if head_type == "hpi_coral_ivr_score":
        if target == "both":
            return num_classes_hpi
        if target == "hpi":
            return num_classes_hpi - 1
        raise ValueError(
            "La variante hpi_coral_ivr_score no soporta target=ivr en heatmaps."
        )
    if head_type == "hpi_coral_ivr_score_conditioned":
        if target == "both":
            return num_classes_hpi
        if target == "hpi":
            return num_classes_hpi - 1
        raise ValueError(
            "La variante hpi_coral_ivr_score_conditioned no soporta target=ivr en heatmaps."
        )
    if head_type in {"hpi_coral_ivr_dual", "hpi_coral_ivr_dual_conditioned"}:
        if target == "both":
            return num_classes_hpi + 1
        if target == "hpi":
            return num_classes_hpi - 1
        raise ValueError(
            f"La variante {head_type} no soporta target=ivr en heatmaps."
        )
    if target == "both":
        return (num_classes_hpi - 1) + (num_classes_ivr - 1)
    if target == "hpi":
        return num_classes_hpi - 1
    return num_classes_ivr - 1


def build_model(
    name: str,
    output_dim: int,
    head_type: str = "ordinal_coral",
    num_classes_hpi: int = 0,
    target: str = "both",
    ivr_conditioning_source: str = "hpi_probs",
    ivr_conditioned_hidden_dim: int = 128,
    ivr_conditioned_dropout: float = 0.1,
) -> nn.Module:
    if head_type == "hpi_coral_ivr_dual_conditioned":
        return build_conditioned_dual_model(
            name,
            pretrained=False,
            num_classes_hpi=num_classes_hpi,
            target=target,
            ivr_conditioning_source=ivr_conditioning_source,
            ivr_hidden_dim=ivr_conditioned_hidden_dim,
            ivr_dropout=ivr_conditioned_dropout,
        )
    if head_type == "hpi_coral_ivr_score_conditioned":
        return build_conditioned_score_model(
            name,
            pretrained=False,
            num_classes_hpi=num_classes_hpi,
            target=target,
            ivr_conditioning_source=ivr_conditioning_source,
            ivr_hidden_dim=ivr_conditioned_hidden_dim,
            ivr_dropout=ivr_conditioned_dropout,
        )
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


def decode_ordinal_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs > 0.5).sum(dim=1)


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
    out = out / out.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return out


def coarse_probs_from_class_probs(
    class_probs: torch.Tensor, bins: Sequence[tuple[int, int]]
) -> torch.Tensor:
    coarse_probs = []
    for start, end in bins:
        coarse_probs.append(class_probs[:, start : end + 1].sum(dim=1))
    out = torch.stack(coarse_probs, dim=1)
    out = out / out.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return out


def decode_ivr_with_coarse_fine(
    ivr_logits: torch.Tensor, bins: Sequence[tuple[int, int]]
) -> torch.Tensor:
    class_probs = ordinal_class_probs(ivr_logits)
    coarse_probs = coarse_probs_from_class_probs(class_probs, bins)
    coarse_idx = coarse_probs.argmax(dim=1)
    pred = torch.zeros((ivr_logits.shape[0],), device=ivr_logits.device, dtype=torch.int64)
    for bin_idx, (start, end) in enumerate(bins):
        mask = coarse_idx == bin_idx
        if not mask.any():
            continue
        local_probs = class_probs[mask, start : end + 1]
        local_pred = local_probs.argmax(dim=1) + start
        pred[mask] = local_pred.to(torch.int64)
    return pred


def parse_bins_from_config(raw: Any) -> list[tuple[int, int]]:
    if raw is None:
        return []
    bins: list[tuple[int, int]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Bin coarse invalido en config: {item}")
        bins.append((int(item[0]), int(item[1])))
    return bins


def remap_raw_ivr_label(
    raw_ivr: int, bins: Sequence[tuple[int, int]], field: str = "ivr"
) -> int:
    for group_idx, (start, end) in enumerate(bins):
        if start <= raw_ivr <= end:
            return group_idx
    raise ValueError(f"{field} fuera de los bins de agrupacion IVR: {raw_ivr}")


def parse_ivr_score_targets(raw: Any) -> dict[int, float]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        out: dict[int, float] = {}
        for k, v in raw.items():
            if v is None:
                continue
            out[int(k)] = float(v)
        return out
    if isinstance(raw, list):
        out = {}
        for idx, value in enumerate(raw):
            if value is None:
                continue
            out[idx] = float(value)
        return out
    raise ValueError(f"ivr_score_targets invalido en config: {raw}")


def decode_ivr_score_to_class(
    scores: torch.Tensor,
    ivr_score_targets: dict[int, float],
    hpi_pred: Optional[torch.Tensor] = None,
    use_hpi_gate: bool = True,
) -> torch.Tensor:
    anchor_classes_raw = sorted(k for k in ivr_score_targets.keys() if k > 0)
    if not anchor_classes_raw:
        raise ValueError("No hay anchors de ivr_score_targets para decodificar IVR score.")

    anchor_classes = torch.tensor(
        anchor_classes_raw,
        device=scores.device,
        dtype=torch.int64,
    )
    anchor_scores = torch.tensor(
        [ivr_score_targets[k] for k in anchor_classes_raw],
        device=scores.device,
        dtype=scores.dtype,
    )
    pred = torch.zeros(scores.shape[0], device=scores.device, dtype=torch.int64)
    valid_mask = torch.ones(scores.shape[0], device=scores.device, dtype=torch.bool)
    if use_hpi_gate and hpi_pred is not None:
        valid_mask = (hpi_pred > 0) & (hpi_pred < 5)
    if valid_mask.any():
        distances = (scores[valid_mask].unsqueeze(1) - anchor_scores.unsqueeze(0)).abs()
        idx = distances.argmin(dim=1)
        pred[valid_mask] = anchor_classes[idx]
    return pred


def ordinal_score_for_class(logits: torch.Tensor, class_idx: int) -> torch.Tensor:
    probs_gt = torch.sigmoid(logits)
    n_classes = logits.shape[1] + 1
    if class_idx < 0 or class_idx >= n_classes:
        raise ValueError(f"Clase ordinal fuera de rango: {class_idx}")
    if class_idx == 0:
        return 1.0 - probs_gt[:, 0]
    if class_idx == n_classes - 1:
        return probs_gt[:, -1]
    return probs_gt[:, class_idx - 1] - probs_gt[:, class_idx]


def resolve_target_layer(model_name: str, model: nn.Module) -> nn.Module:
    backbone = model.backbone if hasattr(model, "backbone") else model
    if model_name == "resnet18":
        return backbone.layer4[-1]
    if model_name == "efficientnet_b0":
        return backbone.features[-1]
    if model_name in {"convnext_tiny", "convnext_small"}:
        return backbone.features[-1]
    raise ValueError(f"No hay target layer Grad-CAM para: {model_name}")


class GradCAMHook:
    def __init__(self, target_layer: nn.Module):
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = target_layer.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output

    def _backward_hook(
        self,
        _module: nn.Module,
        _grad_inputs: tuple[Optional[torch.Tensor], ...],
        grad_outputs: tuple[Optional[torch.Tensor], ...],
    ) -> None:
        self.gradients = grad_outputs[0]

    def build_cam(self, output_size: tuple[int, int]) -> np.ndarray:
        if self.activations is None or self.gradients is None:
            raise RuntimeError("No hay activaciones/gradientes para construir Grad-CAM.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=output_size, mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-8)
        return cam.detach().cpu().numpy()

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()


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


def load_display_image(image_path: str, img_size: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    return image.resize((img_size, img_size))


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    try:
        from matplotlib import colormaps
    except ImportError as exc:
        raise RuntimeError("matplotlib es necesario para generar heatmaps.") from exc

    image_np = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    heat_rgb = colormaps["jet"](cam)[..., :3].astype(np.float32)
    overlay = ((1.0 - alpha) * image_np) + (alpha * heat_rgb)
    overlay = np.clip(overlay, 0.0, 1.0)
    return Image.fromarray((overlay * 255.0).astype(np.uint8))


def build_panel(original: Image.Image, overlay: Image.Image) -> Image.Image:
    panel = Image.new("RGB", (original.width * 2, original.height))
    panel.paste(original, (0, 0))
    panel.paste(overlay, (original.width, 0))
    return panel


def save_prediction_heatmaps(
    model: nn.Module,
    model_name: str,
    head_type: str,
    image_path: str,
    photo_code: str,
    target: str,
    img_size: int,
    device: torch.device,
    output_dir: Path,
    num_classes_hpi: int,
    num_classes_ivr: int,
    use_ivr_coarse_fine: bool = False,
    ivr_coarse_bins: Optional[Sequence[tuple[int, int]]] = None,
    ivr_score_targets: Optional[dict[int, float]] = None,
    ivr_score_hpi_gate: bool = True,
    ivr_app_threshold: float = 0.5,
    pred_hpi: Optional[int] = None,
    pred_ivr: Optional[int] = None,
    true_hpi: Optional[int] = None,
    true_ivr: Optional[int] = None,
) -> list[dict[str, Any]]:
    transform = build_transform(img_size)
    display_image = load_display_image(image_path, img_size)
    input_tensor = transform(display_image).unsqueeze(0).to(device)
    output_dir.mkdir(parents=True, exist_ok=True)

    hook = GradCAMHook(resolve_target_layer(model_name, model))
    records: list[dict[str, Any]] = []

    try:
        with torch.enable_grad():
            logits = model(input_tensor)
            tasks: list[dict[str, Any]] = []

            if target == "both":
                hpi_dim = num_classes_hpi - 1
                hpi_logits = logits[:, :hpi_dim]
                pred_hpi_effective = (
                    int(decode_ordinal_logits(hpi_logits)[0].item())
                    if pred_hpi is None
                    else int(pred_hpi)
                )

                tasks.append(
                    {
                        "task": "hpi",
                        "pred_label": pred_hpi_effective,
                        "true_label": true_hpi,
                        "score": ordinal_score_for_class(hpi_logits, pred_hpi_effective).sum(),
                    }
                )
                if head_type in {
                    "hpi_coral_ivr_score",
                    "hpi_coral_ivr_score_conditioned",
                }:
                    ivr_score_logit = logits[:, hpi_dim]
                    ivr_score_value = torch.sigmoid(ivr_score_logit)
                    if pred_ivr is None:
                        pred_ivr_effective = int(
                            decode_ivr_score_to_class(
                                ivr_score_value,
                                ivr_score_targets or {},
                                hpi_pred=torch.tensor(
                                    [pred_hpi_effective], device=ivr_score_value.device
                                ),
                                use_hpi_gate=(
                                    True
                                    if head_type == "hpi_coral_ivr_score_conditioned"
                                    else ivr_score_hpi_gate
                                ),
                            )[0].item()
                        )
                    else:
                        pred_ivr_effective = int(pred_ivr)
                    if pred_ivr_effective > 0:
                        anchor = float((ivr_score_targets or {})[pred_ivr_effective])
                        tasks.append(
                            {
                                "task": "ivr",
                                "pred_label": pred_ivr_effective,
                                "true_label": true_ivr,
                                "score": -((ivr_score_value - anchor) ** 2).sum(),
                            }
                        )
                elif head_type in {
                    "hpi_coral_ivr_dual",
                    "hpi_coral_ivr_dual_conditioned",
                }:
                    ivr_app_logit = logits[:, hpi_dim]
                    ivr_score_logit = logits[:, hpi_dim + 1]
                    ivr_app_prob = torch.sigmoid(ivr_app_logit)
                    ivr_score_value = torch.sigmoid(ivr_score_logit)
                    pred_ivr_applicable = bool(
                        float(ivr_app_prob[0].item()) >= ivr_app_threshold
                    )
                    if pred_ivr is None:
                        pred_ivr_effective = int(
                            decode_ivr_score_to_class(
                                ivr_score_value,
                                ivr_score_targets or {},
                                hpi_pred=None,
                                use_hpi_gate=False,
                            )[0].item()
                        )
                        if not pred_ivr_applicable:
                            pred_ivr_effective = 0
                    else:
                        pred_ivr_effective = int(pred_ivr)
                        pred_ivr_applicable = pred_ivr_effective > 0
                    tasks.append(
                        {
                            "task": "ivr_app",
                            "pred_label": int(pred_ivr_applicable),
                            "true_label": (
                                None
                                if true_ivr is None
                                else int(true_ivr > 0)
                            ),
                            "score": ivr_app_logit.sum(),
                        }
                    )
                    if pred_ivr_effective > 0:
                        anchor = float((ivr_score_targets or {})[pred_ivr_effective])
                        tasks.append(
                            {
                                "task": "ivr",
                                "pred_label": pred_ivr_effective,
                                "true_label": true_ivr,
                                "score": -((ivr_score_value - anchor) ** 2).sum(),
                            }
                        )
                else:
                    ivr_logits = logits[:, hpi_dim:]
                    if pred_ivr is None:
                        if use_ivr_coarse_fine:
                            pred_ivr_effective = int(
                                decode_ivr_with_coarse_fine(
                                    ivr_logits, ivr_coarse_bins or []
                                )[0].item()
                            )
                        else:
                            pred_ivr_effective = int(
                                decode_ordinal_logits(ivr_logits)[0].item()
                            )
                    else:
                        pred_ivr_effective = int(pred_ivr)
                    tasks.append(
                        {
                            "task": "ivr",
                            "pred_label": pred_ivr_effective,
                            "true_label": true_ivr,
                            "score": ordinal_score_for_class(
                                ivr_logits, pred_ivr_effective
                            ).sum(),
                        }
                    )
            elif target == "hpi":
                pred_hpi_effective = (
                    int(decode_ordinal_logits(logits)[0].item())
                    if pred_hpi is None
                    else int(pred_hpi)
                )
                tasks.append(
                    {
                        "task": "hpi",
                        "pred_label": pred_hpi_effective,
                        "true_label": true_hpi,
                        "score": ordinal_score_for_class(logits, pred_hpi_effective).sum(),
                    }
                )
            else:
                if head_type == "hpi_coral_ivr_score":
                    raise ValueError(
                        "head_type hpi_coral_ivr_score no soporta target=ivr en heatmaps."
                    )
                if pred_ivr is None:
                    if use_ivr_coarse_fine:
                        pred_ivr_effective = int(
                            decode_ivr_with_coarse_fine(logits, ivr_coarse_bins or [])[0].item()
                        )
                    else:
                        pred_ivr_effective = int(decode_ordinal_logits(logits)[0].item())
                else:
                    pred_ivr_effective = int(pred_ivr)
                tasks.append(
                    {
                        "task": "ivr",
                        "pred_label": pred_ivr_effective,
                        "true_label": true_ivr,
                        "score": ordinal_score_for_class(logits, pred_ivr_effective).sum(),
                    }
                )

            stem = photo_code.strip() or Path(image_path).stem
            for idx, task in enumerate(tasks):
                model.zero_grad(set_to_none=True)
                task["score"].backward(retain_graph=idx < (len(tasks) - 1))
                cam = hook.build_cam((img_size, img_size))
                overlay = overlay_heatmap(display_image, cam)
                panel = build_panel(display_image, overlay)
                task_dir = output_dir / task["task"]
                task_dir.mkdir(parents=True, exist_ok=True)
                suffix = f"_p{task['pred_label']}"
                if task["true_label"] is not None:
                    suffix += f"_t{int(task['true_label'])}"
                out_path = task_dir / f"{stem}_{task['task']}{suffix}.png"
                panel.save(out_path)
                records.append(
                    {
                        "photo_cod": stem,
                        "task": task["task"],
                        "image_path": image_path,
                        "output_path": out_path.as_posix(),
                        "pred_label": int(task["pred_label"]),
                        "true_label": (
                            None
                            if task["true_label"] is None
                            else int(task["true_label"])
                        ),
                    }
                )
    finally:
        hook.close()

    return records


def main() -> None:
    args = parse_args()

    run_dir = resolve_run_dir(Path(args.run_dir))
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "heatmaps_single"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.json").open(encoding="utf-8") as f:
        config: dict[str, Any] = json.load(f)

    model_name = args.model or str(config.get("model", "resnet18"))
    target_name = str(config.get("target", "both")) if args.target == "auto" else args.target
    if target_name not in {"both", "hpi", "ivr"}:
        raise ValueError(f"Target no soportado: {target_name}")
    head_type = str(config.get("head_type", "ordinal_coral"))
    if head_type not in {
        "ordinal_coral",
        "hpi_coral_ivr_score",
        "hpi_coral_ivr_score_conditioned",
        "hpi_coral_ivr_dual",
        "hpi_coral_ivr_dual_conditioned",
    }:
        raise ValueError(f"head_type no soportado en heatmaps: {head_type}")

    num_classes_hpi = int(config.get("num_classes_hpi", 0))
    num_classes_ivr = int(config.get("num_classes_ivr", 0))
    if num_classes_hpi < 2 or num_classes_ivr < 2:
        raise ValueError("num_classes_hpi/num_classes_ivr invalidos en config.json")

    ivr_label_mode_raw = config.get("ivr_label_mode", "raw_8")
    ivr_label_mode = str(ivr_label_mode_raw) if ivr_label_mode_raw else "raw_8"
    ivr_grouping_bins_raw = parse_bins_from_config(config.get("ivr_grouping_bins_raw", []))
    if ivr_label_mode != "raw_8" and not ivr_grouping_bins_raw:
        raise ValueError(
            "Run con IVR agrupado pero sin ivr_grouping_bins_raw en config."
        )
    use_ivr_coarse_fine = bool(config.get("use_ivr_coarse_fine_effective", False))
    ivr_coarse_bins = parse_bins_from_config(config.get("ivr_coarse_bins_effective", []))
    ivr_score_targets = parse_ivr_score_targets(config.get("ivr_score_targets", []))
    ivr_score_hpi_gate = bool(config.get("ivr_score_hpi_gate_at_inference", True))
    ivr_app_threshold = float(config.get("ivr_app_threshold", 0.5))
    img_size = args.img_size if args.img_size > 0 else int(config.get("img_size", 224))
    checkpoint_path = run_dir / args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No existe checkpoint: {checkpoint_path}")

    device = resolve_device(args.device)
    output_dim = output_dim_for_target(
        head_type, target_name, num_classes_hpi, num_classes_ivr
    )
    model = build_model(
        model_name,
        output_dim=output_dim,
        head_type=head_type,
        num_classes_hpi=num_classes_hpi,
        target=target_name,
        ivr_conditioning_source=str(config.get("ivr_conditioning_source", "hpi_probs")),
        ivr_conditioned_hidden_dim=int(config.get("ivr_conditioned_hidden_dim", 128)),
        ivr_conditioned_dropout=float(config.get("ivr_conditioned_dropout", 0.1)),
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    true_hpi = args.true_hpi if args.true_hpi >= 0 else None
    true_ivr = args.true_ivr if args.true_ivr >= 0 else None
    if true_ivr is not None and ivr_grouping_bins_raw:
        true_ivr = remap_raw_ivr_label(true_ivr, ivr_grouping_bins_raw, field="true_ivr")
    records = save_prediction_heatmaps(
        model=model,
        model_name=model_name,
        head_type=head_type,
        image_path=args.image_path,
        photo_code=args.photo_cod,
        target=target_name,
        img_size=img_size,
        device=device,
        output_dir=output_dir,
        num_classes_hpi=num_classes_hpi,
        num_classes_ivr=num_classes_ivr,
        use_ivr_coarse_fine=use_ivr_coarse_fine,
        ivr_coarse_bins=ivr_coarse_bins,
        ivr_score_targets=ivr_score_targets,
        ivr_score_hpi_gate=ivr_score_hpi_gate,
        ivr_app_threshold=ivr_app_threshold,
        true_hpi=true_hpi,
        true_ivr=true_ivr,
    )

    print(f"Heatmaps guardados en: {output_dir}")
    for record in records:
        print(
            f"- {record['task']}: pred={record['pred_label']} "
            f"true={record['true_label']} -> {record['output_path']}"
        )


if __name__ == "__main__":
    main()
