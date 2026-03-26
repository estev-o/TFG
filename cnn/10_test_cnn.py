#!/usr/bin/env python3
"""Evaluacion de una CNN ordinal sobre split de test."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    convnext_tiny,
    efficientnet_b0,
    resnet18,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalua una run CNN en split de test")
    parser.add_argument("--run-dir", required=True, help="Directorio de run (el que contiene config.json)")
    parser.add_argument("--test-csv", default="cnn/splits/test.csv")
    parser.add_argument("--checkpoint", default="best.pt", help="Nombre de checkpoint dentro de run-dir")
    parser.add_argument("--model", default="", choices=["", "resnet18", "efficientnet_b0", "convnext_tiny"])
    parser.add_argument("--target", default="auto", choices=["auto", "both", "hpi", "ivr"])
    parser.add_argument("--img-size", type=int, default=0, help="0 para usar el de config.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directorio de salida. Si vacio: <run-dir>/test_eval",
    )
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


def parse_int_label(raw: str, field: str) -> int:
    value = float(raw)
    rounded = int(round(value))
    if abs(value - rounded) > 1e-6:
        raise ValueError(f"{field} debe ser entero, recibido: {raw}")
    if rounded < 0:
        raise ValueError(f"{field} no puede ser negativo, recibido: {raw}")
    return rounded


class TestDataset(Dataset):
    def __init__(self, csv_path: str, transform: transforms.Compose, target: str):
        self.transform = transform
        self.target = target
        self.rows: list[tuple[str, str, int, int]] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(
                    (
                        row["photo_cod"],
                        row["image_path"],
                        parse_int_label(row["hpi"], "hpi"),
                        parse_int_label(row["ivr"], "ivr"),
                    )
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        photo_cod, image_path, hpi, ivr = self.rows[idx]
        image = Image.open(image_path).convert("RGB")
        x = self.transform(image)
        if self.target == "both":
            y = torch.tensor([hpi, ivr], dtype=torch.int64)
        elif self.target == "hpi":
            y = torch.tensor([hpi], dtype=torch.int64)
        else:
            y = torch.tensor([ivr], dtype=torch.int64)
        return x, y, photo_cod, image_path


def output_dim_for_target(target: str, num_classes_hpi: int, num_classes_ivr: int) -> int:
    if target == "both":
        return (num_classes_hpi - 1) + (num_classes_ivr - 1)
    if target == "hpi":
        return num_classes_hpi - 1
    return num_classes_ivr - 1


def build_model(name: str, output_dim: int) -> nn.Module:
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
    raise ValueError(f"Modelo no soportado: {name}")


def decode_ordinal_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs > 0.5).sum(dim=1)


def decode_predictions(
    logits: torch.Tensor,
    target: str,
    num_classes_hpi: int,
    num_classes_ivr: int,
) -> torch.Tensor:
    if target == "both":
        hpi_dim = num_classes_hpi - 1
        pred_hpi = decode_ordinal_logits(logits[:, :hpi_dim])
        pred_ivr = decode_ordinal_logits(logits[:, hpi_dim:])
        return torch.stack([pred_hpi, pred_ivr], dim=1).to(torch.int64)

    pred = decode_ordinal_logits(logits)
    return pred.unsqueeze(1).to(torch.int64)


def save_predictions_csv(
    out_path: Path,
    photo_codes: list[str],
    image_paths: list[str],
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    target: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if target == "both":
        fieldnames = [
            "photo_cod",
            "image_path",
            "true_hpi",
            "pred_hpi",
            "abs_err_hpi",
            "true_ivr",
            "pred_ivr",
            "abs_err_ivr",
        ]
    elif target == "hpi":
        fieldnames = ["photo_cod", "image_path", "true_hpi", "pred_hpi", "abs_err_hpi"]
    else:
        fieldnames = ["photo_cod", "image_path", "true_ivr", "pred_ivr", "abs_err_ivr"]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, code in enumerate(photo_codes):
            row: dict[str, int | str] = {
                "photo_cod": code,
                "image_path": image_paths[i],
            }
            if target in {"both", "hpi"}:
                true_hpi = int(y_true[i, 0].item())
                pred_hpi = int(y_pred[i, 0].item())
                row["true_hpi"] = true_hpi
                row["pred_hpi"] = pred_hpi
                row["abs_err_hpi"] = abs(pred_hpi - true_hpi)
            if target == "both":
                true_ivr = int(y_true[i, 1].item())
                pred_ivr = int(y_pred[i, 1].item())
                row["true_ivr"] = true_ivr
                row["pred_ivr"] = pred_ivr
                row["abs_err_ivr"] = abs(pred_ivr - true_ivr)
            if target == "ivr":
                true_ivr = int(y_true[i, 0].item())
                pred_ivr = int(y_pred[i, 0].item())
                row["true_ivr"] = true_ivr
                row["pred_ivr"] = pred_ivr
                row["abs_err_ivr"] = abs(pred_ivr - true_ivr)
            writer.writerow(row)


def confusion_matrix_counts(
    y_true_int: torch.Tensor, y_pred_int: torch.Tensor, min_label: int, max_label: int
) -> torch.Tensor:
    n_classes = max_label - min_label + 1
    cm = torch.zeros((n_classes, n_classes), dtype=torch.int64)
    true_idx = (y_true_int - min_label).to(torch.int64)
    pred_idx = (y_pred_int - min_label).to(torch.int64)
    valid = (
        (true_idx >= 0)
        & (true_idx < n_classes)
        & (pred_idx >= 0)
        & (pred_idx < n_classes)
    )
    true_idx = true_idx[valid]
    pred_idx = pred_idx[valid]
    for t, p in zip(true_idx.tolist(), pred_idx.tolist()):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(
    out_path: Path,
    cm: torch.Tensor,
    labels: list[int],
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm.numpy(), cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([str(x) for x in labels])
    ax.set_yticklabels([str(x) for x in labels])
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title(title)

    max_val = int(cm.max().item()) if cm.numel() else 0
    threshold = max_val / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j].item())
            color = "white" if val > threshold else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)


def save_plots(
    out_dir: Path,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    target: str,
    class_limits: dict[str, tuple[int, int]],
) -> tuple[list[str], dict[str, Any]]:
    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        return [], {}

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    info: dict[str, Any] = {}

    for legacy_name in ("5_real_vs_pred.png", "5_residuals_hist.png"):
        legacy_path = out_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    target_slices: list[tuple[str, int]]
    if target == "both":
        target_slices = [("hpi", 0), ("ivr", 1)]
    elif target == "hpi":
        target_slices = [("hpi", 0)]
    else:
        target_slices = [("ivr", 0)]

    for target_name, idx in target_slices:
        true_vals = y_true[:, idx].to(torch.int64)
        min_label, max_label = class_limits[target_name]
        pred_vals = y_pred[:, idx].to(torch.int64).clamp(min=min_label, max=max_label)

        labels = list(range(min_label, max_label + 1))
        cm = confusion_matrix_counts(true_vals, pred_vals, min_label, max_label)
        out_name = f"5_confusion_{target_name}.png"
        plot_confusion_matrix(
            out_dir / out_name,
            cm,
            labels,
            title=f"5) Confusion Matrix - {target_name.upper()}",
        )
        outputs.append(out_name)
        info[target_name] = {
            "labels": labels,
            "min_label": min_label,
            "max_label": max_label,
            "pred_postprocess": "clipped_to_config_label_range",
            "matrix": cm.tolist(),
        }

    return outputs, info


def main() -> None:
    args = parse_args()

    run_dir = resolve_run_dir(Path(args.run_dir))
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "test_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.json").open(encoding="utf-8") as f:
        config: dict[str, Any] = json.load(f)

    model_name = args.model or str(config.get("model", "resnet18"))
    target_name = str(config.get("target", "both")) if args.target == "auto" else args.target
    if target_name not in {"both", "hpi", "ivr"}:
        raise ValueError(f"Target no soportado: {target_name}")

    head_type = str(config.get("head_type", ""))
    if head_type != "ordinal_coral":
        raise RuntimeError(
            "Esta version de 10_test_cnn.py espera checkpoints ordinales (head_type=ordinal_coral)."
        )

    num_classes_hpi = int(config.get("num_classes_hpi", 0))
    num_classes_ivr = int(config.get("num_classes_ivr", 0))
    if num_classes_hpi < 2 or num_classes_ivr < 2:
        raise ValueError("num_classes_hpi/num_classes_ivr invalidos en config.json")

    img_size = args.img_size if args.img_size > 0 else int(config.get("img_size", 224))
    checkpoint_path = run_dir / args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No existe checkpoint: {checkpoint_path}")

    device = resolve_device(args.device)
    output_dim = output_dim_for_target(target_name, num_classes_hpi, num_classes_ivr)

    test_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_ds = TestDataset(args.test_csv, test_tfms, target=target_name)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(model_name, output_dim=output_dim).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    y_true_parts: list[torch.Tensor] = []
    y_pred_parts: list[torch.Tensor] = []
    photo_codes: list[str] = []
    image_paths: list[str] = []

    with torch.no_grad():
        for x, y, batch_codes, batch_paths in test_loader:
            x = x.to(device, non_blocking=True)
            logits = model(x).cpu()
            pred = decode_predictions(logits, target_name, num_classes_hpi, num_classes_ivr)
            y_true_parts.append(y)
            y_pred_parts.append(pred)
            photo_codes.extend(batch_codes)
            image_paths.extend(batch_paths)

    y_true = torch.cat(y_true_parts, dim=0).to(torch.int64)
    y_pred = torch.cat(y_pred_parts, dim=0).to(torch.int64)

    abs_err_disc = (y_pred - y_true).abs()
    abs_err = abs_err_disc.to(torch.float32)
    sq_err = abs_err.pow(2)

    if target_name == "both":
        mae_hpi = float(abs_err[:, 0].mean().item())
        mae_ivr = float(abs_err[:, 1].mean().item())
        mae_mean = (mae_hpi + mae_ivr) / 2.0

        rmse_hpi = float(torch.sqrt(sq_err[:, 0].mean()).item())
        rmse_ivr = float(torch.sqrt(sq_err[:, 1].mean()).item())
        rmse_mean = (rmse_hpi + rmse_ivr) / 2.0
    elif target_name == "hpi":
        mae_hpi = float(abs_err[:, 0].mean().item())
        mae_ivr = None
        mae_mean = mae_hpi

        rmse_hpi = float(torch.sqrt(sq_err[:, 0].mean()).item())
        rmse_ivr = None
        rmse_mean = rmse_hpi
    else:
        mae_hpi = None
        mae_ivr = float(abs_err[:, 0].mean().item())
        mae_mean = mae_ivr

        rmse_hpi = None
        rmse_ivr = float(torch.sqrt(sq_err[:, 0].mean()).item())
        rmse_mean = rmse_ivr

    def discrete_acc(max_diff: int) -> dict[str, float]:
        if target_name == "both":
            ok_hpi = (abs_err_disc[:, 0] <= max_diff).float().mean().item()
            ok_ivr = (abs_err_disc[:, 1] <= max_diff).float().mean().item()
            ok_both = ((abs_err_disc[:, 0] <= max_diff) & (abs_err_disc[:, 1] <= max_diff)).float().mean().item()
            ok_mean = (abs_err_disc <= max_diff).float().mean().item()
            return {
                "max_diff": max_diff,
                "acc_hpi": float(ok_hpi),
                "acc_ivr": float(ok_ivr),
                "acc_both": float(ok_both),
                "acc_mean": float(ok_mean),
            }
        ok_target = (abs_err_disc[:, 0] <= max_diff).float().mean().item()
        return {
            "max_diff": max_diff,
            f"acc_{target_name}": float(ok_target),
            "acc_mean": float(ok_target),
        }

    acc_exact = discrete_acc(0)
    acc_within_1 = discrete_acc(1)
    acc_within_2 = discrete_acc(2)

    class_limits = {
        "hpi": (0, num_classes_hpi - 1),
        "ivr": (0, num_classes_ivr - 1),
    }
    plot_files, confusion_info = save_plots(output_dir, y_true, y_pred, target=target_name, class_limits=class_limits)

    predictions_path = output_dir / "predictions_test.csv"
    save_predictions_csv(predictions_path, photo_codes, image_paths, y_true, y_pred, target=target_name)

    metrics = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "model": model_name,
        "target": target_name,
        "head_type": head_type,
        "num_classes_hpi": num_classes_hpi,
        "num_classes_ivr": num_classes_ivr,
        "n_test_samples": len(test_ds),
        "1_mae": {
            "mae_hpi": mae_hpi,
            "mae_ivr": mae_ivr,
            "mae_mean": mae_mean,
        },
        "2_rmse": {
            "rmse_hpi": rmse_hpi,
            "rmse_ivr": rmse_ivr,
            "rmse_mean": rmse_mean,
        },
        "3_discrete_accuracy": {
            "exact_match": acc_exact,
            "within_1": acc_within_1,
            "within_2": acc_within_2,
        },
        "5_plots": {
            "generated": len(plot_files) > 0,
            "files": plot_files,
            "confusion_matrix": confusion_info,
        },
        "outputs": {
            "metrics_json": "metrics_test.json",
            "predictions_csv": "predictions_test.csv",
        },
    }

    metrics_path = output_dir / "metrics_test.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Evaluacion completada en: {output_dir}")
    if target_name == "both":
        print(f"1) MAE: mae_hpi={mae_hpi:.4f} mae_ivr={mae_ivr:.4f} mae_mean={mae_mean:.4f}")
        print(f"2) RMSE: rmse_hpi={rmse_hpi:.4f} rmse_ivr={rmse_ivr:.4f} rmse_mean={rmse_mean:.4f}")
        print(
            "3) Acc discreta: "
            f"exact acc_hpi={acc_exact['acc_hpi']:.4f} acc_ivr={acc_exact['acc_ivr']:.4f} acc_both={acc_exact['acc_both']:.4f}; "
            f"within1 acc_hpi={acc_within_1['acc_hpi']:.4f} acc_ivr={acc_within_1['acc_ivr']:.4f} acc_both={acc_within_1['acc_both']:.4f}; "
            f"within2 acc_hpi={acc_within_2['acc_hpi']:.4f} acc_ivr={acc_within_2['acc_ivr']:.4f} acc_both={acc_within_2['acc_both']:.4f}"
        )
    elif target_name == "hpi":
        print(f"1) MAE: mae_hpi={mae_hpi:.4f} mae_mean={mae_mean:.4f}")
        print(f"2) RMSE: rmse_hpi={rmse_hpi:.4f} rmse_mean={rmse_mean:.4f}")
        print(
            "3) Acc discreta: "
            f"exact acc_hpi={acc_exact['acc_hpi']:.4f}; "
            f"within1 acc_hpi={acc_within_1['acc_hpi']:.4f}; "
            f"within2 acc_hpi={acc_within_2['acc_hpi']:.4f}"
        )
    else:
        print(f"1) MAE: mae_ivr={mae_ivr:.4f} mae_mean={mae_mean:.4f}")
        print(f"2) RMSE: rmse_ivr={rmse_ivr:.4f} rmse_mean={rmse_mean:.4f}")
        print(
            "3) Acc discreta: "
            f"exact acc_ivr={acc_exact['acc_ivr']:.4f}; "
            f"within1 acc_ivr={acc_within_1['acc_ivr']:.4f}; "
            f"within2 acc_ivr={acc_within_2['acc_ivr']:.4f}"
        )
    if plot_files:
        print(f"5) Plots (confusion matrix): {', '.join(plot_files)}")
    else:
        print("5) Plots: no generados (falta matplotlib).")


if __name__ == "__main__":
    main()
