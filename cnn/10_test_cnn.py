#!/usr/bin/env python3
"""Evaluacion de una CNN de regresion sobre split de test."""

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


class TestDataset(Dataset):
    def __init__(self, csv_path: str, transform: transforms.Compose):
        self.transform = transform
        self.rows: list[tuple[str, str, float, float]] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(
                    (
                        row["photo_cod"],
                        row["image_path"],
                        float(row["hpi"]),
                        float(row["ivr"]),
                    )
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        photo_cod, image_path, hpi, ivr = self.rows[idx]
        image = Image.open(image_path).convert("RGB")
        x = self.transform(image)
        y = torch.tensor([hpi, ivr], dtype=torch.float32)
        return x, y, photo_cod, image_path


def build_model(name: str) -> nn.Module:
    if name == "resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model
    if name == "efficientnet_b0":
        model = efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        return model
    if name == "convnext_tiny":
        model = convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
        return model
    raise ValueError(f"Modelo no soportado: {name}")


def save_predictions_csv(
    out_path: Path,
    photo_codes: list[str],
    image_paths: list[str],
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "photo_cod",
                "image_path",
                "true_hpi",
                "pred_hpi",
                "abs_err_hpi",
                "true_ivr",
                "pred_ivr",
                "abs_err_ivr",
            ],
        )
        writer.writeheader()
        for i, code in enumerate(photo_codes):
            true_hpi = float(y_true[i, 0].item())
            pred_hpi = float(y_pred[i, 0].item())
            true_ivr = float(y_true[i, 1].item())
            pred_ivr = float(y_pred[i, 1].item())
            writer.writerow(
                {
                    "photo_cod": code,
                    "image_path": image_paths[i],
                    "true_hpi": true_hpi,
                    "pred_hpi": pred_hpi,
                    "abs_err_hpi": abs(pred_hpi - true_hpi),
                    "true_ivr": true_ivr,
                    "pred_ivr": pred_ivr,
                    "abs_err_ivr": abs(pred_ivr - true_ivr),
                }
            )


def save_plots(out_dir: Path, y_true: torch.Tensor, y_pred: torch.Tensor) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    # 5) Plots - real vs pred
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for j, label in enumerate(["HPI", "IVR"]):
        t = y_true[:, j].cpu().numpy()
        p = y_pred[:, j].cpu().numpy()
        lo = min(t.min(), p.min())
        hi = max(t.max(), p.max())
        axes[j].scatter(t, p, s=12, alpha=0.6)
        axes[j].plot([lo, hi], [lo, hi], linestyle="--")
        axes[j].set_title(f"5) Real vs Pred - {label}")
        axes[j].set_xlabel("Real")
        axes[j].set_ylabel("Pred")
        axes[j].grid(True, alpha=0.3)
    fig.tight_layout()
    p1 = out_dir / "5_real_vs_pred.png"
    fig.savefig(p1, dpi=150)
    outputs.append(p1.name)

    # 5) Plots - residual histogram
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    residuals = y_pred - y_true
    for j, label in enumerate(["HPI", "IVR"]):
        r = residuals[:, j].cpu().numpy()
        axes2[j].hist(r, bins=30, alpha=0.8)
        axes2[j].set_title(f"5) Residuales - {label}")
        axes2[j].set_xlabel("Pred - Real")
        axes2[j].set_ylabel("Frecuencia")
        axes2[j].grid(True, alpha=0.3)
    fig2.tight_layout()
    p2 = out_dir / "5_residuals_hist.png"
    fig2.savefig(p2, dpi=150)
    outputs.append(p2.name)

    return outputs


def main() -> None:
    args = parse_args()

    run_dir = resolve_run_dir(Path(args.run_dir))
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "test_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.json").open(encoding="utf-8") as f:
        config: dict[str, Any] = json.load(f)

    model_name = args.model or str(config.get("model", "resnet18"))
    img_size = args.img_size if args.img_size > 0 else int(config.get("img_size", 224))
    checkpoint_path = run_dir / args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No existe checkpoint: {checkpoint_path}")

    device = resolve_device(args.device)

    test_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_ds = TestDataset(args.test_csv, test_tfms)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(model_name).to(device)
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
            pred = model(x).cpu()
            y_true_parts.append(y)
            y_pred_parts.append(pred)
            photo_codes.extend(batch_codes)
            image_paths.extend(batch_paths)

    y_true = torch.cat(y_true_parts, dim=0)
    y_pred = torch.cat(y_pred_parts, dim=0)
    abs_err = (y_pred - y_true).abs()
    sq_err = (y_pred - y_true).pow(2)

    # 1) MAE
    mae_hpi = float(abs_err[:, 0].mean().item())
    mae_ivr = float(abs_err[:, 1].mean().item())
    mae_mean = (mae_hpi + mae_ivr) / 2.0

    # 2) RMSE
    rmse_hpi = float(torch.sqrt(sq_err[:, 0].mean()).item())
    rmse_ivr = float(torch.sqrt(sq_err[:, 1].mean()).item())
    rmse_mean = (rmse_hpi + rmse_ivr) / 2.0

    # 4) Accuracy por tolerancia
    def tol_metrics(tol: float) -> dict[str, float]:
        ok_hpi = (abs_err[:, 0] <= tol).float().mean().item()
        ok_ivr = (abs_err[:, 1] <= tol).float().mean().item()
        ok_both = ((abs_err[:, 0] <= tol) & (abs_err[:, 1] <= tol)).float().mean().item()
        ok_mean = (abs_err <= tol).float().mean().item()
        return {
            "tol": tol,
            "acc_hpi": float(ok_hpi),
            "acc_ivr": float(ok_ivr),
            "acc_both": float(ok_both),
            "acc_mean": float(ok_mean),
        }

    tol_05 = tol_metrics(0.5)
    tol_10 = tol_metrics(1.0)

    # 5) Plots
    plot_files = save_plots(output_dir, y_true, y_pred)

    predictions_path = output_dir / "predictions_test.csv"
    save_predictions_csv(predictions_path, photo_codes, image_paths, y_true, y_pred)

    metrics = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "model": model_name,
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
        "4_tolerance_accuracy": {
            "tol_0_5": tol_05,
            "tol_1_0": tol_10,
        },
        "5_plots": {
            "generated": len(plot_files) > 0,
            "files": plot_files,
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
    print(f"1) MAE: mae_hpi={mae_hpi:.4f} mae_ivr={mae_ivr:.4f} mae_mean={mae_mean:.4f}")
    print(f"2) RMSE: rmse_hpi={rmse_hpi:.4f} rmse_ivr={rmse_ivr:.4f} rmse_mean={rmse_mean:.4f}")
    print(
        "4) Acc tolerancia: "
        f"tol=0.5 acc_hpi={tol_05['acc_hpi']:.4f} acc_ivr={tol_05['acc_ivr']:.4f} acc_both={tol_05['acc_both']:.4f}; "
        f"tol=1.0 acc_hpi={tol_10['acc_hpi']:.4f} acc_ivr={tol_10['acc_ivr']:.4f} acc_both={tol_10['acc_both']:.4f}"
    )
    if plot_files:
        print(f"5) Plots: {', '.join(plot_files)}")
    else:
        print("5) Plots: no generados (falta matplotlib).")


if __name__ == "__main__":
    main()
