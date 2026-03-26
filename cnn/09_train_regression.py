#!/usr/bin/env python3
"""Entrenamiento base de CNN para regresión multisalida (HPI, IVR)."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    convnext_tiny,
    efficientnet_b0,
    resnet18,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN regressor for HPI/IVR")
    parser.add_argument("--train-csv", default="cnn/splits/train.csv")
    parser.add_argument("--val-csv", default="cnn/splits/val.csv")
    parser.add_argument("--out-dir", default="cnn/runs/baseline")
    parser.add_argument("--model", choices=["resnet18", "efficientnet_b0", "convnext_tiny"], default="resnet18")
    parser.add_argument("--target", choices=["both", "hpi", "ivr"], default="both")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["mae", "huber", "mse"], default="mae")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--plot-metrics-csv", default="", help="Solo genera gráfica desde un metrics.csv y termina")
    parser.add_argument("--plot-output", default="", help="Salida PNG para --plot-metrics-csv")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se pidió --device cuda pero CUDA no está disponible.")
    return torch.device(device_arg)


class KelpRegressionDataset(Dataset):
    def __init__(self, csv_path: str, transform: transforms.Compose, target: str, max_samples: int = 0):
        self.transform = transform
        self.target = target
        self.samples: list[tuple[str, float, float]] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["image_path"], float(row["hpi"]), float(row["ivr"])))

        if max_samples > 0:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, hpi, ivr = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        x = self.transform(image)
        if self.target == "both":
            y = torch.tensor([hpi, ivr], dtype=torch.float32)
        elif self.target == "hpi":
            y = torch.tensor([hpi], dtype=torch.float32)
        else:
            y = torch.tensor([ivr], dtype=torch.float32)
        return x, y


def build_model(name: str, pretrained: bool, output_dim: int) -> nn.Module:
    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, output_dim)
        return model
    if name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, output_dim)
        return model
    if name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, output_dim)
        return model
    raise ValueError(f"Modelo no soportado: {name}")


def build_loss(name: str, huber_delta: float) -> nn.Module:
    if name == "mae":
        return nn.L1Loss()
    if name == "huber":
        return nn.HuberLoss(delta=huber_delta)
    if name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Loss no soportada: {name}")


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    mae_hpi: Optional[float]
    mae_ivr: Optional[float]
    mae_mean: float


def run_epoch_train(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            pred = model(x)
            loss = criterion(pred, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = y.size(0)
        total_loss += float(loss.detach().item()) * bs
        n += bs

    return total_loss / max(n, 1)


def run_epoch_val(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target: str,
) -> tuple[float, Optional[float], Optional[float], float]:
    model.eval()
    total_loss = 0.0
    total_abs_col0 = 0.0
    total_abs_col1 = 0.0
    out_dim = None
    n = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            if out_dim is None:
                out_dim = pred.shape[1]
            loss = criterion(pred, y)

            abs_err = (pred - y).abs()
            bs = y.size(0)
            total_loss += float(loss.item()) * bs
            total_abs_col0 += float(abs_err[:, 0].sum().item())
            if abs_err.shape[1] > 1:
                total_abs_col1 += float(abs_err[:, 1].sum().item())
            n += bs

    val_loss = total_loss / max(n, 1)
    if out_dim == 2:
        mae_hpi = total_abs_col0 / max(n, 1)
        mae_ivr = total_abs_col1 / max(n, 1)
        mae_mean = (mae_hpi + mae_ivr) / 2.0
    else:
        if target == "hpi":
            mae_hpi = total_abs_col0 / max(n, 1)
            mae_ivr = None
            mae_mean = mae_hpi
        else:
            mae_hpi = None
            mae_ivr = total_abs_col0 / max(n, 1)
            mae_mean = mae_ivr
    return val_loss, mae_hpi, mae_ivr, mae_mean


def save_metrics_csv(path: Path, metrics: Iterable[EpochMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "mae_hpi", "mae_ivr", "mae_mean"],
        )
        writer.writeheader()
        for m in metrics:
            row = asdict(m)
            if row["mae_hpi"] is None:
                row["mae_hpi"] = ""
            if row["mae_ivr"] is None:
                row["mae_ivr"] = ""
            writer.writerow(row)


def save_training_plot(metrics: list[EpochMetrics], output: Path, target: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Aviso: no se pudo generar gráfica (falta matplotlib).")
        return False

    if not metrics:
        return False

    epochs = [m.epoch for m in metrics]
    train_loss = [m.train_loss for m in metrics]
    val_loss = [m.val_loss for m in metrics]
    mae_hpi = [m.mae_hpi for m in metrics if m.mae_hpi is not None]
    mae_ivr = [m.mae_ivr for m in metrics if m.mae_ivr is not None]
    mae_mean = [m.mae_mean for m in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, marker="o", label="train_loss")
    axes[0].plot(epochs, val_loss, marker="o", label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if target in {"both", "hpi"} and len(mae_hpi) == len(epochs):
        axes[1].plot(epochs, mae_hpi, marker="o", label="mae_hpi")
    if target == "both" and len(mae_ivr) == len(epochs):
        axes[1].plot(epochs, mae_ivr, marker="o", label="mae_ivr")
    if target == "ivr" and len(mae_ivr) == len(epochs):
        axes[1].plot(epochs, mae_ivr, marker="o", label="mae_ivr")
    axes[1].plot(epochs, mae_mean, marker="o", label="mae_mean")
    axes[1].set_title("MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    return True


def load_metrics_csv(path: Path) -> list[EpochMetrics]:
    def parse_optional_float(v: str) -> Optional[float]:
        if v is None:
            return None
        s = str(v).strip()
        if s == "":
            return None
        return float(s)

    metrics: list[EpochMetrics] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics.append(
                EpochMetrics(
                    epoch=int(float(row["epoch"])),
                    train_loss=float(row["train_loss"]),
                    val_loss=float(row["val_loss"]),
                    mae_hpi=parse_optional_float(row["mae_hpi"]),
                    mae_ivr=parse_optional_float(row["mae_ivr"]),
                    mae_mean=float(row["mae_mean"]),
                )
            )
    return metrics


def main() -> None:
    args = parse_args()
    if args.plot_metrics_csv:
        metrics_path = Path(args.plot_metrics_csv)
        plot_path = Path(args.plot_output) if args.plot_output else metrics_path.parent / "training_curves.png"
        metrics = load_metrics_csv(metrics_path)
        if save_training_plot(metrics, plot_path, target=args.target):
            print(f"Gráfica de entrenamiento guardada en: {plot_path}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)
    amp_enabled = args.amp and device.type == "cuda"

    train_tfms = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = KelpRegressionDataset(args.train_csv, train_tfms, target=args.target, max_samples=args.max_train_samples)
    val_ds = KelpRegressionDataset(args.val_csv, val_tfms, target=args.target, max_samples=args.max_val_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    output_dim = 2 if args.target == "both" else 1
    model = build_model(args.model, args.pretrained, output_dim=output_dim).to(device)
    criterion = build_loss(args.loss, args.huber_delta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    history: list[EpochMetrics] = []
    best_mae = float("inf")

    print(
        f"Entrenando model={args.model} device={device} train={len(train_ds)} val={len(val_ds)} "
        f"epochs={args.epochs} batch={args.batch_size}"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch_train(model, train_loader, criterion, optimizer, device, scaler, amp_enabled)
        val_loss, mae_hpi, mae_ivr, mae_mean = run_epoch_val(model, val_loader, criterion, device, args.target)

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            mae_hpi=mae_hpi,
            mae_ivr=mae_ivr,
            mae_mean=mae_mean,
        )
        history.append(metrics)
        save_metrics_csv(out_dir / "metrics.csv", history)

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, out_dir / "last.pt")
        if mae_mean < best_mae:
            best_mae = mae_mean
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, out_dir / "best.pt")

        if args.target == "both":
            print(
                f"[{epoch:03d}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"mae_hpi={mae_hpi:.4f} mae_ivr={mae_ivr:.4f} mae_mean={mae_mean:.4f}"
            )
        elif args.target == "hpi":
            print(
                f"[{epoch:03d}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"mae_hpi={mae_hpi:.4f}"
            )
        else:
            print(
                f"[{epoch:03d}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"mae_ivr={mae_ivr:.4f}"
            )

    plot_path = out_dir / "training_curves.png"
    if save_training_plot(history, plot_path, target=args.target):
        print(f"Gráfica de entrenamiento guardada en: {plot_path}")

    print(f"Entrenamiento finalizado. Mejor mae_mean={best_mae:.4f}. Salida: {out_dir}")


if __name__ == "__main__":
    main()
