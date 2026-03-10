#!/usr/bin/env python3
"""Entrenamiento base de CNN para regresión multisalida (HPI, IVR)."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

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
    def __init__(self, csv_path: str, transform: transforms.Compose, max_samples: int = 0):
        self.transform = transform
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
        y = torch.tensor([hpi, ivr], dtype=torch.float32)
        return x, y


def build_model(name: str, pretrained: bool) -> nn.Module:
    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model
    if name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        return model
    if name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
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
    mae_hpi: float
    mae_ivr: float
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
) -> tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_abs_hpi = 0.0
    total_abs_ivr = 0.0
    n = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            loss = criterion(pred, y)

            abs_err = (pred - y).abs()
            bs = y.size(0)
            total_loss += float(loss.item()) * bs
            total_abs_hpi += float(abs_err[:, 0].sum().item())
            total_abs_ivr += float(abs_err[:, 1].sum().item())
            n += bs

    val_loss = total_loss / max(n, 1)
    mae_hpi = total_abs_hpi / max(n, 1)
    mae_ivr = total_abs_ivr / max(n, 1)
    mae_mean = (mae_hpi + mae_ivr) / 2.0
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
            writer.writerow(asdict(m))


def main() -> None:
    args = parse_args()
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

    train_ds = KelpRegressionDataset(args.train_csv, train_tfms, max_samples=args.max_train_samples)
    val_ds = KelpRegressionDataset(args.val_csv, val_tfms, max_samples=args.max_val_samples)

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

    model = build_model(args.model, args.pretrained).to(device)
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
        val_loss, mae_hpi, mae_ivr, mae_mean = run_epoch_val(model, val_loader, criterion, device)

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

        print(
            f"[{epoch:03d}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"mae_hpi={mae_hpi:.4f} mae_ivr={mae_ivr:.4f} mae_mean={mae_mean:.4f}"
        )

    print(f"Entrenamiento finalizado. Mejor mae_mean={best_mae:.4f}. Salida: {out_dir}")


if __name__ == "__main__":
    main()
