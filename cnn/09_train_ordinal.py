#!/usr/bin/env python3
"""Entrenamiento CNN con clasificacion ordinal (CORAL) para HPI/IVR."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import (
    ConvNeXt_Small_Weights,
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    convnext_small,
    convnext_tiny,
    efficientnet_b0,
    resnet18,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CNN ordinal classifier for HPI/IVR"
    )
    parser.add_argument("--train-csv", default="cnn/splits/train.csv")
    parser.add_argument("--val-csv", default="cnn/splits/val.csv")
    parser.add_argument("--out-dir", default="cnn/runs/baseline")
    parser.add_argument(
        "--model",
        choices=["resnet18", "efficientnet_b0", "convnext_small", "convnext_tiny"],
        default="resnet18",
    )
    parser.add_argument("--target", choices=["both", "hpi", "ivr"], default="both")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--loss", choices=["ordinal_bce", "mae", "huber", "mse"], default="ordinal_bce"
    )
    parser.add_argument(
        "--both-loss-weight-hpi",
        type=float,
        default=0.4,
        help="Peso de la loss de HPI cuando target=both (se normaliza con IVR).",
    )
    parser.add_argument(
        "--both-loss-weight-ivr",
        type=float,
        default=0.6,
        help="Peso de la loss de IVR cuando target=both (se normaliza con HPI).",
    )
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument(
        "--ivr-distance-loss",
        choices=["none", "huber", "mse"],
        default="none",
        help="Compat: mantenido para no romper Makefile (no usado en esta version principal).",
    )
    parser.add_argument(
        "--ivr-distance-weight",
        type=float,
        default=0.0,
        help="Compat: mantenido para no romper Makefile (no usado en esta version principal).",
    )
    parser.add_argument(
        "--ivr-distance-delta",
        type=float,
        default=1.0,
        help="Compat: mantenido para no romper Makefile (no usado en esta version principal).",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument(
        "--use-weighted-sampler",
        action="store_true",
        help="Activa WeightedRandomSampler para balancear clases en train.",
    )
    parser.add_argument(
        "--sampler-target",
        choices=["auto", "hpi", "ivr"],
        default="auto",
        help="Variable usada para calcular pesos del sampler (auto: ivr en both, si no target actual).",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Numero de epocas sin mejora para parar (0 = desactivado).",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Mejora minima de mae_mean para resetear paciencia.",
    )
    parser.add_argument(
        "--plot-metrics-csv",
        default="",
        help="Solo genera grafica desde un metrics.csv y termina",
    )
    parser.add_argument(
        "--plot-output", default="", help="Salida PNG para --plot-metrics-csv"
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se pidio --device cuda pero CUDA no esta disponible.")
    return torch.device(device_arg)


def parse_int_label(raw: str, field: str) -> int:
    value = float(raw)
    rounded = int(round(value))
    if abs(value - rounded) > 1e-6:
        raise ValueError(f"{field} debe ser entero, recibido: {raw}")
    if rounded < 0:
        raise ValueError(f"{field} no puede ser negativo, recibido: {raw}")
    return rounded


class KelpOrdinalDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        transform: transforms.Compose,
        target: str,
        max_samples: int = 0,
    ):
        self.transform = transform
        self.target = target
        self.samples: list[tuple[str, int, int]] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(
                    (
                        row["image_path"],
                        parse_int_label(row["hpi"], "hpi"),
                        parse_int_label(row["ivr"], "ivr"),
                    )
                )

        if max_samples > 0:
            self.samples = self.samples[:max_samples]

        if not self.samples:
            raise ValueError(f"No hay muestras en {csv_path}")

        self.max_hpi = max(s[1] for s in self.samples)
        self.max_ivr = max(s[2] for s in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, hpi, ivr = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        x = self.transform(image)
        if self.target == "both":
            y = torch.tensor([hpi, ivr], dtype=torch.int64)
        elif self.target == "hpi":
            y = torch.tensor([hpi], dtype=torch.int64)
        else:
            y = torch.tensor([ivr], dtype=torch.int64)
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
    if name == "convnext_small":
        weights = ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        model = convnext_small(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, output_dim)
        return model
    raise ValueError(f"Modelo no soportado: {name}")


def ordinal_levels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    thresholds = torch.arange(num_classes - 1, device=labels.device)
    return (labels.unsqueeze(1) > thresholds.unsqueeze(0)).to(torch.float32)


def decode_ordinal_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs > 0.5).sum(dim=1)


def resolve_sampler_target(sampler_target: str, training_target: str) -> str:
    if sampler_target != "auto":
        return sampler_target
    if training_target == "both":
        return "ivr"
    return training_target


def compute_sample_weights(dataset: KelpOrdinalDataset, sampler_target: str) -> torch.DoubleTensor:
    counts: dict[int, int] = {}
    for _, hpi, ivr in dataset.samples:
        label = ivr if sampler_target == "ivr" else hpi
        counts[label] = counts.get(label, 0) + 1

    weights: list[float] = []
    for _, hpi, ivr in dataset.samples:
        label = ivr if sampler_target == "ivr" else hpi
        weights.append(1.0 / counts[label])
    return torch.DoubleTensor(weights)


class OrdinalBCELoss(nn.Module):
    def __init__(
        self,
        target: str,
        num_classes_hpi: int,
        num_classes_ivr: int,
        both_weight_hpi: float = 0.5,
        both_weight_ivr: float = 0.5,
    ):
        super().__init__()
        self.target = target
        self.num_classes_hpi = num_classes_hpi
        self.num_classes_ivr = num_classes_ivr
        self.hpi_dim = num_classes_hpi - 1
        self.ivr_dim = num_classes_ivr - 1
        self.both_weight_hpi = both_weight_hpi
        self.both_weight_ivr = both_weight_ivr

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.target == "both":
            hpi_logits = logits[:, : self.hpi_dim]
            ivr_logits = logits[:, self.hpi_dim : self.hpi_dim + self.ivr_dim]

            hpi_levels = ordinal_levels(labels[:, 0], self.num_classes_hpi)
            ivr_levels = ordinal_levels(labels[:, 1], self.num_classes_ivr)

            loss_hpi = F.binary_cross_entropy_with_logits(hpi_logits, hpi_levels)
            loss_ivr = F.binary_cross_entropy_with_logits(ivr_logits, ivr_levels)
            return (self.both_weight_hpi * loss_hpi) + (self.both_weight_ivr * loss_ivr)

        if self.target == "hpi":
            levels = ordinal_levels(labels[:, 0], self.num_classes_hpi)
            return F.binary_cross_entropy_with_logits(logits, levels)

        levels = ordinal_levels(labels[:, 0], self.num_classes_ivr)
        return F.binary_cross_entropy_with_logits(logits, levels)


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    mae_hpi: Optional[float]
    mae_ivr: Optional[float]
    mae_mean: float


def decode_predictions(
    logits: torch.Tensor,
    target: str,
    num_classes_hpi: int,
    num_classes_ivr: int,
) -> torch.Tensor:
    if target == "both":
        hpi_dim = num_classes_hpi - 1
        hpi_pred = decode_ordinal_logits(logits[:, :hpi_dim])
        ivr_pred = decode_ordinal_logits(logits[:, hpi_dim:])
        return torch.stack([hpi_pred, ivr_pred], dim=1).to(torch.float32)

    pred = decode_ordinal_logits(logits)
    return pred.unsqueeze(1).to(torch.float32)


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

        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            logits = model(x)
            loss = criterion(logits, y)

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
    num_classes_hpi: int,
    num_classes_ivr: int,
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
            logits = model(x)
            if out_dim is None:
                out_dim = logits.shape[1]
            loss = criterion(logits, y)

            pred = decode_predictions(logits, target, num_classes_hpi, num_classes_ivr)
            abs_err = (pred - y.to(torch.float32)).abs()
            bs = y.size(0)
            total_loss += float(loss.item()) * bs
            total_abs_col0 += float(abs_err[:, 0].sum().item())
            if abs_err.shape[1] > 1:
                total_abs_col1 += float(abs_err[:, 1].sum().item())
            n += bs

    val_loss = total_loss / max(n, 1)
    if out_dim is None:
        raise RuntimeError("Validacion sin batches.")

    if target == "both":
        mae_hpi = total_abs_col0 / max(n, 1)
        mae_ivr = total_abs_col1 / max(n, 1)
        mae_mean = (mae_hpi + mae_ivr) / 2.0
    elif target == "hpi":
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
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "mae_hpi",
                "mae_ivr",
                "mae_mean",
            ],
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
        print("Aviso: no se pudo generar grafica (falta matplotlib).")
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
    axes[0].set_title("Ordinal BCE Loss")
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
    axes[1].set_title("MAE (clases)")
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


def infer_num_classes(
    train_ds: KelpOrdinalDataset, val_ds: KelpOrdinalDataset
) -> tuple[int, int]:
    num_classes_hpi = max(train_ds.max_hpi, val_ds.max_hpi) + 1
    num_classes_ivr = max(train_ds.max_ivr, val_ds.max_ivr) + 1
    if num_classes_hpi < 2:
        raise ValueError("HPI necesita al menos 2 clases para clasificacion ordinal.")
    if num_classes_ivr < 2:
        raise ValueError("IVR necesita al menos 2 clases para clasificacion ordinal.")
    return num_classes_hpi, num_classes_ivr


def output_dim_for_target(
    target: str, num_classes_hpi: int, num_classes_ivr: int
) -> int:
    if target == "both":
        return (num_classes_hpi - 1) + (num_classes_ivr - 1)
    if target == "hpi":
        return num_classes_hpi - 1
    return num_classes_ivr - 1


def main() -> None:
    args = parse_args()
    if args.plot_metrics_csv:
        metrics_path = Path(args.plot_metrics_csv)
        plot_path = (
            Path(args.plot_output)
            if args.plot_output
            else metrics_path.parent / "training_curves.png"
        )
        metrics = load_metrics_csv(metrics_path)
        if save_training_plot(metrics, plot_path, target=args.target):
            print(f"Grafica de entrenamiento guardada en: {plot_path}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)
    amp_enabled = args.amp and device.type == "cuda"

    if args.loss != "ordinal_bce":
        print(
            f"Aviso: --loss={args.loss} se ignora. Esta version usa clasificacion ordinal (ordinal_bce)."
        )

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

    train_ds = KelpOrdinalDataset(
        args.train_csv,
        train_tfms,
        target=args.target,
        max_samples=args.max_train_samples,
    )
    val_ds = KelpOrdinalDataset(
        args.val_csv, val_tfms, target=args.target, max_samples=args.max_val_samples
    )

    num_classes_hpi, num_classes_ivr = infer_num_classes(train_ds, val_ds)
    output_dim = output_dim_for_target(args.target, num_classes_hpi, num_classes_ivr)
    if args.target == "both":
        raw_hpi = float(args.both_loss_weight_hpi)
        raw_ivr = float(args.both_loss_weight_ivr)
        if raw_hpi < 0 or raw_ivr < 0:
            raise ValueError("Los pesos de loss para both deben ser >= 0.")
        denom = raw_hpi + raw_ivr
        if denom <= 0:
            raise ValueError("La suma de pesos de loss para both debe ser > 0.")
        both_weight_hpi = raw_hpi / denom
        both_weight_ivr = raw_ivr / denom
    else:
        both_weight_hpi = 1.0
        both_weight_ivr = 0.0

    sampler_target_effective = ""
    train_sampler: Optional[WeightedRandomSampler] = None
    if args.use_weighted_sampler:
        sampler_target_effective = resolve_sampler_target(args.sampler_target, args.target)
        sample_weights = compute_sample_weights(train_ds, sampler_target=sampler_target_effective)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
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

    model = build_model(args.model, args.pretrained, output_dim=output_dim).to(device)
    criterion = OrdinalBCELoss(
        args.target,
        num_classes_hpi=num_classes_hpi,
        num_classes_ivr=num_classes_ivr,
        both_weight_hpi=both_weight_hpi,
        both_weight_ivr=both_weight_ivr,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    config_data = vars(args).copy()
    config_data.update(
        {
            "head_type": "ordinal_coral",
            "num_classes_hpi": num_classes_hpi,
            "num_classes_ivr": num_classes_ivr,
            "output_dim": output_dim,
            "both_loss_weight_hpi_effective": both_weight_hpi,
            "both_loss_weight_ivr_effective": both_weight_ivr,
            "weighted_sampler_enabled": bool(args.use_weighted_sampler),
            "weighted_sampler_target_effective": sampler_target_effective,
        }
    )
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    history: list[EpochMetrics] = []
    best_mae = float("inf")
    epochs_without_improvement = 0

    print(
        f"Entrenando model={args.model} target={args.target} device={device} "
        f"train={len(train_ds)} val={len(val_ds)} epochs={args.epochs} batch={args.batch_size} "
        f"classes_hpi={num_classes_hpi} classes_ivr={num_classes_ivr}"
    )
    if args.target == "both":
        print(
            f"Pesos de loss both: hpi={both_weight_hpi:.3f} ivr={both_weight_ivr:.3f}"
        )
    if args.use_weighted_sampler:
        print(
            f"WeightedRandomSampler activo (target={sampler_target_effective})"
        )

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch_train(
            model, train_loader, criterion, optimizer, device, scaler, amp_enabled
        )
        val_loss, mae_hpi, mae_ivr, mae_mean = run_epoch_val(
            model,
            val_loader,
            criterion,
            device,
            args.target,
            num_classes_hpi,
            num_classes_ivr,
        )

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

        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "head_type": "ordinal_coral",
            "target": args.target,
            "num_classes_hpi": num_classes_hpi,
            "num_classes_ivr": num_classes_ivr,
            "output_dim": output_dim,
        }
        torch.save(ckpt_payload, out_dir / "last.pt")
        improvement = best_mae - mae_mean
        if improvement > args.early_stopping_min_delta:
            best_mae = mae_mean
            epochs_without_improvement = 0
            torch.save(ckpt_payload, out_dir / "best.pt")
        else:
            epochs_without_improvement += 1

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

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                "Early stopping activado: "
                f"sin mejora > {args.early_stopping_min_delta:.6f} en "
                f"{epochs_without_improvement} epocas consecutivas."
            )
            break

    plot_path = out_dir / "training_curves.png"
    if save_training_plot(history, plot_path, target=args.target):
        print(f"Grafica de entrenamiento guardada en: {plot_path}")

    print(f"Entrenamiento finalizado. Mejor mae_mean={best_mae:.4f}. Salida: {out_dir}")


if __name__ == "__main__":
    main()
