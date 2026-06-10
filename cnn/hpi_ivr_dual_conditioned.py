#!/usr/bin/env python3
"""Modelos CNN con cabeza IVR condicionada por la salida ordinal de HPI."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
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


ConditioningSource = Literal["hpi_probs", "hpi_logits"]


def _build_feature_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        feature_dim = int(model.fc.in_features)
        model.fc = nn.Identity()
        return model, feature_dim
    if name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        feature_dim = int(model.classifier[1].in_features)
        model.classifier[1] = nn.Identity()
        return model, feature_dim
    if name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = convnext_tiny(weights=weights)
        feature_dim = int(model.classifier[2].in_features)
        model.classifier[2] = nn.Identity()
        return model, feature_dim
    if name == "convnext_small":
        weights = ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        model = convnext_small(weights=weights)
        feature_dim = int(model.classifier[2].in_features)
        model.classifier[2] = nn.Identity()
        return model, feature_dim
    raise ValueError(f"Modelo no soportado: {name}")


class HpiCoralIvrDualConditionedModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        num_classes_hpi: int,
        target: str = "both",
        ivr_conditioning_source: ConditioningSource = "hpi_probs",
        ivr_hidden_dim: int = 128,
        ivr_dropout: float = 0.1,
    ):
        super().__init__()
        if target not in {"both", "hpi"}:
            raise ValueError("Esta variante solo soporta target=both o target=hpi.")
        if num_classes_hpi < 2:
            raise ValueError("num_classes_hpi debe ser >= 2.")
        if ivr_hidden_dim < 1:
            raise ValueError("ivr_hidden_dim debe ser >= 1.")
        if not 0.0 <= ivr_dropout < 1.0:
            raise ValueError("ivr_dropout debe estar en [0, 1).")
        if ivr_conditioning_source not in {"hpi_probs", "hpi_logits"}:
            raise ValueError(
                "ivr_conditioning_source debe ser 'hpi_probs' o 'hpi_logits'."
            )

        self.backbone_name = backbone_name
        self.target = target
        self.num_classes_hpi = num_classes_hpi
        self.hpi_dim = num_classes_hpi - 1
        self.ivr_conditioning_source = ivr_conditioning_source
        self.ivr_hidden_dim = ivr_hidden_dim
        self.ivr_dropout = ivr_dropout

        self.backbone, feature_dim = _build_feature_backbone(backbone_name, pretrained)
        self.feature_dim = feature_dim

        self.hpi_head = nn.Linear(feature_dim, self.hpi_dim)
        if self.target == "both":
            cond_dim = feature_dim + self.hpi_dim
            self.ivr_adapter = nn.Sequential(
                nn.Linear(cond_dim, ivr_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=ivr_dropout),
            )
            self.ivr_app_head = nn.Linear(ivr_hidden_dim, 1)
            self.ivr_score_head = nn.Linear(ivr_hidden_dim, 1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def _hpi_condition_tensor(self, hpi_logits: torch.Tensor) -> torch.Tensor:
        if self.ivr_conditioning_source == "hpi_logits":
            return hpi_logits
        return torch.sigmoid(hpi_logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        hpi_logits = self.hpi_head(features)
        if self.target != "both":
            return hpi_logits

        ivr_cond = self._hpi_condition_tensor(hpi_logits)
        ivr_hidden = self.ivr_adapter(torch.cat([features, ivr_cond], dim=1))
        ivr_app_logits = self.ivr_app_head(ivr_hidden)
        ivr_score_logits = self.ivr_score_head(ivr_hidden)
        return torch.cat([hpi_logits, ivr_app_logits, ivr_score_logits], dim=1)


class HpiCoralIvrScoreConditionedModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        num_classes_hpi: int,
        target: str = "both",
        ivr_conditioning_source: ConditioningSource = "hpi_probs",
        ivr_hidden_dim: int = 128,
        ivr_dropout: float = 0.1,
    ):
        super().__init__()
        if target not in {"both", "hpi"}:
            raise ValueError("Esta variante solo soporta target=both o target=hpi.")
        if num_classes_hpi < 2:
            raise ValueError("num_classes_hpi debe ser >= 2.")
        if ivr_hidden_dim < 1:
            raise ValueError("ivr_hidden_dim debe ser >= 1.")
        if not 0.0 <= ivr_dropout < 1.0:
            raise ValueError("ivr_dropout debe estar en [0, 1).")
        if ivr_conditioning_source not in {"hpi_probs", "hpi_logits"}:
            raise ValueError(
                "ivr_conditioning_source debe ser 'hpi_probs' o 'hpi_logits'."
            )

        self.backbone_name = backbone_name
        self.target = target
        self.num_classes_hpi = num_classes_hpi
        self.hpi_dim = num_classes_hpi - 1
        self.ivr_conditioning_source = ivr_conditioning_source
        self.ivr_hidden_dim = ivr_hidden_dim
        self.ivr_dropout = ivr_dropout

        self.backbone, feature_dim = _build_feature_backbone(backbone_name, pretrained)
        self.feature_dim = feature_dim

        self.hpi_head = nn.Linear(feature_dim, self.hpi_dim)
        if self.target == "both":
            cond_dim = feature_dim + self.hpi_dim
            self.ivr_adapter = nn.Sequential(
                nn.Linear(cond_dim, ivr_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=ivr_dropout),
            )
            self.ivr_score_head = nn.Linear(ivr_hidden_dim, 1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def _hpi_condition_tensor(self, hpi_logits: torch.Tensor) -> torch.Tensor:
        if self.ivr_conditioning_source == "hpi_logits":
            return hpi_logits
        return torch.sigmoid(hpi_logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        hpi_logits = self.hpi_head(features)
        if self.target != "both":
            return hpi_logits

        ivr_cond = self._hpi_condition_tensor(hpi_logits)
        ivr_hidden = self.ivr_adapter(torch.cat([features, ivr_cond], dim=1))
        ivr_score_logits = self.ivr_score_head(ivr_hidden)
        return torch.cat([hpi_logits, ivr_score_logits], dim=1)


def build_conditioned_dual_model(
    name: str,
    pretrained: bool,
    num_classes_hpi: int,
    target: str = "both",
    ivr_conditioning_source: ConditioningSource = "hpi_probs",
    ivr_hidden_dim: int = 128,
    ivr_dropout: float = 0.1,
) -> HpiCoralIvrDualConditionedModel:
    return HpiCoralIvrDualConditionedModel(
        backbone_name=name,
        pretrained=pretrained,
        num_classes_hpi=num_classes_hpi,
        target=target,
        ivr_conditioning_source=ivr_conditioning_source,
        ivr_hidden_dim=ivr_hidden_dim,
        ivr_dropout=ivr_dropout,
    )


def build_conditioned_score_model(
    name: str,
    pretrained: bool,
    num_classes_hpi: int,
    target: str = "both",
    ivr_conditioning_source: ConditioningSource = "hpi_probs",
    ivr_hidden_dim: int = 128,
    ivr_dropout: float = 0.1,
) -> HpiCoralIvrScoreConditionedModel:
    return HpiCoralIvrScoreConditionedModel(
        backbone_name=name,
        pretrained=pretrained,
        num_classes_hpi=num_classes_hpi,
        target=target,
        ivr_conditioning_source=ivr_conditioning_source,
        ivr_hidden_dim=ivr_hidden_dim,
        ivr_dropout=ivr_dropout,
    )
