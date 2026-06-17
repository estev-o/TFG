#!/usr/bin/env python3
"""Analise da proba de peer review entre expertas, sistema e etiquetas orixinais."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


HPI_LABELS = list(range(0, 7))
IVR_LABELS = list(range(0, 8))
IVR_POSITIVE_LABELS = list(range(1, 8))
IVR_APPLICABLE_HPI = {1, 2, 3, 4}


@dataclass(frozen=True)
class RaterValues:
    hpi: Optional[int]
    ivr: Optional[int]


@dataclass
class PeerSample:
    sample_id: str
    anon_filename: str
    photo_cod: str
    original_filename: str
    original_path: str
    original: RaterValues
    sara: RaterValues
    nerea: RaterValues
    system: RaterValues
    solo_flag: bool
    solo_reason: str
    conservador_flag: bool
    conservador_reason: str
    system_conf_hpi: Optional[float]
    system_conf_ivr: Optional[float]
    yolo_conf: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analiza a proba peer_review comparando expertas, sistema e etiqueta orixinal."
    )
    parser.add_argument(
        "--answer-key",
        default="peer_review_dataset/answer_key_private.csv",
        help="CSV privado coas respostas orixinais.",
    )
    parser.add_argument(
        "--review-sara",
        default="peer_review_dataset/review_Sara.csv",
        help="CSV da revision de Sara.",
    )
    parser.add_argument(
        "--review-nerea",
        default="peer_review_dataset/review_Nerea.csv",
        help="CSV da revision de Nerea.",
    )
    parser.add_argument(
        "--system-solo",
        default="inferencia_resultados/20260616_131841_solo_dudoso/puntuaciones_total.csv",
        help="CSV total da inferencia co perfil solo_dudoso.",
    )
    parser.add_argument(
        "--system-conservador",
        default="inferencia_resultados/20260616_132050_conservador/puntuaciones_total.csv",
        help="CSV total da inferencia co perfil conservador.",
    )
    parser.add_argument(
        "--output-dir",
        default="peer_review_analysis",
        help="Directorio onde gardar o informe e os CSV de metricas.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Iteracions bootstrap para IC do 95%% en metricas principais (0 desactiva).",
    )
    parser.add_argument("--seed", type=int, default=20260616)
    return parser.parse_args()


def read_csv_fallback(path: Path, delimiter: str) -> list[dict[str, str]]:
    last_error: Optional[UnicodeDecodeError] = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with path.open(newline="", encoding=encoding) as handle:
                return list(csv.DictReader(handle, delimiter=delimiter))
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError(
        last_error.encoding if last_error else "unknown",
        last_error.object if last_error else b"",
        last_error.start if last_error else 0,
        last_error.end if last_error else 0,
        f"Non se puido ler {path} con utf-8/latin-1",
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize_sample_id(raw: str) -> str:
    value = Path(str(raw).strip()).stem
    return value


def parse_optional_int(raw: Any, field: str, min_value: int, max_value: int) -> Optional[int]:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    text = text.replace(",", ".")
    value = float(text)
    rounded = int(round(value))
    if abs(value - rounded) > 1e-6:
        raise ValueError(f"{field} debe ser enteiro; recibido {raw!r}")
    if rounded < min_value or rounded > max_value:
        raise ValueError(
            f"{field} fora de rango [{min_value}, {max_value}]; recibido {raw!r}"
        )
    return rounded


def parse_optional_float(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    return float(text.replace(",", "."))


def parse_flag(raw: Any) -> bool:
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "si", "sí"}


def parse_human_values(row: dict[str, str], rater_name: str) -> RaterValues:
    hpi = parse_optional_int(row.get("hpi_review"), f"{rater_name}.hpi_review", 0, 6)
    ivr = parse_optional_int(row.get("ivr_review"), f"{rater_name}.ivr_review", 0, 7)
    if ivr is None and hpi is not None and hpi not in IVR_APPLICABLE_HPI:
        ivr = 0
    return RaterValues(hpi=hpi, ivr=ivr)


def require_unique(rows: Iterable[dict[str, str]], id_column: str, source: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = normalize_sample_id(row[id_column])
        if sample_id in out:
            raise ValueError(f"Identificador duplicado en {source}: {sample_id}")
        out[sample_id] = row
    return out


def load_samples(args: argparse.Namespace) -> tuple[list[PeerSample], dict[str, Any]]:
    answer_rows = read_csv_fallback(Path(args.answer_key), delimiter=",")
    sara_rows = read_csv_fallback(Path(args.review_sara), delimiter=";")
    nerea_rows = read_csv_fallback(Path(args.review_nerea), delimiter=";")
    solo_rows = read_csv_fallback(Path(args.system_solo), delimiter=",")
    conservador_rows = read_csv_fallback(Path(args.system_conservador), delimiter=",")

    answer_by_id = require_unique(answer_rows, "anon_filename", "answer_key")
    sara_by_id = require_unique(sara_rows, "anon_filename", "review_Sara")
    nerea_by_id = require_unique(nerea_rows, "anon_filename", "review_Nerea")
    solo_by_id = require_unique(solo_rows, "imagen", "system_solo")
    conservador_by_id = require_unique(conservador_rows, "imagen", "system_conservador")

    expected_ids = set(answer_by_id)
    for source, ids in (
        ("review_Sara", set(sara_by_id)),
        ("review_Nerea", set(nerea_by_id)),
        ("system_solo", set(solo_by_id)),
        ("system_conservador", set(conservador_by_id)),
    ):
        missing = sorted(expected_ids - ids)
        extra = sorted(ids - expected_ids)
        if missing or extra:
            raise ValueError(
                f"IDs non coinciden en {source}: faltan={missing[:5]} extra={extra[:5]}"
            )

    prediction_differences: list[dict[str, Any]] = []
    for sample_id in sorted(expected_ids):
        solo = solo_by_id[sample_id]
        conservador = conservador_by_id[sample_id]
        for column in ("pred_hpi", "pred_ivr", "confianza_hpi", "confianza_ivr", "confianza_yolo"):
            if solo.get(column) != conservador.get(column):
                prediction_differences.append(
                    {
                        "sample_id": sample_id,
                        "column": column,
                        "solo_dudoso": solo.get(column, ""),
                        "conservador": conservador.get(column, ""),
                    }
                )

    samples: list[PeerSample] = []
    for sample_id in sorted(expected_ids):
        answer = answer_by_id[sample_id]
        sara = sara_by_id[sample_id]
        nerea = nerea_by_id[sample_id]
        solo = solo_by_id[sample_id]
        conservador = conservador_by_id[sample_id]

        original = RaterValues(
            hpi=parse_optional_int(answer.get("hpi"), "answer.hpi", 0, 6),
            ivr=parse_optional_int(answer.get("ivr"), "answer.ivr", 0, 7),
        )
        system = RaterValues(
            hpi=parse_optional_int(solo.get("pred_hpi"), "system.pred_hpi", 0, 6),
            ivr=parse_optional_int(solo.get("pred_ivr"), "system.pred_ivr", 0, 7),
        )
        samples.append(
            PeerSample(
                sample_id=sample_id,
                anon_filename=answer["anon_filename"],
                photo_cod=answer.get("photo_cod", ""),
                original_filename=answer.get("original_filename", ""),
                original_path=answer.get("original_path", ""),
                original=original,
                sara=parse_human_values(sara, "Sara"),
                nerea=parse_human_values(nerea, "Nerea"),
                system=system,
                solo_flag=parse_flag(solo.get("necesita_revision")),
                solo_reason=solo.get("motivo_revision", ""),
                conservador_flag=parse_flag(conservador.get("necesita_revision")),
                conservador_reason=conservador.get("motivo_revision", ""),
                system_conf_hpi=parse_optional_float(solo.get("confianza_hpi")),
                system_conf_ivr=parse_optional_float(solo.get("confianza_ivr")),
                yolo_conf=parse_optional_float(solo.get("confianza_yolo")),
            )
        )

    metadata = {
        "n_answer_key": len(answer_rows),
        "n_review_sara": len(sara_rows),
        "n_review_nerea": len(nerea_rows),
        "n_system_solo": len(solo_rows),
        "n_system_conservador": len(conservador_rows),
        "system_predictions_equal": len(prediction_differences) == 0,
        "system_prediction_differences": prediction_differences,
    }
    return samples, metadata


def values_for(samples: Sequence[PeerSample], rater: str, target: str) -> list[Optional[int]]:
    return [getattr(getattr(sample, rater), target) for sample in samples]


def valid_pairs(
    ref_values: Sequence[Optional[int]], comp_values: Sequence[Optional[int]]
) -> tuple[list[int], list[int], int]:
    ref: list[int] = []
    comp: list[int] = []
    missing = 0
    for a, b in zip(ref_values, comp_values):
        if a is None or b is None:
            missing += 1
            continue
        ref.append(a)
        comp.append(b)
    return ref, comp, missing


def safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def round_or_none(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def confusion_matrix(ref: Sequence[int], comp: Sequence[int], labels: Sequence[int]) -> list[list[int]]:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for a, b in zip(ref, comp):
        if a in label_to_idx and b in label_to_idx:
            matrix[label_to_idx[a]][label_to_idx[b]] += 1
    return matrix


def weighted_kappa(
    ref: Sequence[int], comp: Sequence[int], labels: Sequence[int], weight: str = "quadratic"
) -> Optional[float]:
    if not ref:
        return None
    matrix = confusion_matrix(ref, comp, labels)
    n = sum(sum(row) for row in matrix)
    if n == 0:
        return None

    row_totals = [sum(row) for row in matrix]
    col_totals = [sum(matrix[i][j] for i in range(len(labels))) for j in range(len(labels))]
    max_distance = max(len(labels) - 1, 1)

    observed = 0.0
    expected = 0.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            distance = abs(i - j)
            if weight == "linear":
                penalty = distance / max_distance
            elif weight == "quadratic":
                penalty = (distance / max_distance) ** 2
            else:
                raise ValueError(f"Peso kappa non soportado: {weight}")
            observed += penalty * matrix[i][j] / n
            expected += penalty * row_totals[i] * col_totals[j] / (n * n)

    if expected == 0:
        return 1.0 if observed == 0 else None
    return 1.0 - observed / expected


def ordinal_metrics(
    ref_values: Sequence[Optional[int]],
    comp_values: Sequence[Optional[int]],
    labels: Sequence[int],
) -> dict[str, Any]:
    ref, comp, missing = valid_pairs(ref_values, comp_values)
    n = len(ref)
    if n == 0:
        return {
            "n": 0,
            "missing": missing,
            "mae": None,
            "rmse": None,
            "exact": None,
            "within_1": None,
            "within_2": None,
            "kappa_linear": None,
            "kappa_quadratic": None,
            "confusion_matrix": {"labels": [str(x) for x in labels], "matrix": []},
        }

    abs_errors = [abs(a - b) for a, b in zip(ref, comp)]
    sq_errors = [err * err for err in abs_errors]
    return {
        "n": n,
        "missing": missing,
        "mae": sum(abs_errors) / n,
        "rmse": math.sqrt(sum(sq_errors) / n),
        "exact": sum(1 for err in abs_errors if err == 0) / n,
        "within_1": sum(1 for err in abs_errors if err <= 1) / n,
        "within_2": sum(1 for err in abs_errors if err <= 2) / n,
        "kappa_linear": weighted_kappa(ref, comp, labels, "linear"),
        "kappa_quadratic": weighted_kappa(ref, comp, labels, "quadratic"),
        "confusion_matrix": {
            "labels": [str(x) for x in labels],
            "matrix": confusion_matrix(ref, comp, labels),
        },
    }


def binary_metrics(
    ref_values: Sequence[Optional[int]], comp_values: Sequence[Optional[int]]
) -> dict[str, Any]:
    ref_raw, comp_raw, missing = valid_pairs(ref_values, comp_values)
    ref = [value > 0 for value in ref_raw]
    comp = [value > 0 for value in comp_raw]
    n = len(ref)
    if n == 0:
        return {
            "n": 0,
            "missing": missing,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }

    tp = sum(1 for a, b in zip(ref, comp) if a and b)
    fp = sum(1 for a, b in zip(ref, comp) if (not a) and b)
    fn = sum(1 for a, b in zip(ref, comp) if a and (not b))
    tn = sum(1 for a, b in zip(ref, comp) if (not a) and (not b))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "n": n,
        "missing": missing,
        "accuracy": (tp + tn) / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def ivr_conditional_metrics(
    ref_values: Sequence[Optional[int]], comp_values: Sequence[Optional[int]]
) -> dict[str, Any]:
    filtered_ref: list[Optional[int]] = []
    filtered_comp: list[Optional[int]] = []
    missing = 0
    excluded_not_jointly_applicable = 0
    for ref, comp in zip(ref_values, comp_values):
        if ref is None or comp is None:
            missing += 1
            continue
        if ref > 0 and comp > 0:
            filtered_ref.append(ref)
            filtered_comp.append(comp)
        else:
            excluded_not_jointly_applicable += 1
    metrics = ordinal_metrics(filtered_ref, filtered_comp, IVR_POSITIVE_LABELS)
    metrics["missing"] = missing
    metrics["excluded_not_jointly_applicable"] = excluded_not_jointly_applicable
    return metrics


def consistency_metrics(samples: Sequence[PeerSample], rater: str) -> dict[str, Any]:
    n = 0
    missing = 0
    inconsistent = 0
    positive_when_hpi_not_applicable = 0
    zero_when_hpi_applicable = 0
    for sample in samples:
        values = getattr(sample, rater)
        if values.hpi is None or values.ivr is None:
            missing += 1
            continue
        n += 1
        expected_applicable = values.hpi in IVR_APPLICABLE_HPI
        ivr_applicable = values.ivr > 0
        if expected_applicable != ivr_applicable:
            inconsistent += 1
            if ivr_applicable:
                positive_when_hpi_not_applicable += 1
            else:
                zero_when_hpi_applicable += 1
    return {
        "rater": rater,
        "n": n,
        "missing": missing,
        "consistency_accuracy": safe_div(n - inconsistent, n),
        "inconsistent_total": inconsistent,
        "positive_when_hpi_not_applicable": positive_when_hpi_not_applicable,
        "zero_when_hpi_applicable": zero_when_hpi_applicable,
    }


def pairwise_metrics(samples: Sequence[PeerSample], ref: str, comp: str) -> dict[str, Any]:
    ref_hpi = values_for(samples, ref, "hpi")
    comp_hpi = values_for(samples, comp, "hpi")
    ref_ivr = values_for(samples, ref, "ivr")
    comp_ivr = values_for(samples, comp, "ivr")
    return {
        "reference": ref,
        "comparison": comp,
        "n_samples": len(samples),
        "hpi": ordinal_metrics(ref_hpi, comp_hpi, HPI_LABELS),
        "ivr_legacy_all": ordinal_metrics(ref_ivr, comp_ivr, IVR_LABELS),
        "ivr_applicability": binary_metrics(ref_ivr, comp_ivr),
        "ivr_conditional_joint_applicable": ivr_conditional_metrics(ref_ivr, comp_ivr),
    }


def flatten_pairwise_metrics(section: str, metrics: dict[str, Any]) -> dict[str, Any]:
    hpi = metrics["hpi"]
    legacy = metrics["ivr_legacy_all"]
    app = metrics["ivr_applicability"]
    cond = metrics["ivr_conditional_joint_applicable"]
    return {
        "section": section,
        "reference": metrics["reference"],
        "comparison": metrics["comparison"],
        "n_samples": metrics["n_samples"],
        "hpi_n": hpi["n"],
        "hpi_missing": hpi["missing"],
        "hpi_mae": round_or_none(hpi["mae"]),
        "hpi_rmse": round_or_none(hpi["rmse"]),
        "hpi_exact": round_or_none(hpi["exact"]),
        "hpi_within_1": round_or_none(hpi["within_1"]),
        "hpi_within_2": round_or_none(hpi["within_2"]),
        "hpi_kappa_quadratic": round_or_none(hpi["kappa_quadratic"]),
        "ivr_legacy_n": legacy["n"],
        "ivr_legacy_missing": legacy["missing"],
        "ivr_legacy_mae": round_or_none(legacy["mae"]),
        "ivr_legacy_rmse": round_or_none(legacy["rmse"]),
        "ivr_legacy_exact": round_or_none(legacy["exact"]),
        "ivr_legacy_within_1": round_or_none(legacy["within_1"]),
        "ivr_legacy_within_2": round_or_none(legacy["within_2"]),
        "ivr_legacy_kappa_quadratic": round_or_none(legacy["kappa_quadratic"]),
        "ivr_app_n": app["n"],
        "ivr_app_missing": app["missing"],
        "ivr_app_accuracy": round_or_none(app["accuracy"]),
        "ivr_app_precision": round_or_none(app["precision"]),
        "ivr_app_recall": round_or_none(app["recall"]),
        "ivr_app_f1": round_or_none(app["f1"]),
        "ivr_app_tp": app["tp"],
        "ivr_app_fp": app["fp"],
        "ivr_app_fn": app["fn"],
        "ivr_app_tn": app["tn"],
        "ivr_cond_n": cond["n"],
        "ivr_cond_missing": cond["missing"],
        "ivr_cond_excluded_not_jointly_applicable": cond["excluded_not_jointly_applicable"],
        "ivr_cond_mae": round_or_none(cond["mae"]),
        "ivr_cond_rmse": round_or_none(cond["rmse"]),
        "ivr_cond_exact": round_or_none(cond["exact"]),
        "ivr_cond_within_1": round_or_none(cond["within_1"]),
        "ivr_cond_within_2": round_or_none(cond["within_2"]),
        "ivr_cond_kappa_quadratic": round_or_none(cond["kappa_quadratic"]),
    }


def mean_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def per_sample_pairwise_abs(values: Sequence[Optional[int]]) -> list[int]:
    present = [value for value in values if value is not None]
    return [abs(a - b) for idx, a in enumerate(present) for b in present[idx + 1 :]]


def per_sample_range(values: Sequence[Optional[int]]) -> Optional[int]:
    present = [value for value in values if value is not None]
    if len(present) < 2:
        return None
    return max(present) - min(present)


def variability_summary(samples: Sequence[PeerSample], name: str) -> dict[str, Any]:
    hpi_pair_errors: list[int] = []
    ivr_pair_errors: list[int] = []
    hpi_ranges: list[int] = []
    ivr_ranges: list[int] = []
    ivr_app_disagreements = 0
    ivr_app_comparable = 0
    any_hpi_disagreement = 0
    any_ivr_legacy_disagreement = 0

    for sample in samples:
        hpi_values = [sample.original.hpi, sample.sara.hpi, sample.nerea.hpi]
        ivr_values = [sample.original.ivr, sample.sara.ivr, sample.nerea.ivr]

        hpi_errors = per_sample_pairwise_abs(hpi_values)
        ivr_errors = per_sample_pairwise_abs(ivr_values)
        hpi_pair_errors.extend(hpi_errors)
        ivr_pair_errors.extend(ivr_errors)

        hpi_range = per_sample_range(hpi_values)
        ivr_range = per_sample_range(ivr_values)
        if hpi_range is not None:
            hpi_ranges.append(hpi_range)
            if hpi_range > 0:
                any_hpi_disagreement += 1
        if ivr_range is not None:
            ivr_ranges.append(ivr_range)
            if ivr_range > 0:
                any_ivr_legacy_disagreement += 1

        present_ivr = [value for value in ivr_values if value is not None]
        if len(present_ivr) >= 2:
            apps = [value > 0 for value in present_ivr]
            ivr_app_comparable += 1
            if any(app != apps[0] for app in apps[1:]):
                ivr_app_disagreements += 1

    return {
        "subset": name,
        "n_samples": len(samples),
        "hpi_pairwise_mae_experts_original": mean_optional([float(x) for x in hpi_pair_errors]),
        "hpi_mean_range_experts_original": mean_optional([float(x) for x in hpi_ranges]),
        "hpi_any_disagreement_rate": safe_div(any_hpi_disagreement, len(hpi_ranges)),
        "ivr_legacy_pairwise_mae_experts_original": mean_optional(
            [float(x) for x in ivr_pair_errors]
        ),
        "ivr_legacy_mean_range_experts_original": mean_optional([float(x) for x in ivr_ranges]),
        "ivr_legacy_any_disagreement_rate": safe_div(any_ivr_legacy_disagreement, len(ivr_ranges)),
        "ivr_app_disagreement_rate": safe_div(ivr_app_disagreements, ivr_app_comparable),
        "ivr_app_disagreement_count": ivr_app_disagreements,
        "ivr_app_comparable_count": ivr_app_comparable,
    }


def subset_map(samples: Sequence[PeerSample]) -> dict[str, list[PeerSample]]:
    return {
        "all": list(samples),
        "solo_dudoso_flagged": [sample for sample in samples if sample.solo_flag],
        "conservador_flagged": [sample for sample in samples if sample.conservador_flag],
        "conservador_extra": [
            sample for sample in samples if sample.conservador_flag and not sample.solo_flag
        ],
        "safe_in_conservador": [sample for sample in samples if not sample.conservador_flag],
        "safe_in_solo_dudoso": [sample for sample in samples if not sample.solo_flag],
    }


def bootstrap_ci(
    samples: Sequence[PeerSample],
    metric_fn: Any,
    iterations: int,
    rng: random.Random,
) -> dict[str, Optional[float]]:
    if iterations <= 0 or not samples:
        return {"mean": None, "ci95_low": None, "ci95_high": None}
    values: list[float] = []
    n = len(samples)
    for _ in range(iterations):
        draw = [samples[rng.randrange(n)] for _ in range(n)]
        value = metric_fn(draw)
        if value is not None:
            values.append(float(value))
    if not values:
        return {"mean": None, "ci95_low": None, "ci95_high": None}
    values.sort()
    low_idx = int(0.025 * (len(values) - 1))
    high_idx = int(0.975 * (len(values) - 1))
    return {
        "mean": sum(values) / len(values),
        "ci95_low": values[low_idx],
        "ci95_high": values[high_idx],
    }


def build_bootstrap_summary(
    samples: Sequence[PeerSample], iterations: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    specs = [
        (
            "human_variability_hpi_pairwise_mae",
            lambda rows: variability_summary(rows, "bootstrap")[
                "hpi_pairwise_mae_experts_original"
            ],
        ),
        (
            "human_variability_ivr_app_disagreement_rate",
            lambda rows: variability_summary(rows, "bootstrap")["ivr_app_disagreement_rate"],
        ),
        (
            "system_vs_original_hpi_mae",
            lambda rows: pairwise_metrics(rows, "original", "system")["hpi"]["mae"],
        ),
        (
            "system_vs_original_ivr_cond_mae",
            lambda rows: pairwise_metrics(rows, "original", "system")[
                "ivr_conditional_joint_applicable"
            ]["mae"],
        ),
        (
            "sara_vs_nerea_hpi_mae",
            lambda rows: pairwise_metrics(rows, "sara", "nerea")["hpi"]["mae"],
        ),
        (
            "sara_vs_nerea_ivr_cond_mae",
            lambda rows: pairwise_metrics(rows, "sara", "nerea")[
                "ivr_conditional_joint_applicable"
            ]["mae"],
        ),
    ]
    rows: list[dict[str, Any]] = []
    for metric_name, fn in specs:
        ci = bootstrap_ci(samples, fn, iterations, rng)
        rows.append(
            {
                "metric": metric_name,
                "bootstrap_iterations": iterations,
                "mean": round_or_none(ci["mean"]),
                "ci95_low": round_or_none(ci["ci95_low"]),
                "ci95_high": round_or_none(ci["ci95_high"]),
            }
        )
    return rows


def sample_rows(samples: Sequence[PeerSample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        expert_original_hpi = [
            sample.original.hpi,
            sample.sara.hpi,
            sample.nerea.hpi,
        ]
        expert_original_ivr = [
            sample.original.ivr,
            sample.sara.ivr,
            sample.nerea.ivr,
        ]
        rows.append(
            {
                "sample_id": sample.sample_id,
                "anon_filename": sample.anon_filename,
                "photo_cod": sample.photo_cod,
                "original_filename": sample.original_filename,
                "original_path": sample.original_path,
                "original_hpi": sample.original.hpi,
                "original_ivr": sample.original.ivr,
                "sara_hpi": sample.sara.hpi,
                "sara_ivr": sample.sara.ivr,
                "nerea_hpi": sample.nerea.hpi,
                "nerea_ivr": sample.nerea.ivr,
                "system_hpi": sample.system.hpi,
                "system_ivr": sample.system.ivr,
                "solo_dudoso_flag": int(sample.solo_flag),
                "solo_dudoso_reason": sample.solo_reason,
                "conservador_flag": int(sample.conservador_flag),
                "conservador_reason": sample.conservador_reason,
                "system_conf_hpi": sample.system_conf_hpi,
                "system_conf_ivr": sample.system_conf_ivr,
                "yolo_conf": sample.yolo_conf,
                "hpi_range_experts_original": per_sample_range(expert_original_hpi),
                "ivr_range_experts_original": per_sample_range(expert_original_ivr),
                "ivr_app_disagreement_experts_original": int(
                    len({value > 0 for value in expert_original_ivr if value is not None}) > 1
                ),
            }
        )
    return rows


def format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def markdown_table(rows: Sequence[dict[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    header = "| " + " | ".join(title for title, _ in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(key, "")) for _, key in columns) + " |")
    return "\n".join([header, separator, *body])


def build_report(
    samples: Sequence[PeerSample],
    metadata: dict[str, Any],
    pairwise_rows: Sequence[dict[str, Any]],
    variability_rows: Sequence[dict[str, Any]],
    consistency_rows: Sequence[dict[str, Any]],
    bootstrap_rows: Sequence[dict[str, Any]],
) -> str:
    flag_counts = {
        "solo_dudoso": sum(1 for sample in samples if sample.solo_flag),
        "conservador": sum(1 for sample in samples if sample.conservador_flag),
        "conservador_extra": sum(
            1 for sample in samples if sample.conservador_flag and not sample.solo_flag
        ),
    }

    key_pairs = [
        row
        for row in pairwise_rows
        if row["section"] in {
            "expertos_entre_si",
            "expertos_vs_orixinal",
            "sistema_vs_orixinal",
            "sistema_vs_expertos",
        }
    ]
    key_pair_rows = [
        {
            "comparacion": f"{row['reference']} vs {row['comparison']}",
            "hpi_mae": format_float(row["hpi_mae"]),
            "hpi_w1": format_float(row["hpi_within_1"]),
            "hpi_kappa": format_float(row["hpi_kappa_quadratic"]),
            "ivr_app_f1": format_float(row["ivr_app_f1"]),
            "ivr_cond_mae": format_float(row["ivr_cond_mae"]),
            "ivr_cond_w1": format_float(row["ivr_cond_within_1"]),
        }
        for row in key_pairs
    ]

    variability_md_rows = [
        {
            "subset": row["subset"],
            "n": row["n_samples"],
            "hpi_pair_mae": format_float(row["hpi_pairwise_mae_experts_original"]),
            "hpi_any": format_float(row["hpi_any_disagreement_rate"]),
            "ivr_app_dis": format_float(row["ivr_app_disagreement_rate"]),
            "ivr_pair_mae": format_float(row["ivr_legacy_pairwise_mae_experts_original"]),
        }
        for row in variability_rows
    ]

    consistency_md_rows = [
        {
            "avaliador": row["rater"],
            "n": row["n"],
            "acc": format_float(row["consistency_accuracy"]),
            "incons": row["inconsistent_total"],
            "pos_non_app": row["positive_when_hpi_not_applicable"],
            "zero_app": row["zero_when_hpi_applicable"],
        }
        for row in consistency_rows
    ]

    bootstrap_md_rows = [
        {
            "metric": row["metric"],
            "mean": format_float(row["mean"]),
            "low": format_float(row["ci95_low"]),
            "high": format_float(row["ci95_high"]),
        }
        for row in bootstrap_rows
    ]

    lines = [
        "# Analise peer review",
        "",
        f"Mostras analizadas: {len(samples)}.",
        f"Predicions identicas entre solo_dudoso e conservador: {metadata['system_predictions_equal']}.",
        (
            "Como ambos perfis comparten inferencia, o sistema avaliase unha soa vez; "
            "os perfis usanse para estudar se os flags de revision capturan maior variabilidade."
        ),
        "",
        "## Flags de revision",
        "",
        f"- solo_dudoso: {flag_counts['solo_dudoso']} mostras marcadas.",
        f"- conservador: {flag_counts['conservador']} mostras marcadas.",
        f"- adicionais de conservador fronte a solo_dudoso: {flag_counts['conservador_extra']}.",
        "",
        "## Comparacions principais",
        "",
        markdown_table(
            key_pair_rows,
            [
                ("Comparacion", "comparacion"),
                ("HPI MAE", "hpi_mae"),
                ("HPI within1", "hpi_w1"),
                ("HPI kappa quad.", "hpi_kappa"),
                ("IVR app F1", "ivr_app_f1"),
                ("IVR cond MAE", "ivr_cond_mae"),
                ("IVR cond within1", "ivr_cond_w1"),
            ],
        ),
        "",
        "## Variabilidade nas zonas marcadas",
        "",
        markdown_table(
            variability_md_rows,
            [
                ("Subset", "subset"),
                ("n", "n"),
                ("HPI pair MAE", "hpi_pair_mae"),
                ("HPI any disag.", "hpi_any"),
                ("IVR app disag.", "ivr_app_dis"),
                ("IVR pair MAE", "ivr_pair_mae"),
            ],
        ),
        "",
        "## Consistencia HPI-IVR por avaliador",
        "",
        markdown_table(
            consistency_md_rows,
            [
                ("Avaliador", "avaliador"),
                ("n", "n"),
                ("Accuracy", "acc"),
                ("Incons.", "incons"),
                ("IVR>0 con HPI non aplic.", "pos_non_app"),
                ("IVR=0 con HPI aplic.", "zero_app"),
            ],
        ),
    ]
    if bootstrap_rows:
        lines.extend(
            [
                "",
                "## Bootstrap IC 95%",
                "",
                markdown_table(
                    bootstrap_md_rows,
                    [
                        ("Metrica", "metric"),
                        ("Media", "mean"),
                        ("IC low", "low"),
                        ("IC high", "high"),
                    ],
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Lectura",
            "",
            (
                "A comparacion relevante non e so sistema contra etiqueta orixinal: "
                "hai que contrastar esa distancia coa distancia Sara-Nerea e coas distancias "
                "entre cada experta e a etiqueta orixinal. Se o sistema cae nun rango parecido, "
                "a evidencia apoia que funciona como un avaliador experto adicional."
            ),
            (
                "Para IVR, as metricas principais son aplicabilidade e nota condicional. "
                "A metrica legacy sobre todas as mostras queda gardada nos CSV/JSON, pero mestura "
                "a decision de se IVR aplica coa nota 1..7."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, metadata = load_samples(args)
    subsets = subset_map(samples)

    pair_specs = [
        ("expertos_entre_si", "sara", "nerea"),
        ("expertos_vs_orixinal", "original", "sara"),
        ("expertos_vs_orixinal", "original", "nerea"),
        ("sistema_vs_orixinal", "original", "system"),
        ("sistema_vs_expertos", "sara", "system"),
        ("sistema_vs_expertos", "nerea", "system"),
    ]

    pairwise_full: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    for section, ref, comp in pair_specs:
        metrics = pairwise_metrics(samples, ref, comp)
        pairwise_full.append({"section": section, **metrics})
        pairwise_rows.append(flatten_pairwise_metrics(section, metrics))

    variability_rows = [
        {
            key: round_or_none(value) if isinstance(value, float) else value
            for key, value in variability_summary(subset_samples, subset_name).items()
        }
        for subset_name, subset_samples in subsets.items()
    ]

    flag_pair_rows: list[dict[str, Any]] = []
    for subset_name, subset_samples in subsets.items():
        for ref, comp in (("original", "sara"), ("original", "nerea"), ("sara", "nerea")):
            metrics = pairwise_metrics(subset_samples, ref, comp)
            flag_pair_rows.append(flatten_pairwise_metrics(subset_name, metrics))

    consistency_rows = [
        {
            key: round_or_none(value) if isinstance(value, float) else value
            for key, value in consistency_metrics(samples, rater).items()
        }
        for rater in ("original", "sara", "nerea", "system")
    ]

    bootstrap_rows = [
        {
            key: round_or_none(value) if isinstance(value, float) else value
            for key, value in row.items()
        }
        for row in build_bootstrap_summary(samples, args.bootstrap, args.seed)
    ]

    sample_detail_rows = sample_rows(samples)

    pairwise_fieldnames = list(pairwise_rows[0].keys())
    write_csv(output_dir / "pairwise_metrics.csv", pairwise_rows, pairwise_fieldnames)
    write_csv(output_dir / "flag_subset_pairwise_metrics.csv", flag_pair_rows, pairwise_fieldnames)
    write_csv(
        output_dir / "flag_variability_summary.csv",
        variability_rows,
        list(variability_rows[0].keys()),
    )
    write_csv(
        output_dir / "consistency_metrics.csv",
        consistency_rows,
        list(consistency_rows[0].keys()),
    )
    if bootstrap_rows:
        write_csv(
            output_dir / "bootstrap_ci.csv",
            bootstrap_rows,
            list(bootstrap_rows[0].keys()),
        )
    write_csv(
        output_dir / "joined_peer_review.csv",
        sample_detail_rows,
        list(sample_detail_rows[0].keys()),
    )

    report = build_report(
        samples,
        metadata,
        pairwise_rows,
        variability_rows,
        consistency_rows,
        bootstrap_rows,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    full_json = {
        "metadata": metadata,
        "pairwise_metrics": pairwise_full,
        "pairwise_metrics_flat": pairwise_rows,
        "flag_subset_pairwise_metrics": flag_pair_rows,
        "flag_variability_summary": variability_rows,
        "consistency_metrics": consistency_rows,
        "bootstrap_ci": bootstrap_rows,
    }
    (output_dir / "metrics_peer_review.json").write_text(
        json.dumps(full_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Mostras analizadas: {len(samples)}")
    print(f"Predicions iguais solo_dudoso/conservador: {metadata['system_predictions_equal']}")
    print(f"Informe: {(output_dir / 'report.md').as_posix()}")
    print(f"Metricas: {(output_dir / 'pairwise_metrics.csv').as_posix()}")


if __name__ == "__main__":
    main()
