#!/usr/bin/env python3
"""Generate an approximate Gantt chart for the TFG development.

The script intentionally has no third-party dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SVG_OUT = ROOT / "imaxes" / "cronograma_gantt.svg"
TEX_OUT = ROOT / "imaxes" / "cronograma_gantt.tex"

START = date(2025, 10, 1)
END = date(2026, 6, 17)


@dataclass(frozen=True)
class Task:
    name: str
    start: date
    end: date
    color: str


TASKS = [
    Task("Planificación e análise do problema", date(2025, 10, 1), date(2025, 10, 31), "udcblue"),
    Task("Revisión do material e variables expertas", date(2025, 10, 20), date(2025, 11, 30), "udcpink"),
    Task("Limpeza inicial e preparación do dataset", date(2025, 11, 15), date(2025, 12, 9), "udcorange"),
    Task("Detección con YOLO e recorte das algas", date(2025, 12, 1), date(2026, 1, 26), "udcgreen"),
    Task("Normalización e curación dos recortes", date(2026, 1, 26), date(2026, 2, 18), "udcteal"),
    Task("Preparación de splits e primeira CNN", date(2026, 3, 1), date(2026, 3, 25), "udcblue"),
    Task("Reformulación ordinal e comparacións", date(2026, 3, 25), date(2026, 4, 17), "udcpink"),
    Task("Optimización de IVR e análise de erros", date(2026, 4, 16), date(2026, 5, 22), "udcorange"),
    Task("Versión final e integración da pipeline", date(2026, 5, 21), date(2026, 6, 5), "udcgreen"),
    Task("Validación con expertas e peche da memoria", date(2026, 6, 1), date(2026, 6, 17), "udcteal"),
]

MONTHS = [
    ("Out.", date(2025, 10, 1)),
    ("Nov.", date(2025, 11, 1)),
    ("Dec.", date(2025, 12, 1)),
    ("Xan.", date(2026, 1, 1)),
    ("Feb.", date(2026, 2, 1)),
    ("Mar.", date(2026, 3, 1)),
    ("Abr.", date(2026, 4, 1)),
    ("Mai.", date(2026, 5, 1)),
    ("Xuñ.", date(2026, 6, 1)),
]

SVG_COLORS = {
    "udcblue": "#1f77b4",
    "udcpink": "#c23b75",
    "udcorange": "#d97927",
    "udcgreen": "#3f8f3f",
    "udcteal": "#2a8c8c",
}

TIKZ_COLORS = {
    "udcblue": "1f77b4",
    "udcpink": "c23b75",
    "udcorange": "d97927",
    "udcgreen": "3f8f3f",
    "udcteal": "2a8c8c",
}


def days(d: date) -> int:
    return (d - START).days


def generate_svg() -> None:
    width = 1500
    left = 430
    right = 40
    top = 90
    row_h = 46
    chart_w = width - left - right
    height = top + len(TASKS) * row_h + 95
    total = days(END)

    def x(d: date) -> float:
        return left + days(d) / total * chart_w

    lines: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="40" y="42" font-family="Arial, sans-serif" font-size="25" font-weight="700" fill="#222">Cronograma aproximado do desenvolvemento</text>',
        '<text x="40" y="68" font-family="Arial, sans-serif" font-size="15" fill="#555">Outubro de 2025 - xuño de 2026</text>',
    ]

    for label, month in MONTHS:
        xx = x(month)
        lines.append(f'<line x1="{xx:.1f}" y1="{top - 25}" x2="{xx:.1f}" y2="{height - 60}" stroke="#d8d8d8" stroke-width="1"/>')
        lines.append(f'<text x="{xx + 4:.1f}" y="{top - 35}" font-family="Arial, sans-serif" font-size="15" fill="#333">{label}</text>')

    for idx, task in enumerate(TASKS):
        y = top + idx * row_h
        bg = "#f6f6f6" if idx % 2 == 0 else "#ffffff"
        lines.append(f'<rect x="30" y="{y - 24}" width="{width - 60}" height="{row_h}" fill="{bg}"/>')
        lines.append(f'<text x="40" y="{y + 5}" font-family="Arial, sans-serif" font-size="16" fill="#222">{task.name}</text>')
        xx = x(task.start)
        ww = max(4, x(task.end) - xx)
        lines.append(f'<rect x="{xx:.1f}" y="{y - 13}" width="{ww:.1f}" height="24" rx="4" fill="{SVG_COLORS[task.color]}"/>')

    lines.append(f'<line x1="{left}" y1="{top - 25}" x2="{left + chart_w}" y2="{top - 25}" stroke="#777" stroke-width="1.2"/>')
    lines.append("</svg>")
    SVG_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_tex() -> None:
    total = days(END)
    scale = 0.045
    row_h = 0.55
    left_text = -8.0
    chart_w = total * scale
    bottom = -len(TASKS) * row_h - 0.25

    lines: list[str] = [
        r"\documentclass[tikz,border=6pt]{standalone}",
        r"\usepackage{fontspec}",
        r"\setmainfont{Linux Libertine O}",
        r"\usetikzlibrary{calc}",
    ]
    for name, value in TIKZ_COLORS.items():
        lines.append(rf"\definecolor{{{name}}}{{HTML}}{{{value}}}")
    lines.extend(
        [
            r"\begin{document}",
            r"\begin{tikzpicture}[x=1cm,y=1cm]",
            rf"\node[anchor=west,font=\bfseries\large] at ({left_text},0.65) {{Cronograma aproximado do desenvolvemento}};",
            rf"\node[anchor=west,font=\small,text=black!65] at ({left_text},0.25) {{Outubro de 2025 -- xuño de 2026}};",
        ]
    )

    for label, month in MONTHS:
        x = days(month) * scale
        lines.append(rf"\draw[black!18] ({x:.3f},0.05) -- ({x:.3f},{bottom:.3f});")
        lines.append(rf"\node[anchor=west,font=\scriptsize] at ({x + 0.03:.3f},-0.12) {{{label}}};")
    lines.append(rf"\draw[black!45] (0,0.05) -- ({chart_w:.3f},0.05);")

    for idx, task in enumerate(TASKS):
        y = -0.55 - idx * row_h
        x0 = days(task.start) * scale
        x1 = days(task.end) * scale
        lines.append(rf"\node[anchor=east,font=\scriptsize,align=right,text width=7.6cm] at (-0.18,{y:.3f}) {{{task.name}}};")
        lines.append(rf"\fill[{task.color},rounded corners=1.5pt] ({x0:.3f},{y - 0.12:.3f}) rectangle ({x1:.3f},{y + 0.12:.3f});")

    lines.extend([r"\end{tikzpicture}", r"\end{document}"])
    TEX_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    SVG_OUT.parent.mkdir(parents=True, exist_ok=True)
    generate_svg()
    generate_tex()
    print(f"Generated {SVG_OUT.relative_to(ROOT)}")
    print(f"Generated {TEX_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
