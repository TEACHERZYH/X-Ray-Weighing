from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


PALETTE = {
    "data": "#6F7686",
    "physics": "#2E6F9E",
    "temporal": "#4C8B6B",
    "pgt": "#C46B37",
    "pgt_light": "#E3A06D",
    "risk": "#B54E4E",
    "hgb": "#8A5A9E",
    "rf": "#8B7C3F",
    "target": "#B54E4E",
    "grid": "#E7E9EE",
    "text": "#222222",
    "muted": "#6F7686",
    "panel_bg": "#F8FAFC",
}

METHOD_COLORS = {
    "纯数据": PALETTE["data"],
    "纯数据HGB": PALETTE["data"],
    "DataOnly-HGB": PALETTE["data"],
    "DataOnly-Ridge": PALETTE["data"],
    "物理+I0": PALETTE["physics"],
    "Physics+I0": PALETTE["physics"],
    "物理+岭回归": PALETTE["physics"],
    "Physics-Ridge": PALETTE["physics"],
    "物理+时序": PALETTE["temporal"],
    "Physics-Temporal-Ridge": PALETTE["temporal"],
    "PG-TSC-lite": PALETTE["pgt"],
    "PG-TSC-lite（180样本）": PALETTE["pgt"],
    "PG-TSC-TCN": "#8B6FB3",
    "PG-TSC-TCN（20样本）": "#8B6FB3",
    "HGB": PALETTE["hgb"],
    "RandomForest": PALETTE["rf"],
}


def setup_publication_style(font_size: float = 7.5) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Noto Sans SC", "SimHei", "Arial", "DejaVu Sans"],
            "font.size": font_size,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.unicode_minus": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
        }
    )


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color=PALETTE["text"],
    )


def add_target_line(ax: plt.Axes, y: float = 1.0, label: str = "1%目标") -> None:
    ax.axhline(y, color=PALETTE["target"], linewidth=0.85, linestyle=(0, (4, 2)), zorder=1)
    ymin, ymax = ax.get_ylim()
    if ymin <= y <= ymax:
        ax.text(
            0.99,
            y,
            label,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=6.2,
            color=PALETTE["target"],
        )


def soften_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=PALETTE["grid"], linewidth=0.55, zorder=0)
    ax.tick_params(colors=PALETTE["text"], labelsize=6.8)
    ax.xaxis.label.set_size(7.2)
    ax.yaxis.label.set_size(7.2)
    ax.title.set_size(7.8)
    ax.title.set_weight("bold")


def strip_trailing_space(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")


def save_figure(fig: plt.Figure, out_base: Path, *, png_dpi: int = 450, tiff_dpi: int = 450, pdf: bool = True) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    svg = out_base.with_suffix(".svg")
    png = out_base.with_suffix(".png")
    tiff = out_base.with_suffix(".tiff")
    fig.savefig(svg, bbox_inches="tight")
    strip_trailing_space(svg)
    fig.savefig(png, dpi=png_dpi, bbox_inches="tight")
    if pdf:
        fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(tiff, dpi=tiff_dpi, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
