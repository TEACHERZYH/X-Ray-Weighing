from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from figure_style import PALETTE, save_figure, setup_publication_style


OUT = Path("paper/figures")

NAVY = "#0B2248"
BLUE = "#1E6EA9"
TEAL = "#0F766E"
GREEN = "#21805C"
ORANGE = "#C96D2D"
PURPLE = "#6C4BA2"
RED = "#C93F3F"
SLATE = "#536070"
LINE = "#B9C5D2"
PANEL = "#F7F9FC"


def setup() -> None:
    setup_publication_style(font_size=6.8)
    plt.rcParams.update({"axes.spines.left": False, "axes.spines.bottom": False})


def box(ax, xy, wh, text, fc="#F2F4F7", ec="#4F5B66", size=7.2):
    r = Rectangle(xy, wh[0], wh[1], facecolor=fc, edgecolor=ec, linewidth=1.0)
    ax.add_patch(r)
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, text, ha="center", va="center", fontsize=size)


def round_box(ax, xy, wh, text, fc="#F2F4F7", ec="#4F5B66", size=8.2, weight="normal"):
    patch = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle="round,pad=0.03,rounding_size=0.08",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.15,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + wh[0] / 2,
        xy[1] + wh[1] / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        weight=weight,
        linespacing=1.22,
    )


def arrow(ax, start, end, color="#4F5B66"):
    if abs(start[0] - end[0]) > 1e-6 and abs(start[1] - end[1]) > 1e-6:
        poly_arrow(ax, [start, (end[0], start[1]), end], color=color)
        return
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, linewidth=0.72, color=color))


def dashed_arrow(ax, start, end, color="#6B7280"):
    if abs(start[0] - end[0]) > 1e-6 and abs(start[1] - end[1]) > 1e-6:
        poly_arrow(ax, [start, (end[0], start[1]), end], color=color, dashed=True)
        return
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.72,
            linestyle=(0, (4, 3)),
            color=color,
        )
    )


def poly_arrow(ax, points, color="#4F5B66", dashed=False, lw=0.72, ms=9):
    """Draw a clean orthogonal connector; arrow head is only on final segment."""
    if len(points) < 2:
        return
    linestyle = (0, (4, 3)) if dashed else "solid"
    for p0, p1 in zip(points[:-2], points[1:-1]):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=lw, linestyle=linestyle, solid_capstyle="round")
    ax.add_patch(
        FancyArrowPatch(
            points[-2],
            points[-1],
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            linestyle=linestyle,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def card(ax, xy, wh, title, body="", fc="#F8FAFC", ec="#D6DBE3", accent=None, size=6.8, weight="normal"):
    patch = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle="round,pad=0.035,rounding_size=0.09",
        facecolor=fc,
        edgecolor=ec,
        linewidth=0.9,
    )
    ax.add_patch(patch)
    if accent:
        ax.add_patch(
            Rectangle(
                (xy[0], xy[1] + wh[1] - 0.08),
                wh[0],
                0.08,
                facecolor=accent,
                edgecolor="none",
                alpha=0.95,
            )
        )
    if body:
        ax.text(
            xy[0] + wh[0] / 2,
            xy[1] + wh[1] * 0.61,
            title,
            ha="center",
            va="center",
            fontsize=size,
            weight=weight,
            color=PALETTE["text"],
            linespacing=1.18,
        )
        ax.text(
            xy[0] + wh[0] / 2,
            xy[1] + wh[1] * 0.27,
            body,
            ha="center",
            va="center",
            fontsize=size - 1.2,
            color=PALETTE["muted"],
            linespacing=1.15,
        )
    else:
        ax.text(
            xy[0] + wh[0] / 2,
            xy[1] + wh[1] / 2,
            title,
            ha="center",
            va="center",
            fontsize=size,
            weight=weight,
            color=PALETTE["text"],
            linespacing=1.18,
        )


def section_label(ax, label, x, y, color=PALETTE["muted"]):
    ax.text(x, y, label, ha="left", va="center", fontsize=6.4, weight="bold", color=color)
    ax.plot([x, x + 0.75], [y - 0.16, y - 0.16], color=color, linewidth=0.9, solid_capstyle="round")


def chip(ax, x, y, text, fc, color, width=None):
    width = width or max(0.58, 0.18 * len(text))
    patch = FancyBboxPatch(
        (x, y),
        width,
        0.34,
        boxstyle="round,pad=0.025,rounding_size=0.16",
        facecolor=fc,
        edgecolor="none",
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + 0.17, text, ha="center", va="center", fontsize=5.7, color=color)
    return width


def page_title(ax, title, subtitle, *, x, y, width):
    ax.text(x + width / 2, y, title, ha="center", va="center", fontsize=10.9, weight="normal", color=NAVY)
    ax.text(x + width / 2, y - 0.33, subtitle, ha="center", va="center", fontsize=6.7, color=SLATE)


def panel_frame(ax, xy, wh, label, title, *, fc=PANEL, ec="#D9E0EA", label_color=NAVY):
    patch = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle="round,pad=0.035,rounding_size=0.10",
        facecolor=fc,
        edgecolor=ec,
        linewidth=0.9,
    )
    ax.add_patch(patch)
    ax.add_patch(
        FancyBboxPatch(
            (xy[0] + 0.16, xy[1] + wh[1] - 0.44),
            0.42,
            0.26,
            boxstyle="round,pad=0.025,rounding_size=0.05",
            facecolor=label_color,
            edgecolor="none",
        )
    )
    ax.text(xy[0] + 0.37, xy[1] + wh[1] - 0.31, label, ha="center", va="center", fontsize=6.4, color="white", weight="bold")
    ax.text(xy[0] + 0.72, xy[1] + wh[1] - 0.31, title, ha="left", va="center", fontsize=7.0, color=NAVY, weight="bold")


def module_card(ax, xy, wh, title, body="", *, accent=BLUE, fc="#FFFFFF", ec="#CBD5E1", size=6.5, title_y=0.62):
    patch = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle="round,pad=0.035,rounding_size=0.08",
        facecolor=fc,
        edgecolor=ec,
        linewidth=0.95,
    )
    ax.add_patch(patch)
    ax.add_patch(Rectangle((xy[0], xy[1] + wh[1] - 0.08), wh[0], 0.08, facecolor=accent, edgecolor="none"))
    ax.text(
        xy[0] + wh[0] / 2,
        xy[1] + wh[1] * title_y,
        title,
        ha="center",
        va="center",
        fontsize=size,
        color=NAVY,
        weight="bold",
        linespacing=1.15,
    )
    if body:
        ax.text(
            xy[0] + wh[0] / 2,
            xy[1] + wh[1] * 0.25,
            body,
            ha="center",
            va="center",
            fontsize=size - 1.15,
            color=SLATE,
            linespacing=1.13,
        )


def input_module(ax, xy, wh, title, body, *, accent, mode):
    module_card(ax, xy, wh, title, "", accent=accent, size=6.0, title_y=0.66)
    mini_texture(ax, (xy[0] + 0.15, xy[1] + wh[1] * 0.18), (0.36, wh[1] * 0.30), mode=mode)
    ax.text(
        xy[0] + 0.58,
        xy[1] + wh[1] * 0.31,
        body,
        ha="left",
        va="center",
        fontsize=5.35,
        color=SLATE,
    )


def side_band(ax, xy, wh, text, color=NAVY):
    ax.add_patch(
        FancyBboxPatch(
            xy,
            wh[0],
            wh[1],
            boxstyle="round,pad=0.015,rounding_size=0.05",
            facecolor=color,
            edgecolor="none",
        )
    )
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, text, ha="center", va="center", fontsize=6.0, color="white", weight="bold", rotation=90)


def mini_texture(ax, xy, wh, *, mode="signal", seed=0, alpha=1.0):
    x, y = xy
    w, h = wh
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#EEF3F8", edgecolor="#D5DEE8", linewidth=0.45, alpha=alpha))
    if mode == "coal":
        colors = ["#CED9CF", "#96AF9E", "#6D8A76", "#E2E7DF"]
        pts = [
            [(x, y + h * 0.08), (x + w * 0.32, y + h * 0.28), (x + w * 0.18, y + h * 0.92), (x, y + h)],
            [(x + w * 0.18, y), (x + w * 0.70, y + h * 0.18), (x + w * 0.58, y + h * 0.72), (x + w * 0.32, y + h * 0.42)],
            [(x + w * 0.64, y + h * 0.04), (x + w, y + h * 0.22), (x + w, y + h), (x + w * 0.68, y + h * 0.75)],
        ]
        for i, poly in enumerate(pts):
            ax.add_patch(Polygon(poly, closed=True, facecolor=colors[(i + seed) % len(colors)], edgecolor="white", linewidth=0.35, alpha=0.95))
    elif mode == "heat":
        colors = ["#EAF6F5", "#B9E1DB", "#69B8AD", "#248A80"]
        cols, rows = 5, 3
        for i in range(cols):
            for j in range(rows):
                c = colors[(i + 2 * j + seed) % len(colors)]
                ax.add_patch(Rectangle((x + i * w / cols, y + j * h / rows), w / cols - 0.01, h / rows - 0.01, facecolor=c, edgecolor="white", linewidth=0.25))
    elif mode == "serial":
        for k in range(5):
            yy = y + h * (0.18 + 0.15 * k)
            ax.plot([x + w * 0.12, x + w * 0.88], [yy, yy + h * 0.05 * ((k % 2) * 2 - 1)], color=[BLUE, TEAL, ORANGE][k % 3], linewidth=0.8, alpha=0.9)
    else:
        for k in range(6):
            xx = x + w * (0.08 + 0.14 * k)
            ax.plot([xx, xx + w * 0.20], [y + h * 0.15, y + h * 0.82], color="#9BA7B5", linewidth=0.55, alpha=0.65)


def missing_mark(ax, center, radius=0.17, color=RED):
    ax.add_patch(Circle(center, radius, facecolor="#FFF5F5", edgecolor=color, linewidth=1.0))
    ax.plot([center[0] - radius * 0.62, center[0] + radius * 0.62], [center[1] + radius * 0.62, center[1] - radius * 0.62], color=color, linewidth=1.0)


def state_card(ax, xy, wh, title, subtitle, *, accent, missing=False, degraded=False, combined=False):
    module_card(ax, xy, wh, title, "", accent=accent, fc="#FFFFFF", ec=accent, size=6.0, title_y=0.78)
    tile_y = xy[1] + 0.28
    mini_texture(ax, (xy[0] + 0.26, tile_y), (0.58, 0.34), mode="coal", seed=int(xy[0] * 10))
    ax.text(xy[0] + 1.02, tile_y + 0.17, "+", ha="center", va="center", fontsize=8.2, color=NAVY, weight="bold")
    if missing:
        missing_mark(ax, (xy[0] + 1.44, tile_y + 0.17), radius=0.15)
    else:
        mini_texture(ax, (xy[0] + 1.22, tile_y), (0.58, 0.34), mode="heat" if degraded else "signal", seed=int(xy[0] * 7))
    if combined:
        missing_mark(ax, (xy[0] + 2.02, tile_y + 0.17), radius=0.12, color=PURPLE)
    ax.text(xy[0] + wh[0] / 2, xy[1] + 0.16, subtitle, ha="center", va="center", fontsize=5.4, color=SLATE)


def _legacy_fig1() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.3))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    page_title(
        ax,
        "X-RAY BELT WEIGHING",
        "观测链路、漂移来源与现场可观测边界",
        x=0.45,
        y=8.08,
        width=15.10,
    )

    panel_frame(ax, (0.50, 2.72), (9.90, 4.72), "a", "物理透射测量链", fc="#F7FAFD", label_color=BLUE)
    panel_frame(ax, (10.75, 2.72), (4.75, 4.72), "b", "漂移与安装偏差", fc="#FFFCF7", label_color=ORANGE)
    panel_frame(ax, (0.50, 0.42), (15.00, 1.78), "c", "现场状态与可观测边界", fc="#F8FAFC", label_color=TEAL)

    # Panel a: strictly left-to-right block diagram.
    input_specs = [
        ((0.96, 6.15), "X射线串口", "U/I/C/T/接收值", BLUE, "serial"),
        ((0.96, 5.12), "环境工况", "温湿度/速度", TEAL, "signal"),
        ((0.96, 4.09), "煤种接口", "煤种/灰分标记", ORANGE, "heat"),
    ]
    for xy, title, body, accent, mode in input_specs:
        input_module(ax, xy, (1.78, 0.72), title, body, accent=accent, mode=mode)
        poly_arrow(ax, [(2.74, xy[1] + 0.36), (3.10, xy[1] + 0.36), (3.10, 5.48), (3.43, 5.48)], "#6B7786", lw=0.95)
    module_card(ax, (3.43, 5.02), (1.46, 0.92), "多源观测", "X(t-L+1:t)", accent=SLATE, fc="#FFFFFF", size=6.3)
    module_card(ax, (5.32, 5.02), (1.54, 0.92), "空皮带基准", "I0 在线更新", accent=TEAL, fc="#FFFFFF", size=6.2)
    module_card(ax, (7.26, 5.02), (1.66, 0.92), "衰减特征", "A=ln(I0/I)", accent=TEAL, fc="#FFFFFF", size=6.2)
    module_card(ax, (7.26, 3.56), (1.66, 0.88), "计量目标", "q 与累计M", accent=GREEN, fc="#FFFFFF", size=6.2)
    for s, e in [((4.89, 5.48), (5.32, 5.48)), ((6.86, 5.48), (7.26, 5.48)), ((8.09, 5.02), (8.09, 4.44))]:
        arrow(ax, s, e, "#657384")

    # Compact transmission block avoids diagonal visual dominance.
    ax.add_patch(
        FancyBboxPatch(
            (3.56, 3.12),
            3.26,
            1.26,
            boxstyle="round,pad=0.035,rounding_size=0.08",
            facecolor="#FFFFFF",
            edgecolor="#C9D3DF",
            linewidth=0.95,
        )
    )
    ax.add_patch(Rectangle((3.56, 4.30), 3.26, 0.08, facecolor=BLUE, edgecolor="none"))
    ax.text(5.19, 4.06, "透射测量单元", ha="center", va="center", fontsize=6.3, color=NAVY, weight="bold")
    module_card(ax, (3.74, 3.35), (0.84, 0.42), "源端", "I0,U,I,T", accent=BLUE, size=5.0)
    module_card(ax, (5.80, 3.35), (0.84, 0.42), "接收端", "I,基准", accent=BLUE, size=5.0)
    ax.add_patch(Rectangle((4.78, 3.37), 0.80, 0.10, facecolor="#D8DDE6", edgecolor="#4B5563", linewidth=0.55))
    ax.add_patch(Polygon([(4.80, 3.47), (5.00, 3.78), (5.35, 3.86), (5.58, 3.47)], facecolor="#DCE9DF", edgecolor="#6D8A76", linewidth=0.65))
    ax.plot([4.58, 5.80], [3.56, 3.56], color=BLUE, linewidth=0.9)
    ax.text(5.19, 3.94, "煤层改变透射强度", ha="center", va="center", fontsize=5.3, color=GREEN)
    poly_arrow(ax, [(4.17, 5.02), (4.17, 4.38)], TEAL, lw=0.95)
    poly_arrow(ax, [(6.14, 4.38), (6.14, 5.02)], TEAL, lw=0.95)

    # Panel b: independent source boxes feed into a shared bias bus.
    drift_specs = [
        ((11.18, 6.17), "设备漂移", "源端老化\n接收端基准变化", BLUE),
        ((13.28, 6.17), "环境扰动", "温湿度\n电源波动", RED),
        ((11.18, 4.82), "安装差异", "源-探距离\n角度/皮带姿态", ORANGE),
        ((13.28, 4.82), "煤质未完全观测", "水分/粒度缺失\n灰分部分缺失", TEAL),
    ]
    bus_y = 4.18
    for xy, title, body, accent in drift_specs:
        module_card(ax, xy, (1.56, 0.82), title, body, accent=accent, size=5.8)
        poly_arrow(ax, [(xy[0] + 0.78, xy[1]), (xy[0] + 0.78, bus_y)], "#9D645B", lw=0.85)
    ax.plot([11.96, 14.06], [bus_y, bus_y], color="#9D645B", linewidth=0.9)
    module_card(ax, (12.05, 3.08), (1.90, 0.78), "固定标定映射漂移", "f(x) 随域变化失准", accent=RED, fc="#FFF6F4", ec="#D7A19A", size=6.0)
    poly_arrow(ax, [(13.00, bus_y), (13.00, 3.86)], "#9D645B", lw=0.95)
    poly_arrow(ax, [(8.92, 3.98), (10.55, 3.98), (10.55, 3.47), (12.05, 3.47)], RED, dashed=True, lw=0.95)
    ax.text(10.56, 4.13, "跨域影响", fontsize=5.4, color=RED, ha="center", va="center")

    # Panel c: evenly spaced state cards with no icon overlap.
    state_specs = [
        ((1.10, 0.72), "完整可观测", "串口+环境+煤质", TEAL, False, False, False),
        ((4.70, 0.72), "灰分缺失", "ash缺失标记", RED, True, False, False),
        ((8.30, 0.72), "隐藏扰动", "水分/粒度不可观测", ORANGE, False, True, False),
        ((11.90, 0.72), "组合漂移", "安装+环境+缺失", PURPLE, True, True, True),
    ]
    for xy, title, subtitle, accent, miss, degraded, combined in state_specs:
        state_card(ax, xy, (2.68, 0.90), title, subtitle, accent=accent, missing=miss, degraded=degraded, combined=combined)

    ax.text(0.78, 0.18, "蓝/绿为现场可直接观测变量；红/橙/紫表示缺失、隐藏扰动或域漂移边界。", fontsize=5.9, color=SLATE, ha="left")
    save_figure(fig, OUT / "fig1_system_error_sources", png_dpi=450, tiff_dpi=450, pdf=False)
    plt.close(fig)


def _legacy_fig2() -> None:
    fig, ax = plt.subplots(figsize=(10.4, 5.45))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.55)
    ax.axis("off")

    page_title(
        ax,
        "PG-TSC",
        "Physics-guided Transferable Self-Calibration Framework",
        x=0.45,
        y=8.05,
        width=15.10,
    )

    bands = [
        ((0.55, 6.28), (14.90, 1.08), "#FFF7EC", "OFFLINE", "离线初始化"),
        ((0.55, 3.02), (14.90, 2.72), "#F7FAFD", "ONLINE", "在线推理"),
        ((0.55, 0.78), (14.90, 1.72), "#FAF8FC", "FIELD", "现场闭环"),
    ]
    for xy, wh, fc, band, label in bands:
        ax.add_patch(FancyBboxPatch(xy, wh[0], wh[1], boxstyle="round,pad=0.035,rounding_size=0.10", facecolor=fc, edgecolor="#DDE2EA", linewidth=0.85))
        side_band(ax, (xy[0] + 0.10, xy[1] + 0.14), (0.48, wh[1] - 0.28), band)
        ax.text(xy[0] + 0.78, xy[1] + wh[1] - 0.14, label, ha="left", va="center", fontsize=6.3, color=SLATE, weight="bold")

    # Offline row.
    module_card(ax, (3.15, 6.52), (2.02, 0.56), "高覆盖仿真器", "设备/安装/环境/煤质域", accent=ORANGE, size=6.1)
    module_card(ax, (5.88, 6.52), (2.02, 0.56), "仿真预训练", "初始化 θ_sim", accent=ORANGE, size=6.1)
    module_card(ax, (8.62, 6.52), (2.16, 0.56), "通用扰动规律", "源域参数先验", accent=TEAL, size=6.1)
    dashed_arrow(ax, (5.17, 6.80), (5.88, 6.80), ORANGE)
    dashed_arrow(ax, (7.90, 6.80), (8.62, 6.80), ORANGE)
    poly_arrow(ax, [(9.70, 6.52), (9.70, 6.02), (7.98, 6.02), (7.98, 5.00)], ORANGE, dashed=True, lw=0.95)
    ax.text(7.98, 6.14, "初始化参数", fontsize=5.5, color=ORANGE, ha="center", va="center")

    # Online inputs.
    online_inputs = [
        ((1.24, 4.83), "X射线串口", "U/I/C/T/接收值", BLUE, "serial"),
        ((1.24, 3.93), "环境工况", "温湿度/速度/煤种", TEAL, "signal"),
        ((1.24, 3.03), "煤质接口", "灰分+缺失标记", ORANGE, "heat"),
    ]
    for xy, title, body, accent, mode in online_inputs:
        input_module(ax, xy, (1.86, 0.64), title, body, accent=accent, mode=mode)
        poly_arrow(ax, [(3.10, xy[1] + 0.32), (3.44, xy[1] + 0.32), (3.44, 4.40), (3.70, 4.40)], "#657384", lw=0.95)
    module_card(ax, (3.70, 3.82), (1.26, 1.16), "现场观测", "X_t", accent=SLATE, fc="#FFFFFF", size=6.1)

    # Core PG-TSC blocks.
    stage_specs = [
        ((5.35, 3.70), "1", "物理特征层", "I0在线更新\nA=ln(I0/I)\nq_phy", TEAL, "#E8F3EE"),
        ((7.20, 3.70), "2", "时序残差层", "缺失感知编码\nTCN/统计特征\nΔq", BLUE, "#EAF0FA"),
        ((9.05, 3.70), "3", "设备适配层", "少样本a,b校准\n累计质量一致性\nq_final", ORANGE, "#FFF0D9"),
        ((10.90, 3.70), "4", "维护边界层", "串口异常防护\n覆盖外检测\n风险触发", PURPLE, "#F2E9F5"),
    ]
    for xy, badge, title, body, accent, fc in stage_specs:
        module_card(ax, xy, (1.56, 1.28), title, body, accent=accent, fc=fc, ec="#C9D3DF", size=5.85)
        ax.add_patch(Circle((xy[0] + 0.18, xy[1] + 1.05), 0.14, facecolor=accent, edgecolor="none"))
        ax.text(xy[0] + 0.18, xy[1] + 1.05, badge, ha="center", va="center", fontsize=5.7, color="white", weight="bold")
    for s, e in [((4.96, 4.40), (5.35, 4.40)), ((6.91, 4.40), (7.20, 4.40)), ((8.76, 4.40), (9.05, 4.40)), ((10.61, 4.40), (10.90, 4.40))]:
        arrow(ax, s, e, "#657384")

    # Outputs.
    module_card(ax, (13.12, 4.88), (1.34, 0.50), "质量流量", "q_t", accent=GREEN, fc="#E8F3EE", size=5.9)
    module_card(ax, (13.12, 4.18), (1.34, 0.50), "累计质量", "M_T", accent=GREEN, fc="#E8F3EE", size=5.9)
    module_card(ax, (13.12, 3.48), (1.34, 0.50), "复核提示", "追加标定", accent=PURPLE, fc="#F2E9F5", size=5.7)
    for y in [5.13, 4.43, 3.73]:
        poly_arrow(ax, [(12.46, 4.40), (12.72, 4.40), (12.72, y), (13.12, y)], "#657384", lw=0.95)

    # Field loop.
    field_specs = [
        ((1.24, 1.30), "标定片段", "参考累计质量", RED),
        ((3.36, 1.30), "主动标定选样", "覆盖负荷与漂移状态", PURPLE),
        ((8.92, 1.30), "安装少样本校准", "比例/偏置+累计约束", ORANGE),
        ((10.90, 1.30), "风险触发维护", "异常/覆盖外/漂移复核", PURPLE),
        ((13.04, 1.30), "运行期更新", "追加样本+人工复核边界", RED),
    ]
    for xy, title, body, accent in field_specs:
        module_card(ax, xy, (1.76, 0.62), title, body, accent=accent, size=5.95)
    dashed_arrow(ax, (3.00, 1.61), (3.36, 1.61), RED)
    poly_arrow(ax, [(5.12, 1.61), (7.08, 1.61), (7.08, 1.61), (8.92, 1.61)], PURPLE, dashed=True, lw=0.95)
    dashed_arrow(ax, (10.68, 1.61), (10.90, 1.61), PURPLE)
    dashed_arrow(ax, (12.66, 1.61), (13.04, 1.61), PURPLE)
    poly_arrow(ax, [(9.80, 1.92), (9.80, 3.70)], ORANGE, dashed=True, lw=0.95)
    poly_arrow(ax, [(11.78, 1.92), (11.78, 3.70)], PURPLE, dashed=True, lw=0.95)
    poly_arrow(ax, [(13.79, 3.48), (13.79, 1.92)], PURPLE, dashed=True, lw=0.95)

    # Legend strip.
    ax.plot([1.05, 1.55], [0.42, 0.42], color="#657384", linewidth=1.0)
    arrow(ax, (1.55, 0.42), (1.90, 0.42), "#657384")
    ax.text(2.02, 0.42, "在线推理", va="center", fontsize=5.8, color=SLATE)
    ax.plot([3.22, 3.98], [0.42, 0.42], color=ORANGE, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(4.10, 0.42, "预训练/适配/维护反馈", va="center", fontsize=5.8, color=SLATE)
    ax.text(8.45, 0.42, "输出同时约束瞬时质量流量和累计质量误差。", va="center", fontsize=5.8, color=SLATE)
    save_figure(fig, OUT / "fig2_pgtcs_framework", png_dpi=450, tiff_dpi=450, pdf=False)
    plt.close(fig)


def simple_frame(ax, xy, wh, label, title, *, fc="#FFFFFF", ec="#DDE3EA", label_color=NAVY):
    patch = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle="round,pad=0.025,rounding_size=0.035",
        facecolor=fc,
        edgecolor=ec,
        linewidth=0.56,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + 0.22,
        xy[1] + wh[1] - 0.25,
        f"{label.upper()}  {title}",
        ha="left",
        va="center",
        fontsize=6.7,
        color=label_color,
        weight="normal",
    )


def simple_box(ax, xy, wh, title, body="", *, accent=BLUE, fc="#FFFFFF", ec="#CBD5E1", size=9.2):
    title_size = max(size * 0.74, 6.2)
    body_size = max((size - 0.7) * 0.74, 5.7)
    patch = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle="round,pad=0.025,rounding_size=0.032",
        facecolor=fc,
        edgecolor=ec,
        linewidth=0.56,
    )
    ax.add_patch(patch)
    if body:
        ax.text(
            xy[0] + wh[0] / 2,
            xy[1] + wh[1] * 0.64,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            color=NAVY,
            weight="normal",
        )
        ax.text(
            xy[0] + wh[0] / 2,
            xy[1] + wh[1] * 0.30,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color=SLATE,
            linespacing=1.10,
        )
    else:
        ax.text(
            xy[0] + wh[0] / 2,
            xy[1] + wh[1] / 2,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            color=NAVY,
            weight="normal",
            linespacing=1.08,
        )


def fig1() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.65))
    ax.set_xlim(0, 15.0)
    ax.set_ylim(0, 9.5)
    ax.axis("off")

    page_title(ax, "X-RAY BELT WEIGHING", "连续称重跨域失准来源与可观测边界", x=0, y=9.15, width=15.0)

    def local_card(xy, wh, title, body="", *, fc="#FFFFFF", ec="#D5DFE8", color=NAVY, size=6.0, lw=0.58):
        patch = FancyBboxPatch(
            xy,
            wh[0],
            wh[1],
            boxstyle="round,pad=0.026,rounding_size=0.045",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
        )
        ax.add_patch(patch)
        if body:
            ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] * 0.66, title, ha="center", va="center", fontsize=size, color=color, weight="normal")
            ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] * 0.30, body, ha="center", va="center", fontsize=size - 0.95, color=SLATE, linespacing=1.10)
        else:
            ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, title, ha="center", va="center", fontsize=size, color=color, linespacing=1.12)

    def pill(x, y, text, *, color=BLUE, fc="#F6FAFE", w=1.02):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                0.30,
                boxstyle="round,pad=0.020,rounding_size=0.15",
                facecolor=fc,
                edgecolor=color,
                linewidth=0.55,
            )
        )
        ax.text(x + w / 2, y + 0.15, text, ha="center", va="center", fontsize=5.05, color=color)

    def arrow_label(text, x, y, color):
        ax.text(x, y, text, ha="center", va="center", fontsize=4.9, color=color)

    # A. Physical measuring scene.
    simple_frame(ax, (0.45, 4.55), (8.30, 4.05), "A", "X射线透射称重链路", fc="#F8FAFD", label_color=BLUE)
    ax.add_patch(Rectangle((3.48, 5.32), 0.55, 2.04, facecolor="#F9E6C8", edgecolor="none", alpha=0.32, zorder=1))
    ax.plot([3.75, 3.75], [5.32, 7.36], color=ORANGE, linewidth=0.76, alpha=0.72, zorder=2)
    ax.add_patch(
        FancyBboxPatch(
            (1.92, 5.48),
            3.88,
            0.36,
            boxstyle="round,pad=0.010,rounding_size=0.05",
            facecolor="#EAF0F6",
            edgecolor="#C6D0DA",
            linewidth=0.56,
            zorder=3,
        )
    )
    ax.plot([2.10, 5.62], [5.80, 5.80], color="#8EA1B5", linewidth=0.72, zorder=4)
    for cx in [2.15, 5.55]:
        ax.add_patch(Circle((cx, 5.42), 0.18, facecolor="#FFFFFF", edgecolor="#B6C2CF", linewidth=0.56, zorder=3))
    coal_pieces = [
        [(2.22, 5.86), (2.52, 6.08), (2.44, 6.36), (2.05, 6.22)],
        [(2.62, 5.84), (3.05, 5.98), (3.02, 6.34), (2.66, 6.30)],
        [(3.12, 5.86), (3.52, 6.12), (3.42, 6.48), (3.00, 6.32)],
        [(3.62, 5.82), (4.05, 6.02), (4.02, 6.40), (3.58, 6.34)],
        [(4.15, 5.85), (4.58, 6.07), (4.48, 6.45), (4.08, 6.24)],
        [(4.68, 5.82), (5.18, 6.02), (5.08, 6.36), (4.62, 6.28)],
    ]
    for i, pts in enumerate(coal_pieces):
        fc = ["#4B5563", "#616B77", "#525B66"][i % 3]
        ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor="#374151", linewidth=0.35, alpha=0.92, zorder=5))
    ax.text(3.86, 6.66, "煤流/皮带", ha="center", va="center", fontsize=5.1, color=SLATE)
    local_card((3.14, 7.46), (1.22, 0.42), "接收端", fc="#EEF6FF", ec="#C8D9E8", color=BLUE, size=5.2)
    local_card((3.10, 4.84), (1.30, 0.42), "X射线源", fc="#FFF3E7", ec="#E8D0B9", color=ORANGE, size=5.2)
    local_card((6.08, 6.63), (2.15, 0.72), "串口与现场输入", "管压/管流/温度/射线值\n环境温湿度/煤种", fc="#FFFFFF", ec="#CBD5E1", color=NAVY, size=5.0)
    local_card((6.08, 5.10), (2.15, 0.62), "衰减物理特征", r"$A=\ln(I_0/I)$", fc="#F1FBF8", ec="#CFE1D9", color=TEAL, size=5.35)
    poly_arrow(ax, [(4.36, 7.67), (5.82, 7.67), (5.82, 6.99), (6.08, 6.99)], BLUE, lw=0.58, ms=6.5)
    poly_arrow(ax, [(4.40, 5.05), (5.82, 5.05), (5.82, 6.63), (6.08, 6.63)], ORANGE, lw=0.58, ms=6.5)
    poly_arrow(ax, [(7.15, 6.63), (7.15, 5.72)], TEAL, lw=0.58, ms=6.5)
    arrow_label("接收值 I", 5.18, 7.52, BLUE)
    arrow_label("U, I0, T", 5.12, 4.82, ORANGE)

    # B. Domain shift mechanism.
    simple_frame(ax, (9.02, 4.55), (5.50, 4.05), "B", "跨域扰动导致固定标定失配", fc="#FFFCF7", label_color=ORANGE)
    local_card((9.52, 7.42), (1.44, 0.58), "源域标定", r"$D_s,\ f_s(x)$", fc="#F3F9FE", ec="#C8D9E8", color=BLUE, size=5.45)
    local_card((12.38, 7.42), (1.44, 0.58), "目标域运行", r"$D_t,\ X_t$", fc="#F1FBF8", ec="#CFE1D9", color=TEAL, size=5.45)
    dashed_arrow(ax, (10.96, 7.71), (12.38, 7.71), RED)
    local_card((10.82, 5.03), (1.90, 0.66), "映射失准", r"$f_s \ne f_t$", fc="#FFF6F4", ec="#E8C9C5", color=RED, size=5.65)
    perturbations = [
        (9.42, 6.45, "环境漂移", "温湿度/源端温度", TEAL, "#F1FBF8"),
        (11.08, 6.45, "安装差异", "几何/距离/对准", ORANGE, "#FFF8EE"),
        (12.74, 6.45, "煤质变化", "灰分缺失/水分粒度", PURPLE, "#F8F4FC"),
    ]
    for x, y, title, body, color, fc in perturbations:
        local_card((x, y), (1.20, 0.58), title, body, fc=fc, ec="#D5DFE8", color=color, size=4.85)
        poly_arrow(ax, [(x + 0.60, y), (x + 0.60, 6.02), (11.77, 6.02), (11.77, 5.69)], color, dashed=True, lw=0.50, ms=5.8)
    ax.text(11.77, 4.80, "同一输入映射在目标域偏移", ha="center", va="center", fontsize=5.05, color=SLATE)

    # C. Observable boundary and self-calibration requirement.
    simple_frame(ax, (0.45, 0.48), (14.07, 3.62), "C", "可观测边界与自校准需求", fc="#FAFCFE", label_color=TEAL)
    ax.text(0.88, 3.55, "直接观测", ha="left", va="center", fontsize=6.0, color=TEAL, weight="normal")
    for i, label in enumerate(["管压", "管流", "温度", "接收值", "环境温湿度", "煤种"]):
        pill(0.88 + (i % 3) * 1.02, 3.04 - (i // 3) * 0.42, label, color=TEAL, fc="#F1FBF8", w=0.90 if label != "环境温湿度" else 1.02)
    ax.text(4.25, 3.55, "部分缺失", ha="left", va="center", fontsize=6.0, color=ORANGE, weight="normal")
    pill(4.25, 3.04, "灰分标签", color=ORANGE, fc="#FFF8EE", w=1.10)
    ax.text(6.05, 3.55, "不可观测", ha="left", va="center", fontsize=6.0, color=RED, weight="normal")
    pill(6.05, 3.04, "水分", color=RED, fc="#FFF6F4", w=0.78)
    pill(6.92, 3.04, "粒度", color=RED, fc="#FFF6F4", w=0.78)
    ax.plot([8.22, 8.22], [1.18, 3.55], color="#D8E0EA", linewidth=0.70, linestyle=(0, (3, 2)))
    ax.text(8.38, 3.55, "自校准模型需要同时处理", ha="left", va="center", fontsize=6.0, color=NAVY, weight="normal")
    need_specs = [
        ((8.55, 2.70), "物理先验", "衰减特征"),
        ((10.00, 2.70), "迁移初始化", "仿真预训练"),
        ((11.45, 2.70), "安装校准", "少样本"),
        ((12.90, 2.70), "灰分缺失", "标记机制"),
        ((8.55, 1.78), "累计一致性", "约束M"),
        ((10.00, 1.78), "风险触发", "追加标定"),
        ((11.45, 1.78), "运行维护", "边界复核"),
        ((12.90, 1.78), "空载基准", "在线更新"),
    ]
    for xy, title, body in need_specs:
        local_card(xy, (1.14, 0.56), title, body, fc="#FFFFFF", ec="#D6DEE8", color=NAVY, size=4.75)
    poly_arrow(ax, [(7.70, 3.19), (8.22, 3.19)], SLATE, lw=0.58, ms=6.4)
    save_figure(fig, OUT / "fig1_system_error_sources", png_dpi=600, tiff_dpi=600, pdf=False)
    plt.close(fig)


def fig2() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.15))
    ax.set_xlim(0, 15.4)
    ax.set_ylim(0, 8.8)
    ax.axis("off")

    page_title(ax, "PG-TSC", "物理约束迁移自校准框架", x=0, y=8.42, width=15.4)

    stage_w = 1.96
    stage_h = 0.66

    def stage_box(x, y, title, body, *, fc, ec="#D7E0EA", color=NAVY, w=stage_w, h=stage_h, size=5.7):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.026,rounding_size=0.045", facecolor=fc, edgecolor=ec, linewidth=0.58)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=size, color=color, weight="normal")
        ax.text(x + w / 2, y + h * 0.29, body, ha="center", va="center", fontsize=size - 0.95, color=SLATE, linespacing=1.10)

    def lane_label(y, text, color):
        ax.text(0.76, y + 0.33, text, fontsize=6.0, color=color, ha="left", va="center", weight="normal")
        ax.plot([0.72, 1.06], [y + 0.07, y + 0.07], color=color, linewidth=0.78)

    simple_frame(ax, (0.45, 0.78), (14.50, 6.88), "A", "仪器级自校准框架", fc="#FAFCFE", label_color=BLUE)
    xcols = [2.38, 5.00, 7.62, 10.24, 12.86]

    lane_label(6.28, "离线源域初始化", ORANGE)
    offline = [
        ("高覆盖仿真器", "源域扰动/安装差异", "#FFF8EE", ORANGE),
        ("物理一致预训练", "衰减规律初始化", "#FFF8EE", ORANGE),
        ("迁移初值", r"$\theta_{\rm sim}$", "#F1FBF8", TEAL),
    ]
    for x, (t, b, fc, c) in zip(xcols[:3], offline):
        stage_box(x, 6.05, t, b, fc=fc, color=c, size=5.4)
    dashed_arrow(ax, (xcols[0] + stage_w + 0.12, 6.38), (xcols[1] - 0.12, 6.38), ORANGE)
    dashed_arrow(ax, (xcols[1] + stage_w + 0.12, 6.38), (xcols[2] - 0.12, 6.38), ORANGE)

    lane_label(4.40, "在线物理约束推理", BLUE)
    online = [
        ("现场观测", "串口/环境/煤种", "#F7F9FC", BLUE),
        ("衰减物理层", r"$A,\ q_{\rm phy}$", "#F1FBF8", TEAL),
        ("时序残差层", r"$\Delta q_t$", "#F5F7FB", NAVY),
        ("设备适配层", "比例/偏置/少样本", "#FFF8EE", ORANGE),
        ("计量输出", r"$q_t,\ M,\ r_t$", "#F8F4FC", PURPLE),
    ]
    for x, (t, b, fc, c) in zip(xcols, online):
        stage_box(x, 4.17, t, b, fc=fc, color=c, size=5.35)
    for x0, x1 in zip(xcols[:-1], xcols[1:]):
        arrow(ax, (x0 + stage_w + 0.13, 4.50), (x1 - 0.13, 4.50), SLATE)

    lane_label(2.45, "安装校准与运行维护", PURPLE)
    maintenance = [
        ("代表性选样", "负荷覆盖/主动标定", "#F8F4FC", PURPLE),
        ("安装少样本校准", "设备级参数更新", "#FFF8EE", ORANGE),
        ("累计一致性", "质量目标锚定", "#F1FBF8", TEAL),
        ("风险触发", "追加标定/复核", "#FFF6F4", RED),
    ]
    for x, (t, b, fc, c) in zip(xcols[1:], maintenance):
        stage_box(x, 2.22, t, b, fc=fc, color=c, size=5.25)
    for x0, x1 in zip(xcols[1:-1], xcols[2:]):
        dashed_arrow(ax, (x0 + stage_w + 0.13, 2.55), (x1 - 0.13, 2.55), PURPLE)

    # Cross-lane links use open vertical corridors to keep the framework readable.
    poly_arrow(ax, [(8.60, 6.05), (8.60, 5.35), (8.60, 4.83)], ORANGE, dashed=True, lw=0.58, ms=6.8)
    poly_arrow(ax, [(8.60, 2.88), (8.60, 3.37), (11.22, 3.37), (11.22, 4.17)], ORANGE, dashed=True, lw=0.58, ms=6.8)
    poly_arrow(ax, [(13.84, 4.17), (13.84, 3.45), (13.84, 2.88)], RED, dashed=True, lw=0.58, ms=6.8)
    poly_arrow(ax, [(11.22, 2.88), (11.22, 4.17)], TEAL, dashed=True, lw=0.58, ms=6.8)

    ax.plot([1.28, 1.74], [1.32, 1.32], color=SLATE, linewidth=0.78)
    arrow(ax, (1.74, 1.32), (2.08, 1.32), SLATE)
    ax.text(2.22, 1.32, "在线推理", va="center", fontsize=5.5, color=SLATE)
    ax.plot([3.42, 4.10], [1.32, 1.32], color=PURPLE, linewidth=0.78, linestyle=(0, (4, 3)))
    ax.text(4.24, 1.32, "预训练/校准/维护反馈", va="center", fontsize=5.5, color=SLATE)
    ax.text(8.10, 1.32, "框架目标：同时约束瞬时质量流量、累计质量误差和运行期风险边界。", va="center", fontsize=5.45, color=SLATE)
    save_figure(fig, OUT / "fig2_pgtcs_framework", png_dpi=600, tiff_dpi=600, pdf=False)
    plt.close(fig)


def fig3() -> None:
    fig, ax = plt.subplots(figsize=(6.9, 4.35))
    ax.set_xlim(0, 14.1)
    ax.set_ylim(0, 8.4)
    ax.axis("off")

    ax.text(7.05, 8.02, "PG-TSC Residual Network", ha="center", va="center", fontsize=9.1, weight="normal", color=NAVY)
    ax.text(7.05, 7.70, "物理约束残差补偿网络结构", ha="center", va="center", fontsize=6.4, color=SLATE)

    simple_frame(ax, (0.42, 1.18), (13.25, 5.92), "A", "物理约束残差架构", fc="#FAFCFE", label_color=BLUE)

    simple_box(ax, (0.78, 4.12), (1.78, 1.02), "时序输入", r"$Z_{t-L+1:t}$", fc="#F7F9FC", ec="#D5DFE8", size=8.0)
    ax.text(1.67, 3.78, "串口/环境/煤种/物理统计", fontsize=5.6, color=SLATE, ha="center", va="center")

    simple_box(ax, (3.08, 5.20), (1.72, 0.82), "物理特征", r"$A=\ln(I_0/I)$", fc="#F1FBF8", ec="#CFE1D9", size=7.5)
    simple_box(ax, (5.18, 5.20), (1.72, 0.82), "基础估计", r"$q_{\rm phy}=F_\phi(z_t)$", fc="#F1FBF8", ec="#CFE1D9", size=7.5)
    simple_box(ax, (8.20, 5.20), (1.86, 0.82), "残差融合", r"$\hat q_t=q_{\rm phy}+\Delta q_t$", fc="#FFF8EE", ec="#E8D8C4", size=7.2)
    simple_box(ax, (10.82, 5.20), (1.74, 0.82), "设备校准", r"$q_t=\max(a\hat q_t+b,0)$", fc="#FFF6F4", ec="#E6D4D0", size=6.9)
    poly_arrow(ax, [(2.56, 4.64), (2.82, 4.64), (2.82, 5.61), (3.08, 5.61)], TEAL, lw=0.72, ms=8)
    poly_arrow(ax, [(4.80, 5.61), (5.18, 5.61)], TEAL, lw=0.72, ms=8)
    poly_arrow(ax, [(6.90, 5.61), (8.20, 5.61)], TEAL, lw=0.72, ms=8)
    poly_arrow(ax, [(10.06, 5.61), (10.82, 5.61)], ORANGE, lw=0.72, ms=8)

    simple_box(ax, (3.08, 3.08), (1.52, 0.78), "缺失感知", "mask+coal", fc="#FFF8EE", ec="#E8D8C4", size=7.2)
    tcn_blocks = [
        (5.02, "TCN-1", "d=1"),
        (6.70, "TCN-2", "d=2"),
        (8.38, "TCN-3", "d=4"),
    ]
    for x, title, body in tcn_blocks:
        simple_box(ax, (x, 3.08), (1.26, 0.78), title, body, fc="#EEF5FF", ec="#D5DFE8", size=7.2)
    simple_box(ax, (10.02, 3.08), (1.32, 0.78), "线性头", r"$\Delta q_t$", fc="#F8F4FC", ec="#DFD6E7", size=7.2)
    for x0, x1 in [(2.56, 3.08), (4.60, 5.02), (6.28, 6.70), (7.96, 8.38), (9.64, 10.02)]:
        poly_arrow(ax, [(x0, 3.47), (x1, 3.47)], SLATE, lw=0.72, ms=8)
    for x in [5.02, 6.70, 8.38]:
        ax.plot([x + 0.14, x + 1.12], [3.98, 3.98], color="#9AA7B5", linewidth=0.55)
        poly_arrow(ax, [(x + 0.14, 3.98), (x + 0.14, 4.23), (x + 1.12, 4.23), (x + 1.12, 3.98)], "#9AA7B5", lw=0.52, ms=5)
    ax.text(7.02, 4.45, "dilated causal convolution + residual skip", fontsize=5.3, color=SLATE, ha="center", va="center")
    poly_arrow(ax, [(10.68, 3.86), (10.68, 4.92), (9.13, 4.92), (9.13, 5.20)], BLUE, lw=0.72, ms=8)

    simple_box(ax, (10.82, 1.98), (1.74, 0.70), "累计质量", r"$M_T=\sum q_t\Delta t$", fc="#F1FBF8", ec="#CFE1D9", size=6.9)
    poly_arrow(ax, [(11.69, 5.20), (11.69, 2.68)], GREEN, lw=0.70, ms=8)
    ax.text(12.74, 5.60, "输出", fontsize=5.5, color=GREEN, ha="left", va="center")

    objective_box = FancyBboxPatch(
        (0.70, 0.32),
        12.70,
        0.62,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        facecolor="#FFFFFF",
        edgecolor="#DDE3EA",
        linewidth=0.56,
    )
    ax.add_patch(objective_box)
    objective = r"$\mathcal{L}=\mathcal{L}_{q}+\lambda_M\mathcal{L}_{M}+\lambda_R\|\Theta-\Theta_{\rm sim}\|_2^2$"
    ax.text(0.95, 0.63, "训练与校准目标", fontsize=5.7, color=PURPLE, ha="left", va="center")
    ax.text(2.68, 0.63, "仿真预训练 θ_sim", fontsize=5.5, color=ORANGE, ha="left", va="center")
    ax.text(4.95, 0.63, objective, fontsize=5.5, color=NAVY, ha="left", va="center")
    ax.text(10.28, 0.63, "少样本更新：a,b/线性头", fontsize=5.5, color=RED, ha="left", va="center")

    ax.text(0.58, 7.34, "主链路保留物理可解释项，TCN只学习未显式建模的时序漂移残差；设备校准层直接服务累计质量计量目标。", fontsize=5.7, color=SLATE, ha="left", va="center")
    save_figure(fig, OUT / "fig3_network_architecture", png_dpi=600, tiff_dpi=600, pdf=False)
    plt.close(fig)


def _thin_box(ax, xy, wh, title, body="", *, fc="#FFFFFF", ec="#D4DEE8", color=NAVY, size=6.1):
    patch = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle="round,pad=0.024,rounding_size=0.035",
        facecolor=fc,
        edgecolor=ec,
        linewidth=0.54,
    )
    ax.add_patch(patch)
    if body:
        ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] * 0.64, title, ha="center", va="center", fontsize=size, color=color)
        ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] * 0.30, body, ha="center", va="center", fontsize=size - 0.95, color=SLATE, linespacing=1.08)
    else:
        ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, title, ha="center", va="center", fontsize=size, color=color, linespacing=1.10)


def _heading(ax, x, y, label, text, color):
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=6.0,
        color="white",
        weight="normal",
        bbox=dict(boxstyle="round,pad=0.14,rounding_size=0.05", facecolor=color, edgecolor="none"),
    )
    ax.text(x + 0.38, y, text, ha="left", va="center", fontsize=6.5, color=color)


def _xray_scene(ax, cx, cy, scale=1.0):
    belt_w, belt_h = 3.55 * scale, 0.34 * scale
    x0 = cx - belt_w / 2
    ax.add_patch(
        FancyBboxPatch(
            (x0, cy),
            belt_w,
            belt_h,
            boxstyle="round,pad=0.006,rounding_size=0.045",
            facecolor="#EDF2F7",
            edgecolor="#B8C5D2",
            linewidth=0.54,
            zorder=3,
        )
    )
    ax.plot([x0 + 0.18 * scale, x0 + belt_w - 0.18 * scale], [cy + belt_h * 0.70, cy + belt_h * 0.70], color="#7D90A4", linewidth=0.62, zorder=4)
    for px in [x0 + 0.24 * scale, x0 + belt_w - 0.24 * scale]:
        ax.add_patch(Circle((px, cy - 0.10 * scale), 0.17 * scale, facecolor="#FFFFFF", edgecolor="#B8C5D2", linewidth=0.54, zorder=3))
    lumps = [
        [(-1.32, 0.42), (-1.08, 0.58), (-1.14, 0.86), (-1.46, 0.73)],
        [(-0.88, 0.40), (-0.46, 0.54), (-0.52, 0.88), (-0.95, 0.79)],
        [(-0.35, 0.41), (0.06, 0.57), (-0.02, 0.91), (-0.42, 0.82)],
        [(0.20, 0.40), (0.64, 0.58), (0.56, 0.88), (0.16, 0.79)],
        [(0.76, 0.41), (1.20, 0.55), (1.14, 0.82), (0.72, 0.74)],
    ]
    for i, pts in enumerate(lumps):
        xy = [(cx + px * scale, cy + py * scale) for px, py in pts]
        ax.add_patch(Polygon(xy, closed=True, facecolor=["#4B5563", "#5E6772", "#525B66"][i % 3], edgecolor="#344052", linewidth=0.34, zorder=5))
    ax.add_patch(Rectangle((cx - 0.17 * scale, cy - 0.92 * scale), 0.34 * scale, 2.35 * scale, facecolor="#F8E2C2", edgecolor="none", alpha=0.35, zorder=1))
    ax.plot([cx, cx], [cy - 0.92 * scale, cy + 1.43 * scale], color=ORANGE, linewidth=0.72, alpha=0.75, zorder=2)
    _thin_box(ax, (cx - 0.62 * scale, cy - 1.24 * scale), (1.24 * scale, 0.36 * scale), "X射线源", fc="#FFF7ED", ec="#E9D5BC", color=ORANGE, size=5.0)
    _thin_box(ax, (cx - 0.58 * scale, cy + 1.44 * scale), (1.16 * scale, 0.36 * scale), "接收端", fc="#EEF6FF", ec="#C7D8E8", color=BLUE, size=5.0)


def fig1() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 15.0)
    ax.set_ylim(0, 9.45)
    ax.axis("off")

    page_title(ax, "X-RAY BELT WEIGHING", "物理透射、跨域扰动与可观测边界", x=0, y=9.07, width=15.0)
    ax.add_patch(FancyBboxPatch((0.45, 0.52), 14.10, 7.95, boxstyle="round,pad=0.025,rounding_size=0.045", facecolor="#FBFCFE", edgecolor="#D9E2EC", linewidth=0.56))

    _heading(ax, 0.88, 8.05, "A", "仪器测量核", BLUE)
    _xray_scene(ax, 3.00, 5.95, scale=0.95)
    _thin_box(ax, (5.25, 6.82), (2.00, 0.68), "串口观测", "U / I0 / T / I", fc="#F7FAFD", ec="#CBD7E3", color=BLUE, size=5.65)
    _thin_box(ax, (5.25, 5.44), (2.00, 0.66), "衰减特征", r"$A=\ln(I_0/I)$", fc="#F1FBF8", ec="#CDE2DA", color=TEAL, size=5.65)
    poly_arrow(ax, [(3.58, 7.58), (4.72, 7.58), (4.72, 7.16), (5.25, 7.16)], BLUE, lw=0.56, ms=6.5)
    poly_arrow(ax, [(3.63, 4.96), (4.74, 4.96), (4.74, 5.77), (5.25, 5.77)], ORANGE, lw=0.56, ms=6.5)
    poly_arrow(ax, [(6.25, 6.82), (6.25, 6.10)], TEAL, lw=0.56, ms=6.5)

    _heading(ax, 8.00, 8.05, "B", "跨域失配", ORANGE)
    _thin_box(ax, (8.05, 7.14), (1.55, 0.58), "源域标定", r"$D_s,\ f_s(x)$", fc="#F2F8FE", ec="#C8D9E8", color=BLUE, size=5.45)
    _thin_box(ax, (12.15, 7.14), (1.55, 0.58), "目标域运行", r"$D_t,\ X_t$", fc="#F0FAF7", ec="#CDE1D9", color=TEAL, size=5.45)
    dashed_arrow(ax, (9.60, 7.43), (12.15, 7.43), RED)
    ax.text(10.88, 7.66, "域偏移", ha="center", va="center", fontsize=5.0, color=RED)
    _thin_box(ax, (10.12, 5.44), (1.90, 0.68), "固定映射失准", r"$f_s(x)\not\equiv f_t(x)$", fc="#FFF6F4", ec="#E7C9C4", color=RED, size=5.5)
    disturb = [
        (8.05, 6.34, "环境", "温湿度/温漂", TEAL, "#F1FBF8"),
        (9.75, 6.34, "安装", "距离/角度", ORANGE, "#FFF8EF"),
        (11.45, 6.34, "煤质", "煤种/灰分", PURPLE, "#F8F4FC"),
        (13.15, 6.34, "隐藏", "水分/粒度", RED, "#FFF6F4"),
    ]
    for x, y, t, b, c, fc in disturb:
        _thin_box(ax, (x, y), (1.18, 0.56), t, b, fc=fc, ec="#D8E0EA", color=c, size=4.9)
        poly_arrow(ax, [(x + 0.59, y), (x + 0.59, 5.12), (11.07, 5.12), (11.07, 5.44)], c, dashed=True, lw=0.48, ms=5.8)

    _heading(ax, 0.88, 3.62, "C", "可观测边界与建模约束", TEAL)
    ax.plot([8.00, 8.00], [1.06, 3.36], color="#D5DEE8", linewidth=0.64, linestyle=(0, (3, 2)))
    groups = [
        (0.88, "直接观测", TEAL, "#F1FBF8", ["管压", "管流", "温度", "射线值", "环境温湿度", "煤种"]),
        (4.35, "部分缺失", ORANGE, "#FFF8EF", ["灰分标签", "参考称重"]),
        (6.80, "不可观测", RED, "#FFF6F4", ["水分", "粒度"]),
    ]
    for x, title, c, fc, items in groups:
        ax.text(x, 3.08, title, ha="left", va="center", fontsize=5.8, color=c)
        for i, item in enumerate(items):
            w = 0.92 if len(item) <= 3 else 1.14
            if title == "不可观测":
                px = x
                py = 2.50 - i * 0.44
            else:
                px = x + (i % 2) * 1.18
                py = 2.50 - (i // 2) * 0.44
            _thin_box(ax, (px, py), (w, 0.30), item, fc=fc, ec=c, color=c, size=4.85)
    ax.text(8.35, 3.08, "自校准问题需要同时约束", ha="left", va="center", fontsize=5.8, color=NAVY)
    needs = [("物理先验", "衰减模型"), ("跨域迁移", "仿真预训练"), ("少样本校准", "安装参数"), ("累计质量", "计量目标"), ("风险触发", "追加标定"), ("在线维护", "空载基准")]
    for i, (t, b) in enumerate(needs):
        x = 8.45 + (i % 3) * 1.78
        y = 2.42 - (i // 3) * 0.72
        _thin_box(ax, (x, y), (1.46, 0.46), t, b, fc="#FFFFFF", ec="#D6DEE8", color=NAVY, size=4.95)
    save_figure(fig, OUT / "fig1_system_error_sources", png_dpi=600, tiff_dpi=600, pdf=False)
    plt.close(fig)


def fig2() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    ax.set_xlim(0, 15.4)
    ax.set_ylim(0, 8.85)
    ax.axis("off")

    page_title(ax, "PG-TSC", "物理约束迁移自校准闭环", x=0, y=8.45, width=15.4)
    ax.add_patch(FancyBboxPatch((0.48, 0.72), 14.45, 7.02, boxstyle="round,pad=0.025,rounding_size=0.045", facecolor="#FBFCFE", edgecolor="#D9E2EC", linewidth=0.56))

    _heading(ax, 0.90, 7.36, "A", "源域初始化", ORANGE)
    _thin_box(ax, (1.20, 6.28), (1.95, 0.62), "高覆盖仿真器", "设备/环境/安装域", fc="#FFF8EF", ec="#E6D5C0", color=ORANGE, size=5.45)
    _thin_box(ax, (3.86, 6.28), (1.95, 0.62), "物理一致预训练", "初始化参数", fc="#FFF8EF", ec="#E6D5C0", color=ORANGE, size=5.45)
    _thin_box(ax, (6.52, 6.28), (1.58, 0.62), "迁移初值", r"$\theta_{\rm sim}$", fc="#F1FBF8", ec="#CDE1DA", color=TEAL, size=5.45)
    dashed_arrow(ax, (3.15, 6.59), (3.86, 6.59), ORANGE)
    dashed_arrow(ax, (5.81, 6.59), (6.52, 6.59), ORANGE)

    _heading(ax, 0.90, 5.40, "B", "在线计量链路", BLUE)
    online = [
        (1.20, "现场观测", "串口/环境/煤种", "#F7FAFD", BLUE),
        (3.65, "物理特征", r"$A,\ q_{\rm phy}$", "#F1FBF8", TEAL),
        (6.10, "时序残差", r"$\Delta q_t$", "#F5F7FB", NAVY),
        (8.55, "设备校准", r"$a,b$", "#FFF8EF", ORANGE),
        (11.00, "计量输出", r"$q_t,\ M_t$", "#F8F4FC", PURPLE),
    ]
    for x, t, b, fc, c in online:
        _thin_box(ax, (x, 4.48), (1.66, 0.66), t, b, fc=fc, ec="#D5DEE8", color=c, size=5.45)
    for (x0, *_), (x1, *__) in zip(online[:-1], online[1:]):
        arrow(ax, (x0 + 1.66 + 0.16, 4.81), (x1 - 0.16, 4.81), SLATE)
    poly_arrow(ax, [(7.31, 6.28), (7.31, 5.70), (6.93, 5.70), (6.93, 5.14)], ORANGE, dashed=True, lw=0.56, ms=6.5)

    _heading(ax, 0.90, 3.04, "C", "校准维护闭环", PURPLE)
    service = [
        (3.05, "代表性选样", "载荷覆盖"),
        (5.30, "少样本标定", "更新校准层"),
        (7.55, "累计一致性", "质量锚定"),
        (9.80, "风险触发", "追加复核"),
        (12.05, "空载基准", "在线更新"),
    ]
    for x, t, b in service:
        _thin_box(ax, (x, 2.10), (1.70, 0.62), t, b, fc="#FFFFFF", ec="#D9E0EA", color=NAVY, size=5.25)
    for (x0, *_), (x1, *__) in zip(service[:-1], service[1:]):
        dashed_arrow(ax, (x0 + 1.70 + 0.13, 2.41), (x1 - 0.13, 2.41), PURPLE)
    poly_arrow(ax, [(6.15, 2.72), (6.15, 3.44), (9.38, 3.44), (9.38, 4.48)], ORANGE, dashed=True, lw=0.56, ms=6.5)
    poly_arrow(ax, [(10.65, 4.48), (10.65, 3.62), (10.65, 2.72)], RED, dashed=True, lw=0.56, ms=6.5)
    poly_arrow(ax, [(12.90, 2.72), (12.90, 3.72), (3.92, 3.72), (3.92, 4.48)], TEAL, dashed=True, lw=0.56, ms=6.5)

    ax.plot([1.20, 1.72], [1.15, 1.15], color=SLATE, linewidth=0.72)
    arrow(ax, (1.72, 1.15), (2.05, 1.15), SLATE)
    ax.text(2.20, 1.15, "在线推理", va="center", fontsize=5.35, color=SLATE)
    ax.plot([3.48, 4.14], [1.15, 1.15], color=PURPLE, linewidth=0.72, linestyle=(0, (4, 3)))
    ax.text(4.28, 1.15, "预训练/校准/维护反馈", va="center", fontsize=5.35, color=SLATE)
    ax.text(7.70, 1.15, "目标：瞬时质量流量、累计质量误差和运行期风险边界协同受控。", va="center", fontsize=5.35, color=SLATE)
    save_figure(fig, OUT / "fig2_pgtcs_framework", png_dpi=600, tiff_dpi=600, pdf=False)
    plt.close(fig)


def fig3() -> None:
    fig, ax = plt.subplots(figsize=(7.35, 4.65))
    ax.set_xlim(0, 15.6)
    ax.set_ylim(0, 9.25)
    ax.axis("off")

    page_title(ax, "PG-TSC Residual Network", "基于实现代码的物理约束残差补偿结构", x=0, y=8.86, width=15.6)
    ax.add_patch(
        FancyBboxPatch(
            (0.42, 0.62),
            14.76,
            7.82,
            boxstyle="round,pad=0.025,rounding_size=0.045",
            facecolor="#FBFCFE",
            edgecolor="#D9E2EC",
            linewidth=0.56,
        )
    )

    _heading(ax, 0.84, 7.92, "A", "物理基础估计分支", TEAL)
    _heading(ax, 0.84, 5.48, "B", "TCN时序残差分支", BLUE)
    _heading(ax, 0.84, 2.36, "C", "训练、微调与少样本校准接口", PURPLE)

    _thin_box(
        ax,
        (0.92, 6.50),
        (2.08, 0.90),
        "20维标准化输入",
        r"$Z_{t-L+1:t}\in\mathbb{R}^{L\times20}$",
        fc="#F7FAFD",
        ec="#D5DEE8",
        color=NAVY,
        size=5.6,
    )
    feature_patch = FancyBboxPatch(
        (0.92, 5.72),
        2.08,
        0.58,
        boxstyle="round,pad=0.024,rounding_size=0.035",
        facecolor="#FFFFFF",
        edgecolor="#D9E2EC",
        linewidth=0.54,
    )
    ax.add_patch(feature_patch)
    ax.text(1.96, 6.10, "输入特征组", fontsize=4.55, color=SLATE, ha="center", va="center")
    ax.text(1.96, 5.91, "串口/环境/煤质", fontsize=4.25, color=SLATE, ha="center", va="center")
    ax.text(1.96, 5.76, "衰减/30 s统计/coal", fontsize=4.25, color=SLATE, ha="center", va="center")

    _thin_box(
        ax,
        (3.55, 6.62),
        (2.22, 0.66),
        "当前物理特征",
        r"$z_t\subset Z,\ A_t=\ln(I_{0,t}/I_t)$",
        fc="#F1FBF8",
        ec="#CDE1DA",
        color=TEAL,
        size=5.25,
    )
    _thin_box(
        ax,
        (6.34, 6.62),
        (2.18, 0.66),
        "基础估计器",
        "StandardScaler + Ridge\n" + r"$\alpha=10$",
        fc="#F1FBF8",
        ec="#CDE1DA",
        color=TEAL,
        size=4.95,
    )
    _thin_box(ax, (9.08, 6.62), (1.72, 0.66), "基础流量", r"$\hat q_t^{phy}=F_\phi(z_t)$", fc="#F1FBF8", ec="#CDE1DA", color=TEAL, size=5.05)

    _thin_box(
        ax,
        (0.92, 4.28),
        (2.08, 0.82),
        "序列张量",
        r"$(B,L,20)\rightarrow(B,20,L)$",
        fc="#F7FAFD",
        ec="#D5DEE8",
        color=NAVY,
        size=5.4,
    )
    _thin_box(
        ax,
        (3.55, 4.15),
        (2.28, 1.06),
        "TCNRegressor",
        "TemporalBlock × N\n小模型: L=30,C=24,N=3\n预训练: L=60,C=48,N=4",
        fc="#EEF5FF",
        ec="#CBD8E6",
        color=BLUE,
        size=5.0,
    )

    tcn_blocks = [(6.40, "B0", "d=1"), (7.48, "B1", "d=2"), (8.56, "B2", "d=4"), (9.64, "B3", "d=8*")]
    for x, title, body in tcn_blocks:
        _thin_box(ax, (x, 4.34), (0.78, 0.62), title, body, fc="#FFFFFF", ec="#CBD8E6", color=BLUE, size=4.75)
    ax.text(8.86, 5.58, "TemporalBlock = [Conv1d(k=3,d) → Chomp → ReLU → Dropout]×2 + skip", fontsize=4.45, color=SLATE, ha="center", va="center")
    ax.text(10.04, 4.06, "* B3仅用于4层预训练配置", fontsize=4.25, color=SLATE, ha="center", va="center")

    _thin_box(ax, (10.88, 4.28), (1.48, 0.78), "最后时刻", r"$h_t=H[:, -1]$", fc="#F7FAFD", ec="#D5DEE8", color=NAVY, size=5.0)
    _thin_box(ax, (12.72, 4.16), (2.02, 0.98), "残差头", "Linear(C,32)\nReLU\nLinear(32,1) → " + r"$\tilde r_t$", fc="#F8F4FC", ec="#DDD5EA", color=PURPLE, size=4.6)

    _thin_box(ax, (11.34, 6.62), (1.58, 0.66), "反标准化", r"$\Delta\hat q_t=\sigma_y\tilde r_t+\mu_y$", fc="#F8F4FC", ec="#DDD5EA", color=PURPLE, size=4.95)
    _thin_box(ax, (13.52, 6.62), (1.28, 0.66), "残差补偿", r"$\hat q_t=\hat q_t^{phy}+\Delta\hat q_t$", fc="#FFF8EF", ec="#E6D5C0", color=ORANGE, size=4.75)
    _thin_box(ax, (13.10, 5.46), (1.70, 0.64), "非负截断", r"$\max(\hat q_t,0)$", fc="#FFF8EF", ec="#E6D5C0", color=ORANGE, size=4.8)
    _thin_box(ax, (13.10, 3.16), (1.70, 0.70), "安装级校准", r"$\hat q_t^{final}=\max(a_d\hat q_t+b_d,0)$", fc="#FFF6F4", ec="#E6D2CE", color=RED, size=4.6)
    _thin_box(ax, (13.10, 2.02), (1.70, 0.62), "累计质量", r"$\hat M=\sum\hat q_t^{final}\Delta t$", fc="#F1FBF8", ec="#CDE1DA", color=TEAL, size=4.75)

    poly_arrow(ax, [(3.00, 6.95), (3.55, 6.95)], TEAL, lw=0.58, ms=6.5)
    poly_arrow(ax, [(5.77, 6.95), (6.34, 6.95)], TEAL, lw=0.58, ms=6.5)
    poly_arrow(ax, [(8.52, 6.95), (9.08, 6.95)], TEAL, lw=0.58, ms=6.5)
    poly_arrow(ax, [(10.80, 6.95), (11.34, 6.95)], TEAL, lw=0.58, ms=6.5)
    poly_arrow(ax, [(12.92, 6.95), (13.52, 6.95)], PURPLE, lw=0.58, ms=6.5)
    poly_arrow(ax, [(14.16, 6.62), (14.16, 6.10)], ORANGE, lw=0.58, ms=6.5)

    poly_arrow(ax, [(3.00, 4.69), (3.55, 4.69)], BLUE, lw=0.58, ms=6.5)
    poly_arrow(ax, [(5.83, 4.69), (6.40, 4.69)], BLUE, lw=0.58, ms=6.5)
    for x0, x1 in [(7.18, 7.48), (8.26, 8.56), (9.34, 9.64)]:
        poly_arrow(ax, [(x0, 4.65), (x1, 4.65)], BLUE, lw=0.50, ms=5.8)
    poly_arrow(ax, [(10.42, 4.69), (10.88, 4.69)], BLUE, lw=0.58, ms=6.5)
    poly_arrow(ax, [(12.36, 4.69), (12.72, 4.69)], BLUE, lw=0.58, ms=6.5)
    poly_arrow(ax, [(13.73, 5.14), (13.73, 6.32), (12.13, 6.32), (12.13, 6.62)], PURPLE, lw=0.58, ms=6.5)
    poly_arrow(ax, [(13.95, 5.46), (13.95, 3.86)], RED, lw=0.58, ms=6.5)
    poly_arrow(ax, [(13.95, 3.16), (13.95, 2.64)], TEAL, lw=0.58, ms=6.5)

    _thin_box(
        ax,
        (0.92, 1.16),
        (3.02, 0.74),
        "残差训练",
        r"SmoothL1($G_\theta(Z)$, 标准化 residual_flow)",
        fc="#FFFFFF",
        ec="#D9E2EC",
        color=NAVY,
        size=4.85,
    )
    _thin_box(
        ax,
        (4.38, 1.16),
        (3.14, 0.74),
        "优化实现",
        "AdamW lr=1e-3, weight_decay=1e-4, grad clip=2",
        fc="#FFFFFF",
        ec="#D9E2EC",
        color=NAVY,
        size=4.8,
    )
    _thin_box(
        ax,
        (7.96, 1.16),
        (3.08, 0.74),
        "累计质量微调",
        r"$L=L_{point}+\lambda_M L_M$",
        fc="#FFFFFF",
        ec="#D9E2EC",
        color=PURPLE,
        size=4.9,
    )
    _thin_box(
        ax,
        (11.48, 1.16),
        (3.32, 0.74),
        "少样本安装校准",
        "最小二乘估计 a_d,b_d；可用质量比例锚定",
        fc="#FFFFFF",
        ec="#D9E2EC",
        color=RED,
        size=4.75,
    )
    save_figure(fig, OUT / "fig3_network_architecture", png_dpi=600, tiff_dpi=600, pdf=False)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate method diagrams for the manuscript.")
    parser.add_argument("--only", choices=["all", "fig1", "fig2", "fig3"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    setup()
    if args.only in {"all", "fig1"}:
        fig1()
        print(f"wrote={OUT / 'fig1_system_error_sources.svg'}")
    if args.only in {"all", "fig2"}:
        fig2()
        print(f"wrote={OUT / 'fig2_pgtcs_framework.svg'}")
    if args.only in {"all", "fig3"}:
        fig3()
        print(f"wrote={OUT / 'fig3_network_architecture.svg'}")


if __name__ == "__main__":
    main()
