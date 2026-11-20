"""
Clinical PDF Report Generator & CSV/JSON Exporter for Saigak-Orto Craniometry.
Generates publication-quality A4 vector PDF reports using Matplotlib PdfPages
with strict non-overlapping bounding boxes and automatic line wrapping.
"""
import csv
import io
import textwrap
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
from core.mesh_io import MeshData, QualityReport


def generate_pdf_report_bytes(
    scan_id: str,
    mesh: MeshData,
    quality: QualityReport,
    craniometry: Optional[Dict],
    landmarks: Optional[Dict[str, List[float]]],
    model_version: str = "Saigak-ML-ExtraTrees-v1.2 (seed=42)",
    patient_age_months: Optional[float] = None,
    doctor_name: str = "Золотухин А.П. / Бойко А.",
    doctor_verified: bool = True,
    manual_correction_applied: bool = False,
    clinical_notes: str = "",
) -> bytes:
    """
    Render a complete A4 clinical PDF report into bytes with strict layout isolation.
    """
    buf = io.BytesIO()
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    age_str = (
        f"{patient_age_months:.1f} мес. (введено врачом)"
        if patient_age_months is not None and patient_age_months > 0
        else "Не указан (ручной ввод)"
    )

    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69), dpi=150)  # A4 Portrait
        fig.patch.set_facecolor("#FFFFFF")

        # ==================== 1. TOP HEADER BANNER ====================
        ax_hdr = fig.add_axes([0.04, 0.915, 0.92, 0.062])
        ax_hdr.set_axis_off()
        hdr_box = patches.FancyBboxPatch(
            (0, 0), 1, 1,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor="#111827", edgecolor="#1F2937", linewidth=1.0
        )
        ax_hdr.add_patch(hdr_box)
        ax_hdr.text(0.02, 0.67, "САЙГАК ОРТО", fontsize=12.5, fontweight="bold", color="#D4A843", va="center")
        ax_hdr.text(
            0.205, 0.67,
            "|  ПРОТОКОЛ 3D-КРАНИОМЕТРИИ И ОЦЕНКИ ДЕФОРМАЦИЙ",
            fontsize=8.3, fontweight="bold", color="#F9FAFB", va="center"
        )
        ax_hdr.text(
            0.02, 0.24,
            f"Код скана: {scan_id[:26]}   •   Дата: {timestamp_str}   •   Возраст: {age_str}",
            fontsize=7.3, color="#D1D5DB", va="center"
        )

        status_colors = {
            "VALID": ("#059669", "КАЧЕСТВО СКАНА: НОРМА"),
            "WARNING": ("#D97706", "КАЧЕСТВО: ДОПУСТИМО"),
            "REJECTED": ("#DC2626", "ОТКАЗ: ДЕФЕКТ СКАНА"),
        }
        st_col, st_txt = status_colors.get(quality.status, ("#6B7280", quality.status))
        pill = patches.FancyBboxPatch(
            (0.775, 0.20), 0.205, 0.60,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor=st_col, edgecolor="none"
        )
        ax_hdr.add_patch(pill)
        ax_hdr.text(0.8775, 0.50, st_txt, fontsize=6.9, fontweight="bold", color="#FFFFFF", ha="center", va="center")

        # ==================== REJECTED SCAN LAYOUT ====================
        if not quality.is_usable or craniometry is None:
            ax_rej = fig.add_axes([0.04, 0.43, 0.92, 0.46])
            ax_rej.set_axis_off()
            rej_box = patches.FancyBboxPatch(
                (0, 0), 1, 1,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                facecolor="#FEF2F2", edgecolor="#EF4444", linewidth=1.4
            )
            ax_rej.add_patch(rej_box)
            ax_rej.text(
                0.035, 0.91,
                "ОТКАЗ ОТ АВТОМАТИЧЕСКОГО РАСЧЁТА КРАНИАЛЬНЫХ ИНДЕКСОВ",
                fontsize=11.0, fontweight="bold", color="#991B1B"
            )
            ax_rej.text(
                0.035, 0.83,
                "Входная 3D-модель не прошла обязательный контроль целостности, полноты или масштаба (п. 5.1 и 8.1 ТЗ).",
                fontsize=8.5, color="#7F1D1D"
            )

            y_cursor = 0.73
            ax_rej.text(0.035, y_cursor, "Выявленные критические дефекты входных данных:", fontsize=9.0, fontweight="bold", color="#111827")
            for d in quality.defects:
                wrapped_d = textwrap.fill(f"• {d}", width=92)
                n_lines = wrapped_d.count("\n") + 1
                y_cursor -= 0.045 * n_lines + 0.015
                ax_rej.text(0.05, y_cursor, wrapped_d, fontsize=8.0, color="#B91C1C", va="bottom")

            for w in quality.warnings:
                wrapped_w = textwrap.fill(f"• [Предупреждение] {w}", width=92)
                n_lines = wrapped_w.count("\n") + 1
                y_cursor -= 0.045 * n_lines + 0.015
                ax_rej.text(0.05, y_cursor, wrapped_w, fontsize=7.8, color="#92400E", va="bottom")

            y_cursor -= 0.04
            tech_summary = (
                f"Технические параметры файла:\n"
                f"  • Формат: {mesh.format_type} ({mesh.file_size_bytes / 1024:.1f} КБ)\n"
                f"  • Полигонов (треугольников): {quality.triangle_count:,}  |  Уникальных вершин: {quality.unique_vertex_count:,}\n"
                f"  • Габариты после масштабирования (×{quality.applied_scale:g}): "
                f"{quality.extents_mm[0]:.1f} × {quality.extents_mm[1]:.1f} × {quality.extents_mm[2]:.1f} мм\n"
                f"  • Замкнутость угловых секторов свода: {quality.vault_closure_pct:.1f}% (порог ≥ 85.0%)\n"
                f"  • Покрытие квадрантов свода: {quality.quadrant_coverage_pct:.1f}% (порог ≥ 35.0%)\n"
                f"  • Оценка качества данных: {quality.quality_score:.1f} / 100"
            )
            ax_rej.text(0.035, y_cursor, tech_summary, fontsize=8.0, color="#1F2937", va="top", family="monospace", linespacing=1.35)

            pts = (mesh.v1 + mesh.v2 + mesh.v3) / 3.0
            step = max(1, len(pts) // 4000)
            pts_sub = pts[::step]

            ax_p1 = fig.add_axes([0.08, 0.16, 0.37, 0.22])
            ax_p1.scatter(pts_sub[:, 1], pts_sub[:, 0], s=1.0, c="#DC2626", alpha=0.5)
            ax_p1.set_title("Вид сверху (XY): нарушение полноты свода", fontsize=8.0, fontweight="bold")
            ax_p1.set_xlabel("Y (мм)", fontsize=7.0)
            ax_p1.set_ylabel("X (мм)", fontsize=7.0)
            ax_p1.tick_params(labelsize=6.5)
            ax_p1.set_aspect("equal", "datalim")
            ax_p1.grid(True, linestyle="--", alpha=0.4)

            ax_p2 = fig.add_axes([0.55, 0.16, 0.37, 0.22])
            ax_p2.scatter(pts_sub[:, 0], pts_sub[:, 2], s=1.0, c="#EA580C", alpha=0.5)
            ax_p2.set_title("Вид сбоку (XZ): профиль дефектного скана", fontsize=8.0, fontweight="bold")
            ax_p2.set_xlabel("X (мм)", fontsize=7.0)
            ax_p2.set_ylabel("Z (мм)", fontsize=7.0)
            ax_p2.tick_params(labelsize=6.5)
            ax_p2.set_aspect("equal", "datalim")
            ax_p2.grid(True, linestyle="--", alpha=0.4)

        else:
            # ==================== 2. STANDARD / DEFORMITY REPORT ====================
            ind = craniometry["indices"]
            lin = craniometry["linear_measurements_mm"]
            reg = craniometry["regional_asymmetry"]
            cls_info = craniometry["classification"]
            cvai_grade = cls_info["cvai_grade"]
            g2d = craniometry["geometry_2d"]

            # Left Plot: 2D Cross-Section Blueprint (with wide margin on left and internal padding)
            ax_2d = fig.add_axes([0.085, 0.56, 0.395, 0.31])
            c_xy = np.array(g2d["contour_xy"])
            ep = g2d["endpoints_xy"]
            center_2d = g2d["center"]

            c_closed = np.vstack([c_xy, c_xy[0]])
            ax_2d.fill(c_closed[:, 0], c_closed[:, 1], color="#F0FDFA", alpha=0.65)
            ax_2d.plot(c_closed[:, 0], c_closed[:, 1], color="#0F766E", linewidth=1.9, label="Контур сечения (Level 3)")

            ax_2d.plot(
                [ep["ap_back"][0], ep["ap_front"][0]],
                [ep["ap_back"][1], ep["ap_front"][1]],
                color="#1D4ED8", linewidth=1.5, linestyle="--", label=f"Длина L = {lin['L_length_ap']:.1f} мм"
            )
            ax_2d.plot(
                [ep["ml_right"][0], ep["ml_left"][0]],
                [ep["ml_right"][1], ep["ml_left"][1]],
                color="#0369A1", linewidth=1.5, linestyle=":", label=f"Ширина W = {lin['W_width_ml']:.1f} мм"
            )
            ax_2d.plot(
                [ep["d1_back"][0], ep["d1_front"][0]],
                [ep["d1_back"][1], ep["d1_front"][1]],
                color="#D97706", linewidth=1.9, label=f"D1 (+30°) = {lin['D1_diagonal_plus30']:.1f} мм"
            )
            ax_2d.plot(
                [ep["d2_back"][0], ep["d2_front"][0]],
                [ep["d2_back"][1], ep["d2_front"][1]],
                color="#DC2626", linewidth=1.9, label=f"D2 (-30°) = {lin['D2_diagonal_minus30']:.1f} мм"
            )

            for key, col in [
                ("ap_front", "#1D4ED8"), ("ap_back", "#1D4ED8"),
                ("d1_front", "#D97706"), ("d1_back", "#D97706"),
                ("d2_front", "#DC2626"), ("d2_back", "#DC2626"),
            ]:
                ax_2d.scatter([ep[key][0]], [ep[key][1]], color=col, s=20, zorder=5)
            ax_2d.scatter([center_2d[0]], [center_2d[1]], color="#111827", s=26, zorder=6)

            # Expand axis limits by 34% so anatomical labels and legend never overlap
            x_min_c, x_max_c = float(c_xy[:, 0].min()), float(c_xy[:, 0].max())
            y_min_c, y_max_c = float(c_xy[:, 1].min()), float(c_xy[:, 1].max())
            span = max(x_max_c - x_min_c, y_max_c - y_min_c) * 0.72
            ax_2d.set_xlim(center_2d[0] - span, center_2d[0] + span)
            ax_2d.set_ylim(center_2d[1] - span * 1.18, center_2d[1] + span * 1.12)

            ax_2d.text(ep["ap_front"][0], ep["ap_front"][1] + span * 0.05, "ПЕРЕД (Glabella)", fontsize=6.8, fontweight="bold", ha="center", va="bottom", color="#1E3A8A")
            ax_2d.text(ep["ap_back"][0] - span * 0.22, ep["ap_back"][1] - span * 0.05, "ЗАТЫЛОК (Opisthocranion)", fontsize=6.6, fontweight="bold", ha="center", va="top", color="#1E3A8A")
            ax_2d.text(ep["ml_right"][0] - span * 0.04, ep["ml_right"][1], "ПРАВ", fontsize=6.6, fontweight="bold", ha="right", va="center", color="#374151")
            ax_2d.text(ep["ml_left"][0] + span * 0.04, ep["ml_left"][1], "ЛЕВ", fontsize=6.6, fontweight="bold", ha="left", va="center", color="#374151")

            ax_2d.set_title(f"Измерительное сечение (Z = {lin['slice_height_z']:.1f} мм / {lin['slice_level_ratio']*100:.0f}% свода)", fontsize=8.3, fontweight="bold", pad=6)
            ax_2d.set_xlabel("Поперечная ось ML (мм)", fontsize=7.2)
            ax_2d.set_ylabel("Переднезадняя ось AP (мм)", fontsize=7.2)
            ax_2d.tick_params(labelsize=6.8)
            ax_2d.set_aspect("equal", "box")
            ax_2d.grid(True, linestyle="--", alpha=0.35)
            ax_2d.legend(loc="lower right", fontsize=5.6, framealpha=0.94, borderpad=0.35)

            # Right Top Plot: Lateral Sagittal Projection (with separated TrR and TrL callouts)
            ax_lat = fig.add_axes([0.56, 0.745, 0.39, 0.125])
            origin = np.array(craniometry["geometry_3d"]["origin"])
            R = np.array(craniometry["geometry_3d"]["axes_rotation_matrix"])
            pts = (mesh.v1 + mesh.v2 + mesh.v3) / 3.0
            step = max(1, len(pts) // 3500)
            local_pts = (pts[::step] - origin) @ R.T

            ax_lat.scatter(local_pts[:, 1], local_pts[:, 2], s=0.7, c="#94A3B8", alpha=0.45)
            ax_lat.axhline(0.0, color="#475569", linestyle="--", linewidth=1.1, label="Базовая пл. (Se-TrR-TrL)")
            ax_lat.axhline(lin["slice_height_z"], color="#0D9488", linestyle="-", linewidth=1.6, label=f"Сечение Z={lin['slice_height_z']:.1f} мм")

            if landmarks:
                # Offset TrR and TrL labels in opposite directions so they never overlap in sagittal view
                lm_offsets = {
                    "Sellion": ("#DC2626", "Se", (4.0, 4.0)),
                    "RightTragion": ("#2563EB", "TrR", (-18.0, 6.0)),
                    "LeftTragion": ("#7C3AED", "TrL", (5.0, -11.0)),
                }
                for lm_name, (lm_col, lm_lbl, (dx_pt, dy_pt)) in lm_offsets.items():
                    if lm_name in landmarks:
                        lm_loc = (np.array(landmarks[lm_name]) - origin) @ R.T
                        ax_lat.scatter([lm_loc[1]], [lm_loc[2]], color=lm_col, s=24, zorder=6)
                        ax_lat.annotate(
                            lm_lbl,
                            xy=(lm_loc[1], lm_loc[2]),
                            xytext=(dx_pt, dy_pt),
                            textcoords="offset points",
                            fontsize=7.0,
                            fontweight="bold",
                            color=lm_col,
                        )

            ax_lat.set_title("Сагиттальная проекция: ориентиры и уровень среза", fontsize=8.0, fontweight="bold", pad=5)
            ax_lat.set_xlabel("Переднезадняя ось AP (мм)", fontsize=6.8, labelpad=2)
            ax_lat.set_ylabel("Высота Z (мм)", fontsize=6.8)
            ax_lat.tick_params(labelsize=6.5)
            ax_lat.legend(loc="upper left", fontsize=5.8)
            ax_lat.grid(True, linestyle=":", alpha=0.4)

            # Right Bottom Plot: Radial Asymmetry Profile Curve
            ax_asym = fig.add_axes([0.56, 0.56, 0.39, 0.115])
            deg_grid = np.linspace(-180, 179, 360)
            asym_prof = np.array(g2d["asymmetry_profile_mm"])
            ax_asym.axhline(0.0, color="#64748B", linewidth=0.9, linestyle="--")
            ax_asym.fill_between(deg_grid, 0, asym_prof, where=(asym_prof >= 0), color="#F97316", alpha=0.35)
            ax_asym.fill_between(deg_grid, 0, asym_prof, where=(asym_prof < 0), color="#0284C7", alpha=0.35)
            ax_asym.plot(deg_grid, asym_prof, color="#1E293B", linewidth=1.1)
            ax_asym.set_title("Карта радиальной асимметрии контура Δr(θ), мм", fontsize=8.0, fontweight="bold", pad=5)
            ax_asym.set_xlabel("Угол θ от передней оси AP (°)", fontsize=6.8, labelpad=2)
            ax_asym.set_ylabel("Δr (мм)", fontsize=6.8)
            ax_asym.tick_params(labelsize=6.5)
            ax_asym.set_xlim(-180, 180)
            ax_asym.grid(True, linestyle=":", alpha=0.4)

            # ==================== 3. KPI CARDS & CVAI SCALE BAR ====================
            ax_scale = fig.add_axes([0.04, 0.422, 0.92, 0.092])
            ax_scale.set_axis_off()
            card_bg = patches.FancyBboxPatch(
                (0, 0), 1, 1,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.0
            )
            ax_scale.add_patch(card_bg)

            # 3 Isolated KPI Boxes on Left so headers never collide
            kpi_specs = [
                (0.015, "ИНДЕКС CI", f"{ind['CI']:.2f}%", "100 × W / L", "#0F172A"),
                (0.180, "АСИММЕТРИЯ CVA", f"{ind['CVA_mm']:.2f} мм", "|D1 − D2|", "#0F172A"),
                (0.345, "ИНДЕКС CVAI", f"{ind['CVAI']:.2f}%", "100×CVA/max", cvai_grade["color"]),
            ]
            for kx, k_title, k_val, k_sub, k_col in kpi_specs:
                sub_box = patches.FancyBboxPatch(
                    (kx, 0.12), 0.152, 0.76,
                    boxstyle="round,pad=0.005,rounding_size=0.015",
                    facecolor="#FFFFFF", edgecolor="#E2E8F0", linewidth=0.8
                )
                ax_scale.add_patch(sub_box)
                ax_scale.text(kx + 0.012, 0.73, k_title, fontsize=6.6, fontweight="bold", color="#475569", va="center")
                ax_scale.text(kx + 0.012, 0.43, k_val, fontsize=13.5, fontweight="bold", color=k_col, va="center")
                ax_scale.text(kx + 0.012, 0.20, k_sub, fontsize=5.8, color="#64748B", va="center")

            # Right half of KPI banner: CVAI Category & Segmented Gauge
            ax_scale.text(0.525, 0.77, "Категория по шкале CVAI [Graham et al., 2020]:", fontsize=7.3, fontweight="bold", color="#334155")
            ax_scale.text(0.525, 0.53, cvai_grade["label"], fontsize=8.6, fontweight="bold", color=cvai_grade["color"])

            bar_x0, bar_w, bar_y, bar_h = 0.525, 0.445, 0.19, 0.17
            segments = [
                (0.0, 3.5, "#0D9488", "<3.5%"),
                (3.5, 6.25, "#65A30D", "3.5-6.25"),
                (6.25, 8.75, "#D97706", "6.25-8.75"),
                (8.75, 11.0, "#EA580C", "8.75-11"),
                (11.0, 14.0, "#DC2626", "≥11%"),
            ]
            for s_min, s_max, s_col, s_lbl in segments:
                sx = bar_x0 + (s_min / 14.0) * bar_w
                sw = ((s_max - s_min) / 14.0) * bar_w
                ax_scale.add_patch(patches.Rectangle((sx, bar_y), sw, bar_h, facecolor=s_col, alpha=0.88, edgecolor="#FFFFFF", linewidth=0.8))
                ax_scale.text(sx + 0.5 * sw, bar_y - 0.11, s_lbl, fontsize=5.8, ha="center", color="#475569")

            marker_val = min(13.8, max(0.1, ind["CVAI_unrounded"]))
            mx = bar_x0 + (marker_val / 14.0) * bar_w
            ax_scale.plot([mx, mx], [bar_y - 0.02, bar_y + bar_h + 0.04], color="#0F172A", linewidth=2.0)
            ax_scale.scatter([mx], [bar_y + bar_h + 0.05], color="#0F172A", marker="v", s=30, zorder=10)

            # ==================== 4. TWO ISOLATED DATA CARDS (LEFT & RIGHT) ====================
            # Left Card: Linear & Regional Measurements
            ax_left = fig.add_axes([0.04, 0.155, 0.45, 0.25])
            ax_left.set_axis_off()
            ax_left.add_patch(patches.FancyBboxPatch(
                (0, 0), 1, 1,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=0.9
            ))

            col_left_lines = [
                "ЛИНЕЙНЫЕ ИЗМЕРЕНИЯ СЕЧЕНИЯ (LEVEL 3):",
                f" • Длина головы (L, AP):        {lin['L_length_ap']:7.2f} мм",
                f" • Ширина головы (W, ML):       {lin['W_width_ml']:7.2f} мм",
                f" • Диагональ D1 (+30°):         {lin['D1_diagonal_plus30']:7.2f} мм",
                f" • Диагональ D2 (-30°):         {lin['D2_diagonal_minus30']:7.2f} мм",
                f" • Периметр сечения:            {lin['perimeter_slice']:7.2f} мм",
                f" • Высота свода / срез Z:   {lin['vault_height']:.1f} / {lin['slice_height_z']:.1f} мм",
                "",
                "РЕГИОНАЛЬНАЯ АСИММЕТРИЯ (Should п. 5.2):",
                f" • Фронтальная:  {reg['frontal_diff_mm']:5.2f} мм ({reg['frontal_asymmetry_pct']:5.2f}%)",
                f" • Затылочная:   {reg['occipital_diff_mm']:5.2f} мм ({reg['occipital_asymmetry_pct']:5.2f}%)",
                f" • Преобладает:  {reg['dominant_asymmetry_zone']}",
            ]
            ax_left.text(
                0.035, 0.94,
                "\n".join(col_left_lines),
                fontsize=7.3, color="#1E293B", va="top", family="monospace", linespacing=1.36
            )

            # Right Card: Formulas, CI Proportions & Anatomical Landmarks
            ax_right = fig.add_axes([0.51, 0.155, 0.45, 0.25])
            ax_right.set_axis_off()
            ax_right.add_patch(patches.FancyBboxPatch(
                (0, 0), 1, 1,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=0.9
            ))

            ci_wrapped = textwrap.fill(f" • Пропорции: {cls_info['ci_interpretation']}", width=48, subsequent_indent="   ")
            col_right_lines = [
                "ФОРМУЛЫ, ПРОПОРЦИИ CI И КАЧЕСТВО:",
                f" • CI   = 100 × W / L = {ind['CI']:.2f}%",
                f" • CVA  = |D1 − D2| = {ind['CVA_mm']:.2f} мм",
                f" • CVAI = 100×|D1-D2|/max = {ind['CVAI']:.2f}%",
                f" • Справочно CVAI_min: {ind['CVAI_min_ref']:.2f}%",
                ci_wrapped,
                "",
                "КООРДИНАТЫ ОРИЕНТИРОВ (X, Y, Z в мм):",
            ]
            if landmarks:
                for k in ["Sellion", "RightTragion", "LeftTragion", "Subnasal"]:
                    if k in landmarks:
                        c = landmarks[k]
                        col_right_lines.append(f" • {k:12s}: {c[0]:5.1f}, {c[1]:6.1f}, {c[2]:5.1f}")
            col_right_lines.append(
                f" • Масштаб ×{quality.applied_scale:g} ({quality.triangle_count:,} тр., {quality.quality_score:.0f}/100)"
            )
            ax_right.text(
                0.035, 0.94,
                "\n".join(col_right_lines),
                fontsize=7.1, color="#1E293B", va="top", family="monospace", linespacing=1.28
            )

        # ==================== 5. FOOTER & DOCTOR VERIFICATION STAMP ====================
        ax_ftr = fig.add_axes([0.04, 0.022, 0.92, 0.118])
        ax_ftr.set_axis_off()
        ftr_box = patches.FancyBboxPatch(
            (0, 0), 1, 1,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor="#F1F5F9", edgecolor="#94A3B8", linewidth=1.0
        )
        ax_ftr.add_patch(ftr_box)

        ver_mode = "Ручная коррекция специалистом + перерасчёт" if manual_correction_applied else "Автоматическая ML-разметка (подтверждено врачом)"
        ver_status = "ПРОВЕРЕНО И ПОДТВЕРЖДЕНО ВРАЧОМ" if doctor_verified else "ОЖИДАЕТ ПРОВЕРКИ СПЕЦИАЛИСТОМ"
        raw_notes = clinical_notes if clinical_notes else "Контроль динамики формы черепа по клиническому протоколу."
        notes_wrapped = textwrap.fill(f"Заключение врача: {raw_notes}", width=114)
        disclaimer_wrapped = textwrap.fill(
            "ДИСКЛЕЙМЕР (п. 6.2, 9.2 ТЗ): Категория CVAI определяется по неокруглённому значению. Значение ниже порога шкалы не означает отсутствие любого заболевания. Прототип предназначен для стандартизации измерений и не используется для самостоятельной диагностики или назначения шлемотерапии.",
            width=118
        )

        ftr_text = (
            f"МЕТОДИКА: STARBAND-GRAHAM-2020-V1.2 (Graham T. et al., J Clin Med 2020;9(4):1027; FDA K141842)  |  МОДЕЛЬ: {model_version}\n"
            f"РЕЖИМ РАЗМЕТКИ: {ver_mode}  |  СТАТУС: [{ver_status}]\n"
            f"Врач / Специалист по ортезам: {doctor_name}                    Подпись: ______________________\n"
            f"{notes_wrapped}\n"
            f"{disclaimer_wrapped}"
        )
        ax_ftr.text(0.018, 0.92, ftr_text, fontsize=6.6, color="#334155", va="top", linespacing=1.28)

        pdf.savefig(fig)
        plt.close(fig)

    return buf.getvalue()


def export_measurement_json(
    scan_id: str,
    quality: QualityReport,
    craniometry: Optional[Dict],
    landmarks: Optional[Dict[str, List[float]]],
    model_version: str = "Saigak-ML-ExtraTrees-v1.2 (seed=42)",
    patient_age_months: Optional[float] = None,
    doctor_name: str = "Золотухин А.П. / Бойко А.",
    doctor_verified: bool = True,
    manual_correction_applied: bool = False,
) -> Dict:
    return {
        "scan_id": scan_id,
        "timestamp": datetime.now().isoformat(),
        "patient_age_months": patient_age_months,
        "model_version": model_version,
        "methodology_version": "STARBAND-GRAHAM-2020-V1.2",
        "doctor_verification": {
            "verified": doctor_verified,
            "doctor_name": doctor_name,
            "manual_correction_applied": manual_correction_applied,
        },
        "input_quality": quality.to_dict(),
        "landmarks_mm": landmarks,
        "craniometry": {
            "indices": craniometry["indices"],
            "linear_measurements_mm": craniometry["linear_measurements_mm"],
            "regional_asymmetry": craniometry["regional_asymmetry"],
            "classification": craniometry["classification"],
        } if craniometry else None,
    }


def export_measurement_csv_string(records: List[Dict]) -> str:
    output = io.StringIO()
    fieldnames = [
        "scan_id", "timestamp", "quality_status", "quality_score", "applied_scale",
        "patient_age_months", "CI_pct", "CVA_mm", "CVAI_pct", "CVAI_min_ref_pct",
        "cvai_grade_code", "cvai_grade_label", "L_mm", "W_mm", "D1_mm", "D2_mm",
        "perimeter_slice_mm", "frontal_asymmetry_pct", "occipital_asymmetry_pct",
        "manual_correction", "doctor_verified", "doctor_name", "model_version", "methodology_version"
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for r in records:
        q = r.get("input_quality", {})
        c = r.get("craniometry") or {}
        ind = c.get("indices", {})
        lin = c.get("linear_measurements_mm", {})
        reg = c.get("regional_asymmetry", {})
        cls_info = c.get("classification", {}).get("cvai_grade", {})
        doc = r.get("doctor_verification", {})
        writer.writerow({
            "scan_id": r.get("scan_id"),
            "timestamp": r.get("timestamp"),
            "quality_status": q.get("status"),
            "quality_score": q.get("quality_score"),
            "applied_scale": q.get("applied_scale"),
            "patient_age_months": r.get("patient_age_months", ""),
            "CI_pct": ind.get("CI", ""),
            "CVA_mm": ind.get("CVA_mm", ""),
            "CVAI_pct": ind.get("CVAI", ""),
            "CVAI_min_ref_pct": ind.get("CVAI_min_ref", ""),
            "cvai_grade_code": cls_info.get("code", ""),
            "cvai_grade_label": cls_info.get("label", ""),
            "L_mm": lin.get("L_length_ap", ""),
            "W_mm": lin.get("W_width_ml", ""),
            "D1_mm": lin.get("D1_diagonal_plus30", ""),
            "D2_mm": lin.get("D2_diagonal_minus30", ""),
            "perimeter_slice_mm": lin.get("perimeter_slice", ""),
            "frontal_asymmetry_pct": reg.get("frontal_asymmetry_pct", ""),
            "occipital_asymmetry_pct": reg.get("occipital_asymmetry_pct", ""),
            "manual_correction": doc.get("manual_correction_applied", False),
            "doctor_verified": doc.get("verified", False),
            "doctor_name": doc.get("doctor_name", ""),
            "model_version": r.get("model_version", ""),
            "methodology_version": r.get("methodology_version", ""),
        })
    return output.getvalue()
