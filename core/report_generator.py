"""
Clinical PDF Report Generator, Excel/CSV/JSON Exporter for Saigak-Orto Craniometry.
Generates publication-quality A4 vector PDF reports using Matplotlib PdfPages
with strict non-overlapping bounding boxes, 2D Level 3 measurement schema (L, W, D1, D2, Q1-Q4),
borderline threshold alerts, doctor intervention audit stamp, and technician defect guidance.
"""
import csv
import io
import textwrap
from datetime import datetime
from typing import Dict, List, Optional
from xml.sax.saxutils import escape as xml_escape
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
    landmark_confidence: Optional[Dict] = None,
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
        ax_hdr = fig.add_axes([0.04, 0.918, 0.92, 0.060])
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
            fontsize=8.2, fontweight="bold", color="#F9FAFB", va="center"
        )
        ax_hdr.text(
            0.02, 0.24,
            f"Код скана: {scan_id[:26]}   •   Дата: {timestamp_str}   •   Возраст: {age_str}",
            fontsize=7.2, color="#D1D5DB", va="center"
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
            ax_rej = fig.add_axes([0.04, 0.43, 0.92, 0.465])
            ax_rej.set_axis_off()
            rej_box = patches.FancyBboxPatch(
                (0, 0), 1, 1,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                facecolor="#FEF2F2", edgecolor="#EF4444", linewidth=1.4
            )
            ax_rej.add_patch(rej_box)
            ax_rej.text(
                0.035, 0.92,
                "ОТКАЗ ОТ АВТОМАТИЧЕСКОГО РАСЧЁТА КРАНИАЛЬНЫХ ИНДЕКСОВ",
                fontsize=10.8, fontweight="bold", color="#991B1B"
            )
            ax_rej.text(
                0.035, 0.85,
                "Входная 3D-модель не прошла обязательный контроль целостности, полноты или масштаба.",
                fontsize=8.3, color="#7F1D1D"
            )

            y_cursor = 0.77
            ax_rej.text(0.035, y_cursor, "Выявленные критические дефекты входных данных:", fontsize=8.8, fontweight="bold", color="#111827")
            for d in quality.defects:
                wrapped_d = textwrap.fill(f"• {d}", width=92)
                n_lines = wrapped_d.count("\n") + 1
                y_cursor -= 0.042 * n_lines + 0.014
                ax_rej.text(0.05, y_cursor, wrapped_d, fontsize=7.8, color="#B91C1C", va="bottom")

            if getattr(quality, "technician_instructions", None):
                y_cursor -= 0.035
                ax_rej.text(0.035, y_cursor, "Рекомендация технику по повторному сканированию:", fontsize=8.8, fontweight="bold", color="#9A3412")
                for ins in quality.technician_instructions[:3]:
                    wrapped_i = textwrap.fill(f"→ {ins}", width=92)
                    n_lines = wrapped_i.count("\n") + 1
                    y_cursor -= 0.042 * n_lines + 0.012
                    ax_rej.text(0.05, y_cursor, wrapped_i, fontsize=7.8, fontweight="bold", color="#7C2D12", va="bottom")

            for w in quality.warnings:
                wrapped_w = textwrap.fill(f"• [Предупреждение] {w}", width=92)
                n_lines = wrapped_w.count("\n") + 1
                y_cursor -= 0.040 * n_lines + 0.012
                ax_rej.text(0.05, y_cursor, wrapped_w, fontsize=7.5, color="#92400E", va="bottom")

            y_cursor -= 0.035
            tech_summary = (
                f"Технические параметры файла:\n"
                f"  • Формат: {mesh.format_type} ({mesh.file_size_bytes / 1024:.1f} КБ) | Единицы: {getattr(quality, 'unit_diagnosis', 'мм')}\n"
                f"  • Полигонов: {quality.triangle_count:,} | Вершин: {quality.unique_vertex_count:,} | Фрагментов: {getattr(quality, 'disconnected_components_count', 1)}\n"
                f"  • Габариты (×{quality.applied_scale:g}): {quality.extents_mm[0]:.1f} × {quality.extents_mm[1]:.1f} × {quality.extents_mm[2]:.1f} мм\n"
                f"  • Замкнутость свода: {quality.vault_closure_pct:.1f}% (≥ 85%) | Квадранты: {quality.quadrant_coverage_pct:.1f}% (≥ 35%) | Оценка: {quality.quality_score:.1f}/100"
            )
            ax_rej.text(0.035, y_cursor, tech_summary, fontsize=7.5, color="#1F2937", va="top", family="monospace", linespacing=1.32)

            pts = (mesh.v1 + mesh.v2 + mesh.v3) / 3.0
            step = max(1, len(pts) // 4000)
            pts_sub = pts[::step]

            ax_p1 = fig.add_axes([0.08, 0.165, 0.37, 0.22])
            ax_p1.scatter(pts_sub[:, 1], pts_sub[:, 0], s=1.0, c="#64748B", alpha=0.45)
            for dz in getattr(quality, "defect_zones", []):
                cxy = dz["center_xy"]
                r_z = dz["radius_mm"]
                # Note plot axes are (Y, X) so angle_deg (from +X to +Y) maps to (sin, cos)
                wedge = patches.Wedge(
                    (cxy[1], cxy[0]), r_z * 1.15,
                    90.0 - dz["angle_end_deg"], 90.0 - dz["angle_start_deg"],
                    facecolor="#EF4444", alpha=0.28, edgecolor="#DC2626", linewidth=1.4, linestyle="--"
                )
                ax_p1.add_patch(wedge)
            ax_p1.set_title("Вид сверху (XY): красным выделена зона дефекта", fontsize=7.8, fontweight="bold", color="#991B1B")
            ax_p1.set_xlabel("Y (мм)", fontsize=7.0)
            ax_p1.set_ylabel("X (мм)", fontsize=7.0)
            ax_p1.tick_params(labelsize=6.5)
            ax_p1.set_aspect("equal", "datalim")
            ax_p1.grid(True, linestyle="--", alpha=0.4)

            ax_p2 = fig.add_axes([0.55, 0.165, 0.37, 0.22])
            ax_p2.scatter(pts_sub[:, 0], pts_sub[:, 2], s=1.0, c="#EA580C", alpha=0.5)
            ax_p2.set_title("Вид сбоку (XZ): профиль дефектного скана", fontsize=7.8, fontweight="bold")
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
            qvol = craniometry.get("quadrant_volumes") or {}
            cls_info = craniometry["classification"]
            cvai_grade = cls_info["cvai_grade"]
            borderline = cls_info.get("borderline_alert") or {}
            g2d = craniometry["geometry_2d"]

            # Left Plot: 2D Cross-Section Blueprint (L, W, D1, D2, Q1-Q4)
            ax_2d = fig.add_axes([0.085, 0.568, 0.395, 0.308])
            c_xy = np.array(g2d["contour_xy"])
            ep = g2d["endpoints_xy"]
            center_2d = g2d["center"]

            c_closed = np.vstack([c_xy, c_xy[0]])
            ax_2d.fill(c_closed[:, 0], c_closed[:, 1], color="#F0FDFA", alpha=0.65)
            ax_2d.plot(c_closed[:, 0], c_closed[:, 1], color="#0F766E", linewidth=1.9, label="Контур Level 3")

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

            x_min_c, x_max_c = float(c_xy[:, 0].min()), float(c_xy[:, 0].max())
            y_min_c, y_max_c = float(c_xy[:, 1].min()), float(c_xy[:, 1].max())
            span = max(x_max_c - x_min_c, y_max_c - y_min_c) * 0.74
            ax_2d.set_xlim(center_2d[0] - span, center_2d[0] + span)
            ax_2d.set_ylim(center_2d[1] - span * 1.18, center_2d[1] + span * 1.12)

            # Subtle Q1-Q4 quadrant labels inside contour
            ax_2d.text(center_2d[0] + span * 0.34, center_2d[1] + span * 0.34, "Q1", fontsize=6.5, fontweight="bold", color="#64748B", ha="center")
            ax_2d.text(center_2d[0] - span * 0.34, center_2d[1] + span * 0.34, "Q2", fontsize=6.5, fontweight="bold", color="#64748B", ha="center")
            ax_2d.text(center_2d[0] + span * 0.34, center_2d[1] - span * 0.34, "Q3", fontsize=6.5, fontweight="bold", color="#64748B", ha="center")
            ax_2d.text(center_2d[0] - span * 0.34, center_2d[1] - span * 0.34, "Q4", fontsize=6.5, fontweight="bold", color="#64748B", ha="center")

            ax_2d.text(ep["ap_front"][0], ep["ap_front"][1] + span * 0.05, "ПЕРЕД (Glabella)", fontsize=6.7, fontweight="bold", ha="center", va="bottom", color="#1E3A8A")
            ax_2d.text(ep["ap_back"][0] - span * 0.22, ep["ap_back"][1] - span * 0.05, "ЗАТЫЛОК (Opisthocranion)", fontsize=6.5, fontweight="bold", ha="center", va="top", color="#1E3A8A")
            ax_2d.text(ep["ml_right"][0] - span * 0.04, ep["ml_right"][1], "ПРАВ", fontsize=6.5, fontweight="bold", ha="right", va="center", color="#374151")
            ax_2d.text(ep["ml_left"][0] + span * 0.04, ep["ml_left"][1], "ЛЕВ", fontsize=6.5, fontweight="bold", ha="left", va="center", color="#374151")

            ax_2d.set_title(f"Схема среза Level {int(round(lin['slice_level_ratio']*10))} (Z = {lin['slice_height_z']:.1f} мм / {lin['slice_level_ratio']*100:.0f}% свода)", fontsize=8.1, fontweight="bold", pad=6)
            ax_2d.set_xlabel("Поперечная ось ML (мм)", fontsize=7.1)
            ax_2d.set_ylabel("Переднезадняя ось AP (мм)", fontsize=7.1)
            ax_2d.tick_params(labelsize=6.6)
            ax_2d.set_aspect("equal", "box")
            ax_2d.grid(True, linestyle="--", alpha=0.35)
            ax_2d.legend(loc="lower right", fontsize=5.5, framealpha=0.94, borderpad=0.32)

            # Right Top Plot: Lateral Sagittal Projection
            ax_lat = fig.add_axes([0.56, 0.752, 0.39, 0.124])
            origin = np.array(craniometry["geometry_3d"]["origin"])
            R = np.array(craniometry["geometry_3d"]["axes_rotation_matrix"])
            pts = (mesh.v1 + mesh.v2 + mesh.v3) / 3.0
            step = max(1, len(pts) // 3500)
            local_pts = (pts[::step] - origin) @ R.T

            ax_lat.scatter(local_pts[:, 1], local_pts[:, 2], s=0.7, c="#94A3B8", alpha=0.45)
            ax_lat.axhline(0.0, color="#475569", linestyle="--", linewidth=1.1, label="Базовая пл. (Se-TrR-TrL)")
            ax_lat.axhline(lin["slice_height_z"], color="#0D9488", linestyle="-", linewidth=1.6, label=f"Сечение Z={lin['slice_height_z']:.1f} мм")

            if landmarks:
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
                            fontsize=6.8,
                            fontweight="bold",
                            color=lm_col,
                        )

            ax_lat.set_title("Сагиттальная проекция: ориентиры и уровень среза", fontsize=7.8, fontweight="bold", pad=4)
            ax_lat.set_xlabel("Переднезадняя ось AP (мм)", fontsize=6.6, labelpad=2)
            ax_lat.set_ylabel("Высота Z (мм)", fontsize=6.6)
            ax_lat.tick_params(labelsize=6.4)
            ax_lat.legend(loc="upper left", fontsize=5.6)
            ax_lat.grid(True, linestyle=":", alpha=0.4)

            # Right Bottom Plot: Radial Asymmetry Profile Curve
            ax_asym = fig.add_axes([0.56, 0.568, 0.39, 0.114])
            deg_grid = np.linspace(-180, 179, 360)
            asym_prof = np.array(g2d["asymmetry_profile_mm"])
            ax_asym.axhline(0.0, color="#64748B", linewidth=0.9, linestyle="--")
            ax_asym.fill_between(deg_grid, 0, asym_prof, where=(asym_prof >= 0), color="#F97316", alpha=0.35)
            ax_asym.fill_between(deg_grid, 0, asym_prof, where=(asym_prof < 0), color="#0284C7", alpha=0.35)
            ax_asym.plot(deg_grid, asym_prof, color="#1E293B", linewidth=1.1)
            ax_asym.set_title("Карта радиальной асимметрии контура Δr(θ), мм", fontsize=7.8, fontweight="bold", pad=4)
            ax_asym.set_xlabel("Угол θ от передней оси AP (°)", fontsize=6.6, labelpad=2)
            ax_asym.set_ylabel("Δr (мм)", fontsize=6.6)
            ax_asym.tick_params(labelsize=6.4)
            ax_asym.set_xlim(-180, 180)
            ax_asym.grid(True, linestyle=":", alpha=0.4)

            # ==================== 3. KPI CARDS & CVAI SCALE BAR ====================
            ax_scale = fig.add_axes([0.04, 0.425, 0.92, 0.098])
            ax_scale.set_axis_off()
            card_bg = patches.FancyBboxPatch(
                (0, 0), 1, 1,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.0
            )
            ax_scale.add_patch(card_bg)

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
                ax_scale.text(kx + 0.012, 0.73, k_title, fontsize=6.5, fontweight="bold", color="#475569", va="center")
                ax_scale.text(kx + 0.012, 0.43, k_val, fontsize=13.2, fontweight="bold", color=k_col, va="center")
                ax_scale.text(kx + 0.012, 0.20, k_sub, fontsize=5.7, color="#64748B", va="center")

            ax_scale.text(0.522, 0.80, "Категория по шкале CVAI [Graham et al., 2020]:", fontsize=7.1, fontweight="bold", color="#334155")
            ax_scale.text(0.522, 0.58, cvai_grade["label"], fontsize=8.3, fontweight="bold", color=cvai_grade["color"])

            if borderline.get("is_borderline"):
                b_txt = f"ПОГРАНИЧНАЯ ЗОНА (±0.6 п.п. у {borderline['nearest_threshold_pct']:.2f}%): требуется проверка врача"
                ax_scale.text(0.522, 0.41, b_txt, fontsize=6.1, fontweight="bold", color="#B45309")

            bar_x0, bar_w, bar_y, bar_h = 0.522, 0.450, 0.17, 0.15
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
                ax_scale.text(sx + 0.5 * sw, bar_y - 0.10, s_lbl, fontsize=5.6, ha="center", color="#475569")

            marker_val = min(13.8, max(0.1, ind["CVAI_unrounded"]))
            mx = bar_x0 + (marker_val / 14.0) * bar_w
            ax_scale.plot([mx, mx], [bar_y - 0.02, bar_y + bar_h + 0.04], color="#0F172A", linewidth=2.0)
            ax_scale.scatter([mx], [bar_y + bar_h + 0.05], color="#0F172A", marker="v", s=28, zorder=10)

            # ==================== 4. TWO ISOLATED DATA CARDS (LEFT & RIGHT) ====================
            ax_left = fig.add_axes([0.04, 0.165, 0.45, 0.245])
            ax_left.set_axis_off()
            ax_left.add_patch(patches.FancyBboxPatch(
                (0, 0), 1, 1,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=0.9
            ))

            col_left_lines = [
                f"ЛИНЕЙНЫЕ ИЗМЕРЕНИЯ СЕЧЕНИЯ (LEVEL {int(round(lin['slice_level_ratio']*10))}):",
                f" • Длина (L, AP) / Ширина (W, ML): {lin['L_length_ap']:.1f} / {lin['W_width_ml']:.1f} мм",
                f" • Диагонали D1 (+30°) / D2 (-30°): {lin['D1_diagonal_plus30']:.1f} / {lin['D2_diagonal_minus30']:.1f} мм",
                f" • Периметр среза / Высота Z:      {lin['perimeter_slice']:.1f} / {lin['slice_height_z']:.1f} мм",
                "",
                "РЕГИОНАЛЬНАЯ АСИММЕТРИЯ И КВАДРАНТЫ (Q1–Q4):",
                f" • Фронтальная / Затылочная: {reg['frontal_diff_mm']:.1f} мм / {reg['occipital_diff_mm']:.1f} мм",
                f" • Преобладает: {reg['dominant_asymmetry_zone']}",
            ]
            if qvol:
                col_left_lines.extend([
                    f" • Q1(Л-Лоб)/Q2(П-Лоб): {qvol.get('Q1_front_left_cm3', 0):.0f} / {qvol.get('Q2_front_right_cm3', 0):.0f} см³ (ASR {qvol.get('anterior_symmetry_ratio', 1):.2f})",
                    f" • Q3(Л-Зат)/Q4(П-Зат): {qvol.get('Q3_back_left_cm3', 0):.0f} / {qvol.get('Q4_back_right_cm3', 0):.0f} см³ (PSR {qvol.get('posterior_symmetry_ratio', 1):.2f})",
                    f" • Суммарный объём свода: {qvol.get('total_vault_volume_cm3', 0):.0f} см³",
                ])
            ax_left.text(
                0.032, 0.94,
                "\n".join(col_left_lines),
                fontsize=6.9, color="#1E293B", va="top", family="monospace", linespacing=1.34
            )

            ax_right = fig.add_axes([0.51, 0.165, 0.45, 0.245])
            ax_right.set_axis_off()
            ax_right.add_patch(patches.FancyBboxPatch(
                (0, 0), 1, 1,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=0.9
            ))

            ci_wrapped = textwrap.fill(f" • Пропорции: {cls_info['ci_interpretation']}", width=50, subsequent_indent="   ")
            conf_str = (
                f"Уверенность ML: {landmark_confidence['overall_confidence_pct']:.0f}%"
                if landmark_confidence and "overall_confidence_pct" in landmark_confidence
                else "Режим: стандартный"
            )
            col_right_lines = [
                "ФОРМУЛЫ, ПРОПОРЦИИ CI И КАЧЕСТВО:",
                f" • CI = {ind['CI']:.2f}% | CVA = {ind['CVA_mm']:.2f} мм | CVAI = {ind['CVAI']:.2f}%",
                f" • Справочно CVAI_min: {ind['CVAI_min_ref']:.2f}% | {conf_str}",
                ci_wrapped,
                "",
                "КООРДИНАТЫ ОРИЕНТИРОВ (X, Y, Z в мм):",
            ]
            if landmarks:
                per_lm = (landmark_confidence or {}).get("per_landmark", {})
                for k in ["Sellion", "RightTragion", "LeftTragion", "Subnasal"]:
                    if k in landmarks:
                        c = landmarks[k]
                        cf = per_lm.get(k, {}).get("confidence_pct")
                        cf_txt = f" ({cf:.0f}%)" if cf is not None else ""
                        col_right_lines.append(f" • {k:12s}: {c[0]:5.1f}, {c[1]:6.1f}, {c[2]:5.1f}{cf_txt}")
            col_right_lines.append(
                f" • Масштаб ×{quality.applied_scale:g} ({quality.triangle_count:,} тр., {quality.quality_score:.0f}/100)"
            )
            ax_right.text(
                0.032, 0.94,
                "\n".join(col_right_lines),
                fontsize=6.8, color="#1E293B", va="top", family="monospace", linespacing=1.28
            )

        # ==================== 5. FOOTER & DOCTOR INTERVENTION AUDIT STAMP ====================
        ax_ftr = fig.add_axes([0.04, 0.020, 0.92, 0.132])
        ax_ftr.set_axis_off()
        ftr_box = patches.FancyBboxPatch(
            (0, 0), 1, 1,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor="#F1F5F9", edgecolor="#94A3B8", linewidth=1.0
        )
        ax_ftr.add_patch(ftr_box)

        audit_lm = (
            "Ориентиры: скорректированы врачом вручную + пересчёт"
            if manual_correction_applied
            else "Ориентиры: автопоиск ML (подтверждено врачом)" if doctor_verified else "Ориентиры: автопоиск ML (ожидает проверки врача)"
        )
        ver_status = "ПОДТВЕРЖДЕНО ВРАЧОМ" if doctor_verified else "ТРЕБУЕТ ПОДТВЕРЖДЕНИЯ"
        raw_notes = clinical_notes if clinical_notes else "___________________________________________________________________________"
        notes_wrapped = textwrap.fill(f"Клинический комментарий врача: {raw_notes}", width=114)
        disclaimer_wrapped = textwrap.fill(
            "ДИСКЛЕЙМЕР: Программный инструмент поддержки принятия клинических решений — стандартизирует 3D-краниометрию, но не заменяет очный врачебный осмотр, клинический диагноз или назначение ортезотерапии. Категория CVAI определяется по неокруглённому значению.",
            width=118
        )

        ftr_text = (
            f"МЕТОДИКА: STARBAND-GRAHAM-2020-V1.2 (Graham T. et al., J Clin Med 2020;9(4):1027; FDA K141842)  |  МОДЕЛЬ: {model_version}\n"
            f"АУДИТ ВМЕШАТЕЛЬСТВА: {audit_lm}  |  СТАТУС: [{ver_status}]\n"
            f"{notes_wrapped}\n"
            f"Врач / Специалист по ортезам: {doctor_name}      Дата: {timestamp_str[:10]}      Подпись: ______________________\n"
            f"{disclaimer_wrapped}"
        )
        ax_ftr.text(0.018, 0.93, ftr_text, fontsize=6.4, color="#334155", va="top", linespacing=1.28)

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
    clinical_notes: str = "",
    landmark_confidence: Optional[Dict] = None,
) -> Dict:
    return {
        "scan_id": scan_id,
        "timestamp": datetime.now().isoformat(),
        "patient_age_months": patient_age_months,
        "model_version": model_version,
        "methodology_version": "STARBAND-GRAHAM-2020-V1.2",
        "clinical_disclaimer": "Инструмент поддержки принятия клинических решений; не заменяет врачебный диагноз.",
        "doctor_verification": {
            "verified": doctor_verified,
            "doctor_name": doctor_name,
            "manual_correction_applied": manual_correction_applied,
            "landmarks_audit_mode": "manual_corrected" if manual_correction_applied else "ml_auto",
            "clinical_notes": clinical_notes,
        },
        "input_quality": quality.to_dict(),
        "landmarks_mm": landmarks,
        "landmark_confidence": landmark_confidence,
        "craniometry": {
            "indices": craniometry["indices"],
            "linear_measurements_mm": craniometry["linear_measurements_mm"],
            "regional_asymmetry": craniometry["regional_asymmetry"],
            "quadrant_volumes": craniometry.get("quadrant_volumes"),
            "multi_level_slices": craniometry.get("multi_level_slices"),
            "classification": craniometry["classification"],
        } if craniometry else None,
    }


CSV_FIELDNAMES = [
    "scan_id", "timestamp", "quality_status", "quality_score", "applied_scale",
    "patient_age_months", "CI_pct", "CVA_mm", "CVAI_pct", "CVAI_min_ref_pct",
    "cvai_grade_code", "cvai_grade_label", "is_borderline",
    "L_mm", "W_mm", "D1_mm", "D2_mm", "perimeter_slice_mm",
    "frontal_asymmetry_pct", "occipital_asymmetry_pct",
    "Q1_cm3", "Q2_cm3", "Q3_cm3", "Q4_cm3", "ASR", "PSR",
    "manual_correction", "doctor_verified", "doctor_name", "model_version", "methodology_version",
]


def _flatten_record_row(r: Dict) -> Dict:
    q = r.get("input_quality", {})
    c = r.get("craniometry") or {}
    ind = c.get("indices", {})
    lin = c.get("linear_measurements_mm", {})
    reg = c.get("regional_asymmetry", {})
    qvol = c.get("quadrant_volumes") or {}
    cls_wrap = c.get("classification", {})
    cls_info = cls_wrap.get("cvai_grade", {})
    borderline = cls_wrap.get("borderline_alert", {})
    doc = r.get("doctor_verification", {})
    return {
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
        "is_borderline": borderline.get("is_borderline", ""),
        "L_mm": lin.get("L_length_ap", ""),
        "W_mm": lin.get("W_width_ml", ""),
        "D1_mm": lin.get("D1_diagonal_plus30", ""),
        "D2_mm": lin.get("D2_diagonal_minus30", ""),
        "perimeter_slice_mm": lin.get("perimeter_slice", ""),
        "frontal_asymmetry_pct": reg.get("frontal_asymmetry_pct", ""),
        "occipital_asymmetry_pct": reg.get("occipital_asymmetry_pct", ""),
        "Q1_cm3": qvol.get("Q1_front_left_cm3", ""),
        "Q2_cm3": qvol.get("Q2_front_right_cm3", ""),
        "Q3_cm3": qvol.get("Q3_back_left_cm3", ""),
        "Q4_cm3": qvol.get("Q4_back_right_cm3", ""),
        "ASR": qvol.get("anterior_symmetry_ratio", ""),
        "PSR": qvol.get("posterior_symmetry_ratio", ""),
        "manual_correction": doc.get("manual_correction_applied", False),
        "doctor_verified": doc.get("verified", False),
        "doctor_name": doc.get("doctor_name", ""),
        "model_version": r.get("model_version", ""),
        "methodology_version": r.get("methodology_version", ""),
    }


def export_measurement_csv_string(records: List[Dict]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    for r in records:
        writer.writerow(_flatten_record_row(r))
    return output.getvalue()


def export_measurement_excel_xml(records: List[Dict]) -> str:
    """
    Generate a clean Excel SpreadsheetML (.xls) XML workbook with styled header and typed numeric cells.
    Opens natively in Microsoft Excel and LibreOffice Calc without external openpyxl dependencies.
    """
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<?mso-application progid="Excel.Sheet"?>',
        '<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"',
        ' xmlns:o="urn:schemas-microsoft-com:office:office"',
        ' xmlns:x="urn:schemas-microsoft-com:office:excel"',
        ' xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">',
        ' <Styles>',
        '  <Style ss:ID="hdr"><Font ss:Bold="1" ss:Color="#FFFFFF"/><Interior ss:Color="#111827" ss:Pattern="Solid"/></Style>',
        '  <Style ss:ID="num"><NumberFormat ss:Format="0.00"/></Style>',
        ' </Styles>',
        ' <Worksheet ss:Name="Saigak-Orto Craniometry">',
        '  <Table>',
        '   <Row>',
    ]
    for col in CSV_FIELDNAMES:
        lines.append(f'    <Cell ss:StyleID="hdr"><Data ss:Type="String">{xml_escape(col)}</Data></Cell>')
    lines.append('   </Row>')

    for r in records:
        row = _flatten_record_row(r)
        lines.append('   <Row>')
        for col in CSV_FIELDNAMES:
            val = row.get(col, "")
            if isinstance(val, bool):
                lines.append(f'    <Cell><Data ss:Type="String">{"Да" if val else "Нет"}</Data></Cell>')
            elif isinstance(val, (int, float)) and val != "":
                lines.append(f'    <Cell ss:StyleID="num"><Data ss:Type="Number">{val}</Data></Cell>')
            else:
                lines.append(f'    <Cell><Data ss:Type="String">{xml_escape(str(val if val is not None else ""))}</Data></Cell>')
        lines.append('   </Row>')

    lines.extend([
        '  </Table>',
        ' </Worksheet>',
        '</Workbook>',
    ])
    return "\n".join(lines)
