"""
Deterministic 3D Craniometry Engine & Clinical Classification Module.
Implements the STARband / Graham et al. (2020) measurement protocol:
- Base plane defined by Sellion (Nasion), RightTragion, LeftTragion
- Measurement cross-section parallel to base plane at level_ratio of vault height (default 0.30 = Level 3)
- Exact calculation of L, W, D1 (30 deg), D2 (-30 deg), Perimeter, CI, CVA, CVAI
- Separate Frontal and Occipital asymmetry evaluation & 3D Surface Asymmetry Heatmap
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from core.mesh_io import MeshData


METHODOLOGY_REGISTRY = {
    "id": "STARBAND-GRAHAM-2020-V1.2",
    "name": "Протокол краниометрии по 3D-скану (базовая плоскость Se–TrR–TrL, сечение Level 3 / 30° диагонали)",
    "source": "Graham T. et al. Significant Factors in Cranial Remolding Orthotic Treatment of Asymmetrical Brachycephaly. J Clin Med. 2020;9(4):1027. DOI: 10.3390/jcm9041027; FDA K141842.",
    "thresholds_version": "Graham-2020-STARband-v1.0",
    "applicable_age_months": "0–18 месяцев (шлемотерапия преимущественно 3–18 мес. по решению врача)",
    "approval_status": "Базовый профиль CVAI утверждён по публикации [2]; профиль порогов CI ожидает утверждения клиническим экспертом",
    "formulas": {
        "CI": "CI = 100 × W / L (%)",
        "CVA": "CVA = |D1 − D2| (мм)",
        "CVAI": "CVAI = 100 × |D1 − D2| / max(D1, D2) (%)",
        "CVAI_min_ref": "CVAI_min = 100 × |D1 − D2| / min(D1, D2) (%) [справочная метрика]",
    },
    "cvai_scale": [
        {"min": 0.0, "max": 3.5, "level": 0, "code": "NORMAL", "label": "Ниже порога асимметрии по шкале (< 3,5%)", "color": "#0D9488"},
        {"min": 3.5, "max": 6.25, "level": 1, "code": "MILD", "label": "Лёгкая асимметрия (3,5% – 6,25%)", "color": "#65A30D"},
        {"min": 6.25, "max": 8.75, "level": 2, "code": "MODERATE", "label": "Умеренная асимметрия (6,25% – 8,75%)", "color": "#D97706"},
        {"min": 8.75, "max": 11.0, "level": 3, "code": "SEVERE", "label": "Тяжёлая асимметрия (8,75% – 11,0%)", "color": "#EA580C"},
        {"min": 11.0, "max": 100.0, "level": 4, "code": "VERY_SEVERE", "label": "Очень тяжёлая асимметрия (≥ 11,0%)", "color": "#DC2626"},
    ],
}


def classify_cvai(cvai_unrounded: float) -> Dict:
    """
    Classify asymmetry severity strictly by the unrounded CVAI value
    according to Section 6.2 of the Technical Specification.
    """
    if cvai_unrounded < 3.5:
        item = METHODOLOGY_REGISTRY["cvai_scale"][0]
    elif cvai_unrounded < 6.25:
        item = METHODOLOGY_REGISTRY["cvai_scale"][1]
    elif cvai_unrounded < 8.75:
        item = METHODOLOGY_REGISTRY["cvai_scale"][2]
    elif cvai_unrounded < 11.0:
        item = METHODOLOGY_REGISTRY["cvai_scale"][3]
    else:
        item = METHODOLOGY_REGISTRY["cvai_scale"][4]
    return dict(item)


def build_cranial_frame(
    sellion: np.ndarray,
    right_tragion: np.ndarray,
    left_tragion: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct orthonormal right-handed cranial coordinate system from landmarks
    following the STARband / Graham et al. (2020) clinical protocol:
      - Origin: Midpoint O between RightTragion and LeftTragion
      - Base Plane Normal Z_local (e_cc): Normal to the plane passing through (Sellion, RightTragion, LeftTragion)
      - Y_local (e_ap): Anterior-Posterior midline unit vector from O directly through Sellion
      - X_local (e_ml): Medial-Lateral unit vector perpendicular to e_ap in the base plane (Right -> Left)
    Returns:
      origin (3,), R (3, 3) where rows are [e_ml, e_ap, e_cc]
    """
    se = np.asarray(sellion, dtype=np.float64).flatten()
    rt = np.asarray(right_tragion, dtype=np.float64).flatten()
    lt = np.asarray(left_tragion, dtype=np.float64).flatten()

    origin = 0.5 * (rt + lt)
    vec_ml_raw = lt - rt
    if np.linalg.norm(vec_ml_raw) < 1e-6:
        raise ValueError("Точки правого и левого козелков (RightTragion, LeftTragion) совпадают.")

    vec_se = se - origin
    norm_se = np.linalg.norm(vec_se)
    if norm_se < 1e-6:
        raise ValueError("Точка Sellion совпадает с серединой битрагиальной линии.")

    # Plane normal: e_ap x vec_ml_raw points superiorly (towards cranial vertex)
    e_ap = vec_se / norm_se
    e_cc = np.cross(e_ap, vec_ml_raw)
    norm_cc = np.linalg.norm(e_cc)
    if norm_cc < 1e-6:
        raise ValueError("Ориентиры Sellion, RightTragion, LeftTragion коллинеарны.")
    e_cc = e_cc / norm_cc
    if e_cc[2] < 0:
        e_cc = -e_cc

    # Medial-Lateral axis strictly orthogonal to AP midline inside the base plane
    e_ml = np.cross(e_cc, e_ap)
    e_ml = e_ml / np.linalg.norm(e_ml)

    R = np.vstack([e_ml, e_ap, e_cc])
    return origin, R


def compute_craniometry(
    mesh: MeshData,
    landmarks: Dict[str, List[float]],
    level_ratio: float = 0.30,
    diagonal_angle_deg: float = 30.0,
    expert_ci_profile_approved: bool = False,
) -> Dict:
    """
    Perform full deterministic craniometric measurement on a 3D mesh given landmarks.
    All linear outputs are in mm, indices in %.
    """
    se = np.array(landmarks["Sellion"], dtype=np.float64)
    rt = np.array(landmarks["RightTragion"], dtype=np.float64)
    lt = np.array(landmarks["LeftTragion"], dtype=np.float64)

    origin, R = build_cranial_frame(se, rt, lt)

    # Transform triangle vertices into local cranial coordinate frame
    p1 = (mesh.v1 - origin) @ R.T
    p2 = (mesh.v2 - origin) @ R.T
    p3 = (mesh.v3 - origin) @ R.T

    z_max = float(max(p1[:, 2].max(), p2[:, 2].max(), p3[:, 2].max()))
    z_min = float(min(p1[:, 2].min(), p2[:, 2].min(), p3[:, 2].min()))
    if z_max <= 10.0:
        raise ValueError(f"Недостаточная высота свода черепа над базовой плоскостью ({z_max:.1f} мм).")

    level_ratio = float(np.clip(level_ratio, 0.12, 0.70))
    z_slice = level_ratio * z_max

    # Exact edge-plane intersections at Z_local = z_slice
    inter_segments = []
    for pa, pb in [(p1, p2), (p2, p3), (p3, p1)]:
        za = pa[:, 2] - z_slice
        zb = pb[:, 2] - z_slice
        mask = (za * zb < 0.0) | (za == 0.0)
        if np.any(mask):
            denom = za[mask] - zb[mask]
            denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
            t = za[mask] / denom
            valid_t = (t >= 0.0) & (t <= 1.0)
            if np.any(valid_t):
                idx = np.where(mask)[0][valid_t]
                t_val = t[valid_t][:, None]
                xy = pa[idx, :2] + t_val * (pb[idx, :2] - pa[idx, :2])
                inter_segments.append(xy)

    if not inter_segments:
        raise ValueError("Секущая плоскость не пересекает полигональную сетку на выбранном уровне.")

    raw_pts = np.vstack(inter_segments)
    if len(raw_pts) < 60:
        raise ValueError(f"В сечении недостаточно точек пересечения ({len(raw_pts)}) для достоверного расчёта.")

    # Center coordinate origin in the cross-section plane at the intersection of AP and ML midpoints
    y_back = float(np.percentile(raw_pts[:, 1], 0.5))
    y_front = float(np.percentile(raw_pts[:, 1], 99.5))
    y_mid = 0.5 * (y_back + y_front)
    x_right = float(np.percentile(raw_pts[:, 0], 0.5))
    x_left = float(np.percentile(raw_pts[:, 0], 99.5))
    x_mid = 0.5 * (x_right + x_left)

    section_center_2d = np.array([x_mid, y_mid], dtype=np.float64)
    centered_pts = raw_pts - section_center_2d

    # Polar representation: theta = 0 is +Y_local (Anterior / Glabella),
    # theta = +90 deg is +X_local (Left ear side), theta = -90 deg is -X_local (Right ear side)
    angles_rad = np.arctan2(centered_pts[:, 0], centered_pts[:, 1])
    radii_mm = np.linalg.norm(centered_pts, axis=1)

    # Check angular completeness of the cross-section (36 bins of 10 degrees)
    hist, _ = np.histogram(angles_rad, bins=36, range=(-np.pi, np.pi))
    empty_bins = int(np.sum(hist == 0))
    if empty_bins > 3:
        raise ValueError(
            f"Контур измерительного сечения незамкнут: отсутствуют данные в {empty_bins * 10}° углового диапазона."
        )

    # Sample smooth, deterministic 360-point polar contour (1 degree resolution) using Gaussian angular kernel
    deg_grid = np.linspace(-180.0, 179.0, 360)
    contour_radii = np.zeros(360, dtype=np.float64)
    sigma_rad = np.deg2rad(2.0)
    for i, d in enumerate(deg_grid):
        tr = np.deg2rad(d)
        ang_diff = np.abs(np.arctan2(np.sin(angles_rad - tr), np.cos(angles_rad - tr)))
        m = ang_diff <= np.deg2rad(6.0)
        if np.any(m):
            w = np.exp(-0.5 * (ang_diff[m] / sigma_rad) ** 2)
            contour_radii[i] = float(np.sum(radii_mm[m] * w) / np.sum(w))
        else:
            contour_radii[i] = float(np.median(radii_mm))

    rad_grid = np.deg2rad(deg_grid)
    contour_2d = np.column_stack([
        section_center_2d[0] + contour_radii * np.sin(rad_grid),
        section_center_2d[1] + contour_radii * np.cos(rad_grid),
    ])

    # Compute exact polygonal perimeter of the closed 2D cross-section
    diffs = np.roll(contour_2d, -1, axis=0) - contour_2d
    perimeter_mm = float(np.sum(np.linalg.norm(diffs, axis=1)))

    # Helper to evaluate smooth kernel-weighted radius at exact angle (in degrees from +AP axis)
    def radius_at_angle(target_deg: float) -> Tuple[float, np.ndarray]:
        tr = np.deg2rad(target_deg)
        ang_diff = np.abs(np.arctan2(np.sin(angles_rad - tr), np.cos(angles_rad - tr)))
        m = ang_diff <= np.deg2rad(5.5)
        if not np.any(m):
            m = ang_diff <= np.deg2rad(10.0)
        w = np.exp(-0.5 * (ang_diff[m] / np.deg2rad(1.8)) ** 2)
        r = float(np.sum(radii_mm[m] * w) / np.sum(w))
        pt_2d = section_center_2d + np.array([r * np.sin(tr), r * np.cos(tr)])
        return r, pt_2d

    # 1. AP Length (L): 0 deg (Anterior/Glabella) <-> 180 deg (Posterior/Opisthocranion)
    r_front, pt_front = radius_at_angle(0.0)
    r_back, pt_back = radius_at_angle(180.0)
    L_mm = r_front + r_back

    # 2. ML Width (W): +90 deg (Left Euryon) <-> -90 deg (Right Euryon)
    r_left, pt_left = radius_at_angle(90.0)
    r_right, pt_right = radius_at_angle(-90.0)
    W_mm = r_left + r_right

    # 3. Oblique Diagonals at +/- 30 degrees to AP axis:
    # D1: +30 deg (Left-Anterior) <-> -150 deg (Right-Posterior)
    alpha = float(diagonal_angle_deg)
    r_d1_front, pt_d1_front = radius_at_angle(alpha)
    r_d1_back, pt_d1_back = radius_at_angle(alpha - 180.0)
    D1_mm = r_d1_front + r_d1_back

    # D2: -30 deg (Right-Anterior) <-> +150 deg (Left-Posterior)
    r_d2_front, pt_d2_front = radius_at_angle(-alpha)
    r_d2_back, pt_d2_back = radius_at_angle(180.0 - alpha)
    D2_mm = r_d2_front + r_d2_back

    # 4. Cranial Indices (Section 6.1)
    ci_unrounded = 100.0 * W_mm / max(L_mm, 1e-6)
    cva_unrounded = abs(D1_mm - D2_mm)
    cvai_unrounded = 100.0 * cva_unrounded / max(D1_mm, D2_mm, 1e-6)
    cvai_min_unrounded = 100.0 * cva_unrounded / max(min(D1_mm, D2_mm), 1e-6)

    # 5. Separate Frontal and Occipital Asymmetry (Should 5.2)
    frontal_diff_mm = abs(r_d1_front - r_d2_front)
    frontal_asymmetry_pct = 100.0 * frontal_diff_mm / max(r_d1_front, r_d2_front, 1e-6)
    occipital_diff_mm = abs(r_d1_back - r_d2_back)
    occipital_asymmetry_pct = 100.0 * occipital_diff_mm / max(r_d1_back, r_d2_back, 1e-6)

    # 6. Radial Asymmetry Profile (Left vs Right mirror difference in mm)
    # For angle theta in [0..180], compare r(+theta) vs r(-theta)
    asymmetry_profile_mm = []
    for i, d in enumerate(deg_grid):
        mirror_d = -d
        if mirror_d == 180.0:
            mirror_d = -180.0
        mirror_idx = int(np.argmin(np.abs(deg_grid - mirror_d)))
        asymmetry_profile_mm.append(round(float(contour_radii[i] - contour_radii[mirror_idx]), 3))

    # Helper to convert 2D cross-section points back to 3D world space for Three.js visualization
    def to_world_3d(pt2: np.ndarray) -> List[float]:
        local_3d = np.array([pt2[0], pt2[1], z_slice], dtype=np.float64)
        world_3d = origin + local_3d @ R
        return [round(float(x), 3) for x in world_3d]

    contour_3d = [to_world_3d(pt) for pt in contour_2d[::2]]  # 180 points for smooth 3D ring

    cvai_classification = classify_cvai(cvai_unrounded)

    # Proportional description for CI (Section 6.2: output numerically without automatic severity until expert approves profile)
    if expert_ci_profile_approved:
        if ci_unrounded < 75.0:
            ci_note = f"CI = {ci_unrounded:.2f}% (долихоцефалическая пропорция по экспертному профилю)"
        elif ci_unrounded <= 85.0:
            ci_note = f"CI = {ci_unrounded:.2f}% (мезоцефалическая пропорция / норма по экспертному профилю)"
        else:
            ci_note = f"CI = {ci_unrounded:.2f}% (брахицефалическая пропорция по экспертному профилю)"
    else:
        ci_note = (
            f"CI = {ci_unrounded:.2f}% (выводится численно без автоматической степени тяжести до утверждения возрастного профиля экспертом)"
        )

    return {
        "indices": {
            "CI": round(ci_unrounded, 2),
            "CI_unrounded": ci_unrounded,
            "CVA_mm": round(cva_unrounded, 2),
            "CVA_unrounded": cva_unrounded,
            "CVAI": round(cvai_unrounded, 2),
            "CVAI_unrounded": cvai_unrounded,
            "CVAI_min_ref": round(cvai_min_unrounded, 2),
        },
        "linear_measurements_mm": {
            "L_length_ap": round(L_mm, 2),
            "W_width_ml": round(W_mm, 2),
            "D1_diagonal_plus30": round(D1_mm, 2),
            "D2_diagonal_minus30": round(D2_mm, 2),
            "perimeter_slice": round(perimeter_mm, 2),
            "perimeter_label": "Периметр измерительного сечения (Level 3)",
            "vault_height": round(z_max, 2),
            "slice_height_z": round(z_slice, 2),
            "slice_level_ratio": round(level_ratio, 3),
            "diagonal_angle_deg": round(alpha, 1),
        },
        "regional_asymmetry": {
            "frontal_diff_mm": round(frontal_diff_mm, 2),
            "frontal_asymmetry_pct": round(frontal_asymmetry_pct, 2),
            "occipital_diff_mm": round(occipital_diff_mm, 2),
            "occipital_asymmetry_pct": round(occipital_asymmetry_pct, 2),
            "r_d1_front_mm": round(r_d1_front, 2),
            "r_d2_front_mm": round(r_d2_front, 2),
            "r_d1_back_mm": round(r_d1_back, 2),
            "r_d2_back_mm": round(r_d2_back, 2),
            "dominant_asymmetry_zone": (
                "Затылочная область" if occipital_diff_mm >= frontal_diff_mm else "Лобно-височная область"
            ),
        },
        "classification": {
            "cvai_grade": cvai_classification,
            "ci_interpretation": ci_note,
            "expert_ci_profile_approved": expert_ci_profile_approved,
        },
        "geometry_3d": {
            "origin": [round(float(x), 3) for x in origin],
            "axes_rotation_matrix": [[round(float(v), 5) for v in row] for row in R],
            "slice_center_world": to_world_3d(section_center_2d),
            "contour_3d": contour_3d,
            "axes_endpoints_3d": {
                "ap_front": to_world_3d(pt_front),
                "ap_back": to_world_3d(pt_back),
                "ml_left": to_world_3d(pt_left),
                "ml_right": to_world_3d(pt_right),
                "d1_front": to_world_3d(pt_d1_front),
                "d1_back": to_world_3d(pt_d1_back),
                "d2_front": to_world_3d(pt_d2_front),
                "d2_back": to_world_3d(pt_d2_back),
            },
        },
        "geometry_2d": {
            "center": [round(float(x), 3) for x in section_center_2d],
            "contour_xy": [[round(float(pt[0]), 2), round(float(pt[1]), 2)] for pt in contour_2d],
            "asymmetry_profile_mm": asymmetry_profile_mm,
            "endpoints_xy": {
                "ap_front": [round(float(pt_front[0]), 2), round(float(pt_front[1]), 2)],
                "ap_back": [round(float(pt_back[0]), 2), round(float(pt_back[1]), 2)],
                "ml_left": [round(float(pt_left[0]), 2), round(float(pt_left[1]), 2)],
                "ml_right": [round(float(pt_right[0]), 2), round(float(pt_right[1]), 2)],
                "d1_front": [round(float(pt_d1_front[0]), 2), round(float(pt_d1_front[1]), 2)],
                "d1_back": [round(float(pt_d1_back[0]), 2), round(float(pt_d1_back[1]), 2)],
                "d2_front": [round(float(pt_d2_front[0]), 2), round(float(pt_d2_front[1]), 2)],
                "d2_back": [round(float(pt_d2_back[0]), 2), round(float(pt_d2_back[1]), 2)],
            },
        },
        "methodology": METHODOLOGY_REGISTRY,
    }
