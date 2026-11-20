"""
Deterministic 3D Craniometry Engine & Clinical Classification Module.
Implements the STARband / Graham et al. (2020) measurement protocol:
- Base plane defined by Sellion (Nasion), RightTragion, LeftTragion
- Measurement cross-section parallel to base plane at level_ratio of vault height (default 0.30 = Level 3)
- Exact calculation of L, W, D1 (30 deg), D2 (-30 deg), Perimeter, CI, CVA, CVAI
- Automatic CVAI borderline threshold alert (+/-0.60 pp around 3.5, 6.25, 8.75, 11.0%)
- 4-Quadrant cranial vault volumes (Q1-Q4 in cm^3), ASR & PSR symmetry ratios
- Multi-level cross-section summary (20% Level 2, 30% Level 3, 50% Level 5)
- Dynamic longitudinal Before/After scan comparison in canonical cranial coordinates
"""
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
    "borderline_margin_pp": 0.60,
    "formulas": {
        "CI": "CI = 100 × W / L (%)",
        "CVA": "CVA = |D1 − D2| (мм)",
        "CVAI": "CVAI = 100 × |D1 − D2| / max(D1, D2) (%)",
        "CVAI_min_ref": "CVAI_min = 100 × |D1 − D2| / min(D1, D2) (%) [справочная метрика]",
        "ASR": "ASR = min(Q1, Q2) / max(Q1, Q2) [передняя объёмная симметрия]",
        "PSR": "PSR = min(Q3, Q4) / max(Q3, Q4) [задняя объёмная симметрия]",
    },
    "cvai_scale": [
        {"min": 0.0, "max": 3.5, "level": 0, "code": "NORMAL", "label": "Ниже порога асимметрии по шкале (< 3,5%)", "color": "#0D9488"},
        {"min": 3.5, "max": 6.25, "level": 1, "code": "MILD", "label": "Лёгкая асимметрия (3,5% – 6,25%)", "color": "#65A30D"},
        {"min": 6.25, "max": 8.75, "level": 2, "code": "MODERATE", "label": "Умеренная асимметрия (6,25% – 8,75%)", "color": "#D97706"},
        {"min": 8.75, "max": 11.0, "level": 3, "code": "SEVERE", "label": "Тяжёлая асимметрия (8,75% – 11,0%)", "color": "#EA580C"},
        {"min": 11.0, "max": 100.0, "level": 4, "code": "VERY_SEVERE", "label": "Очень тяжёлая асимметрия (≥ 11,0%)", "color": "#DC2626"},
    ],
}

CVAI_THRESHOLDS = [
    (3.50, 0, 1),
    (6.25, 1, 2),
    (8.75, 2, 3),
    (11.00, 3, 4),
]


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


def evaluate_cvai_borderline(cvai_unrounded: float, margin_pp: float = 0.60) -> Dict:
    """
    Flag cases falling within +/- margin_pp (default 0.60 pp, matching CVAI MAE 0.56 pp)
    of any Graham et al. (2020) severity threshold (3.50%, 6.25%, 8.75%, 11.00%)
    so the clinician visually verifies landmarks instead of relying blindly on the tier.
    """
    scale = METHODOLOGY_REGISTRY["cvai_scale"]
    nearest_thr = None
    min_dist = float("inf")
    adj_grade = None

    for thr, low_idx, high_idx in CVAI_THRESHOLDS:
        dist = abs(cvai_unrounded - thr)
        if dist < min_dist:
            min_dist = dist
            nearest_thr = thr
            adj_grade = scale[high_idx] if cvai_unrounded < thr else scale[low_idx]

    is_borderline = min_dist <= margin_pp
    if is_borderline and adj_grade is not None:
        msg = (
            f"Пограничное значение (±{margin_pp:.2f} п.п. у порога {nearest_thr:.2f}%): "
            f"CVAI = {cvai_unrounded:.2f}% отстоит на {min_dist:.2f} п.п. от границы с категорией «{adj_grade['label']}». "
            f"Требуется визуальная проверка ориентиров Se–TrR–TrL врачом."
        )
    else:
        msg = (
            f"Устойчивая категория: расстояние до ближайшего порога ({nearest_thr:.2f}%) составляет {min_dist:.2f} п.п. (> ±{margin_pp:.2f} п.п.)."
        )

    return {
        "is_borderline": bool(is_borderline),
        "margin_pp": round(margin_pp, 2),
        "nearest_threshold_pct": nearest_thr,
        "distance_to_threshold_pp": round(float(min_dist), 3),
        "adjacent_grade_code": adj_grade["code"] if adj_grade else None,
        "adjacent_grade_label": adj_grade["label"] if adj_grade else None,
        "warning_message": msg,
    }


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


def _extract_slice_metrics(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    z_max: float,
    level_ratio: float,
    diagonal_angle_deg: float = 30.0,
) -> Dict:
    """
    Extract exact 2D cross-section contour and linear/index metrics at a given level_ratio of vault height.
    """
    level_ratio = float(np.clip(level_ratio, 0.12, 0.70))
    z_slice = level_ratio * z_max

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

    y_back = float(np.percentile(raw_pts[:, 1], 0.5))
    y_front = float(np.percentile(raw_pts[:, 1], 99.5))
    y_mid = 0.5 * (y_back + y_front)
    x_right = float(np.percentile(raw_pts[:, 0], 0.5))
    x_left = float(np.percentile(raw_pts[:, 0], 99.5))
    x_mid = 0.5 * (x_right + x_left)

    section_center_2d = np.array([x_mid, y_mid], dtype=np.float64)
    centered_pts = raw_pts - section_center_2d

    angles_rad = np.arctan2(centered_pts[:, 0], centered_pts[:, 1])
    radii_mm = np.linalg.norm(centered_pts, axis=1)

    hist, _ = np.histogram(angles_rad, bins=36, range=(-np.pi, np.pi))
    empty_bins = int(np.sum(hist == 0))
    if empty_bins > 3:
        raise ValueError(
            f"Контур измерительного сечения незамкнут: отсутствуют данные в {empty_bins * 10}° углового диапазона."
        )

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

    diffs = np.roll(contour_2d, -1, axis=0) - contour_2d
    perimeter_mm = float(np.sum(np.linalg.norm(diffs, axis=1)))

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

    r_front, pt_front = radius_at_angle(0.0)
    r_back, pt_back = radius_at_angle(180.0)
    L_mm = r_front + r_back

    r_left, pt_left = radius_at_angle(90.0)
    r_right, pt_right = radius_at_angle(-90.0)
    W_mm = r_left + r_right

    alpha = float(diagonal_angle_deg)
    r_d1_front, pt_d1_front = radius_at_angle(alpha)
    r_d1_back, pt_d1_back = radius_at_angle(alpha - 180.0)
    D1_mm = r_d1_front + r_d1_back

    r_d2_front, pt_d2_front = radius_at_angle(-alpha)
    r_d2_back, pt_d2_back = radius_at_angle(180.0 - alpha)
    D2_mm = r_d2_front + r_d2_back

    ci_unrounded = 100.0 * W_mm / max(L_mm, 1e-6)
    cva_unrounded = abs(D1_mm - D2_mm)
    cvai_unrounded = 100.0 * cva_unrounded / max(D1_mm, D2_mm, 1e-6)
    cvai_min_unrounded = 100.0 * cva_unrounded / max(min(D1_mm, D2_mm), 1e-6)

    return {
        "level_ratio": level_ratio,
        "z_slice": z_slice,
        "section_center_2d": section_center_2d,
        "deg_grid": deg_grid,
        "contour_radii": contour_radii,
        "contour_2d": contour_2d,
        "perimeter_mm": perimeter_mm,
        "L_mm": L_mm,
        "W_mm": W_mm,
        "D1_mm": D1_mm,
        "D2_mm": D2_mm,
        "r_d1_front": r_d1_front,
        "r_d2_front": r_d2_front,
        "r_d1_back": r_d1_back,
        "r_d2_back": r_d2_back,
        "pt_front": pt_front,
        "pt_back": pt_back,
        "pt_left": pt_left,
        "pt_right": pt_right,
        "pt_d1_front": pt_d1_front,
        "pt_d1_back": pt_d1_back,
        "pt_d2_front": pt_d2_front,
        "pt_d2_back": pt_d2_back,
        "ci_unrounded": ci_unrounded,
        "cva_unrounded": cva_unrounded,
        "cvai_unrounded": cvai_unrounded,
        "cvai_min_unrounded": cvai_min_unrounded,
    }


def compute_quadrant_volumes(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    z_max: float,
    section_center_2d: np.ndarray,
) -> Dict:
    """
    Compute 4-quadrant cranial vault volumes (Q1-Q4 in cm^3) above the base plane (Z_local in [0, z_max])
    using cylindrical polar integration across 20 height layers and 72 angular sectors (5 deg each):
      - Q1 (Front-Left):  +X_local, +Y_local (theta in [0, +90 deg])
      - Q2 (Front-Right): -X_local, +Y_local (theta in [-90, 0 deg])
      - Q3 (Back-Left):   +X_local, -Y_local (theta in [+90, +180 deg])
      - Q4 (Back-Right):  -X_local, -Y_local (theta in [-180, -90 deg])
    Also computes clinical Anterior Symmetry Ratio (ASR), Posterior Symmetry Ratio (PSR),
    and Anterior-Posterior Volume Ratio (APVR).
    """
    centroids = (p1 + p2 + p3) / 3.0
    vault_mask = (centroids[:, 2] >= 0.0) & (centroids[:, 2] <= z_max)
    vpts = centroids[vault_mask]
    if len(vpts) < 100:
        return {
            "Q1_front_left_cm3": 0.0,
            "Q2_front_right_cm3": 0.0,
            "Q3_back_left_cm3": 0.0,
            "Q4_back_right_cm3": 0.0,
            "total_vault_volume_cm3": 0.0,
            "anterior_symmetry_ratio": 1.0,
            "posterior_symmetry_ratio": 1.0,
            "ap_volume_ratio": 1.0,
            "flattened_quadrant_code": "NONE",
            "flattened_quadrant_label": "Нет данных",
        }

    x = vpts[:, 0] - section_center_2d[0]
    y = vpts[:, 1] - section_center_2d[1]
    z = vpts[:, 2]

    radii = np.hypot(x, y)
    angles = np.arctan2(x, y)  # [-pi, pi], 0 = +Y (front), +pi/2 = +X (left), -pi/2 = -X (right)

    n_z = 20
    n_a = 72
    dz = z_max / n_z
    dtheta = (2.0 * np.pi) / n_a

    z_idx = np.clip((z / max(z_max, 1e-6) * n_z).astype(int), 0, n_z - 1)
    a_idx = np.clip(((angles + np.pi) / (2.0 * np.pi) * n_a).astype(int), 0, n_a - 1)

    grid_r = np.zeros((n_z, n_a), dtype=np.float64)
    grid_cnt = np.zeros((n_z, n_a), dtype=np.int32)
    np.add.at(grid_r, (z_idx, a_idx), radii)
    np.add.at(grid_cnt, (z_idx, a_idx), 1)
    valid = grid_cnt > 0
    grid_r[valid] /= grid_cnt[valid]

    # Fill sparse cap cells per Z-row with row median so integration stays smooth near vertex
    for zi in range(n_z):
        row_v = valid[zi]
        if np.any(row_v) and not np.all(row_v):
            grid_r[zi, ~row_v] = float(np.median(grid_r[zi, row_v]))

    # Sector volume element in mm^3: 0.5 * r^2 * dtheta * dz -> convert to cm^3 (/ 1000)
    vol_grid_cm3 = 0.5 * (grid_r ** 2) * dtheta * dz / 1000.0
    bin_angles_deg = -180.0 + (np.arange(n_a) + 0.5) * (360.0 / n_a)

    q1 = float(vol_grid_cm3[:, (bin_angles_deg >= 0.0) & (bin_angles_deg < 90.0)].sum())
    q2 = float(vol_grid_cm3[:, (bin_angles_deg >= -90.0) & (bin_angles_deg < 0.0)].sum())
    q3 = float(vol_grid_cm3[:, bin_angles_deg >= 90.0].sum())
    q4 = float(vol_grid_cm3[:, bin_angles_deg < -90.0].sum())

    total = q1 + q2 + q3 + q4
    asr = min(q1, q2) / max(q1, q2, 1e-6)
    psr = min(q3, q4) / max(q3, q4, 1e-6)
    apvr = (q1 + q2) / max(q3 + q4, 1e-6)

    q_map = {
        "Q1": (q1, "Q1: Левый лобный"),
        "Q2": (q2, "Q2: Правый лобный"),
        "Q3": (q3, "Q3: Левый затылочный"),
        "Q4": (q4, "Q4: Правый затылочный"),
    }
    # Determine dominant flattened quadrant (check occipital first if PSR <= ASR)
    if psr <= asr:
        flat_code = "Q4" if q4 <= q3 else "Q3"
    else:
        flat_code = "Q2" if q2 <= q1 else "Q1"

    return {
        "Q1_front_left_cm3": round(q1, 1),
        "Q2_front_right_cm3": round(q2, 1),
        "Q3_back_left_cm3": round(q3, 1),
        "Q4_back_right_cm3": round(q4, 1),
        "total_vault_volume_cm3": round(total, 1),
        "anterior_symmetry_ratio": round(float(asr), 3),
        "posterior_symmetry_ratio": round(float(psr), 3),
        "ap_volume_ratio": round(float(apvr), 3),
        "flattened_quadrant_code": flat_code,
        "flattened_quadrant_label": q_map[flat_code][1],
    }


def compute_craniometry(
    mesh: MeshData,
    landmarks: Dict[str, List[float]],
    level_ratio: float = 0.30,
    diagonal_angle_deg: float = 30.0,
    expert_ci_profile_approved: bool = False,
    include_aux_levels: bool = True,
) -> Dict:
    """
    Perform full deterministic craniometric measurement on a 3D mesh given landmarks.
    All linear outputs are in mm, indices in %, volumes in cm^3.
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
    if z_max <= 10.0:
        raise ValueError(f"Недостаточная высота свода черепа над базовой плоскостью ({z_max:.1f} мм).")

    sl = _extract_slice_metrics(p1, p2, p3, z_max, level_ratio, diagonal_angle_deg)

    z_slice = sl["z_slice"]
    section_center_2d = sl["section_center_2d"]
    deg_grid = sl["deg_grid"]
    contour_radii = sl["contour_radii"]
    contour_2d = sl["contour_2d"]
    perimeter_mm = sl["perimeter_mm"]
    L_mm = sl["L_mm"]
    W_mm = sl["W_mm"]
    D1_mm = sl["D1_mm"]
    D2_mm = sl["D2_mm"]
    ci_unrounded = sl["ci_unrounded"]
    cva_unrounded = sl["cva_unrounded"]
    cvai_unrounded = sl["cvai_unrounded"]
    cvai_min_unrounded = sl["cvai_min_unrounded"]

    # Separate Frontal and Occipital Asymmetry (Should 5.2)
    frontal_diff_mm = abs(sl["r_d1_front"] - sl["r_d2_front"])
    frontal_asymmetry_pct = 100.0 * frontal_diff_mm / max(sl["r_d1_front"], sl["r_d2_front"], 1e-6)
    occipital_diff_mm = abs(sl["r_d1_back"] - sl["r_d2_back"])
    occipital_asymmetry_pct = 100.0 * occipital_diff_mm / max(sl["r_d1_back"], sl["r_d2_back"], 1e-6)

    # Radial Asymmetry Profile (Left vs Right mirror difference in mm)
    asymmetry_profile_mm = []
    for i, d in enumerate(deg_grid):
        mirror_d = -d
        if mirror_d == 180.0:
            mirror_d = -180.0
        mirror_idx = int(np.argmin(np.abs(deg_grid - mirror_d)))
        asymmetry_profile_mm.append(round(float(contour_radii[i] - contour_radii[mirror_idx]), 3))

    def to_world_3d(pt2: np.ndarray) -> List[float]:
        local_3d = np.array([pt2[0], pt2[1], z_slice], dtype=np.float64)
        world_3d = origin + local_3d @ R
        return [round(float(x), 3) for x in world_3d]

    contour_3d = [to_world_3d(pt) for pt in contour_2d[::2]]

    cvai_classification = classify_cvai(cvai_unrounded)
    borderline_alert = evaluate_cvai_borderline(cvai_unrounded, margin_pp=0.60)
    quadrant_volumes = compute_quadrant_volumes(p1, p2, p3, z_max, section_center_2d)

    # Optional multi-level slices (20%, 30%, 50% of cranial vault height) for combined BP/SP deformities
    multi_level_slices: List[Dict] = []
    if include_aux_levels:
        for ratio_opt, lbl_opt in [
            (0.20, "Level 2 (20%)"),
            (0.30, "Level 3 (30%, стандарт)"),
            (0.50, "Level 5 (50%)"),
        ]:
            try:
                if abs(ratio_opt - sl["level_ratio"]) < 1e-4:
                    aux = sl
                else:
                    aux = _extract_slice_metrics(p1, p2, p3, z_max, ratio_opt, diagonal_angle_deg)
                multi_level_slices.append({
                    "level_ratio": ratio_opt,
                    "label": lbl_opt,
                    "z_mm": round(aux["z_slice"], 1),
                    "CI": round(aux["ci_unrounded"], 2),
                    "CVA_mm": round(aux["cva_unrounded"], 2),
                    "CVAI": round(aux["cvai_unrounded"], 2),
                    "L_mm": round(aux["L_mm"], 1),
                    "W_mm": round(aux["W_mm"], 1),
                    "perimeter_mm": round(aux["perimeter_mm"], 1),
                })
            except Exception:
                pass

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

    level_num = int(round(sl["level_ratio"] * 10))
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
            "perimeter_label": f"Периметр измерительного сечения (Level {level_num})",
            "vault_height": round(z_max, 2),
            "slice_height_z": round(z_slice, 2),
            "slice_level_ratio": round(sl["level_ratio"], 3),
            "diagonal_angle_deg": round(float(diagonal_angle_deg), 1),
        },
        "regional_asymmetry": {
            "frontal_diff_mm": round(frontal_diff_mm, 2),
            "frontal_asymmetry_pct": round(frontal_asymmetry_pct, 2),
            "occipital_diff_mm": round(occipital_diff_mm, 2),
            "occipital_asymmetry_pct": round(occipital_asymmetry_pct, 2),
            "r_d1_front_mm": round(sl["r_d1_front"], 2),
            "r_d2_front_mm": round(sl["r_d2_front"], 2),
            "r_d1_back_mm": round(sl["r_d1_back"], 2),
            "r_d2_back_mm": round(sl["r_d2_back"], 2),
            "dominant_asymmetry_zone": (
                "Затылочная область" if occipital_diff_mm >= frontal_diff_mm else "Лобно-височная область"
            ),
        },
        "quadrant_volumes": quadrant_volumes,
        "multi_level_slices": multi_level_slices,
        "classification": {
            "cvai_grade": cvai_classification,
            "borderline_alert": borderline_alert,
            "ci_interpretation": ci_note,
            "expert_ci_profile_approved": expert_ci_profile_approved,
        },
        "geometry_3d": {
            "origin": [round(float(x), 3) for x in origin],
            "axes_rotation_matrix": [[round(float(v), 5) for v in row] for row in R],
            "slice_center_world": to_world_3d(section_center_2d),
            "contour_3d": contour_3d,
            "axes_endpoints_3d": {
                "ap_front": to_world_3d(sl["pt_front"]),
                "ap_back": to_world_3d(sl["pt_back"]),
                "ml_left": to_world_3d(sl["pt_left"]),
                "ml_right": to_world_3d(sl["pt_right"]),
                "d1_front": to_world_3d(sl["pt_d1_front"]),
                "d1_back": to_world_3d(sl["pt_d1_back"]),
                "d2_front": to_world_3d(sl["pt_d2_front"]),
                "d2_back": to_world_3d(sl["pt_d2_back"]),
            },
        },
        "geometry_2d": {
            "center": [round(float(x), 3) for x in section_center_2d],
            "contour_xy": [[round(float(pt[0]), 2), round(float(pt[1]), 2)] for pt in contour_2d],
            "contour_radii_mm": [round(float(r), 2) for r in contour_radii],
            "asymmetry_profile_mm": asymmetry_profile_mm,
            "endpoints_xy": {
                "ap_front": [round(float(sl["pt_front"][0]), 2), round(float(sl["pt_front"][1]), 2)],
                "ap_back": [round(float(sl["pt_back"][0]), 2), round(float(sl["pt_back"][1]), 2)],
                "ml_left": [round(float(sl["pt_left"][0]), 2), round(float(sl["pt_left"][1]), 2)],
                "ml_right": [round(float(sl["pt_right"][0]), 2), round(float(sl["pt_right"][1]), 2)],
                "d1_front": [round(float(sl["pt_d1_front"][0]), 2), round(float(sl["pt_d1_front"][1]), 2)],
                "d1_back": [round(float(sl["pt_d1_back"][0]), 2), round(float(sl["pt_d1_back"][1]), 2)],
                "d2_front": [round(float(sl["pt_d2_front"][0]), 2), round(float(sl["pt_d2_front"][1]), 2)],
                "d2_back": [round(float(sl["pt_d2_back"][0]), 2), round(float(sl["pt_d2_back"][1]), 2)],
            },
        },
        "methodology": METHODOLOGY_REGISTRY,
    }


def compare_two_scans(
    cran_before: Dict,
    cran_after: Dict,
    scan_before_id: str,
    scan_after_id: str,
) -> Dict:
    """
    Longitudinal dynamic comparison ('Before / After' helmet therapy) of two scans
    co-registered in their canonical Se-TrR-TrL cranial frames.
    Computes exact deltas for CI, CVAI, CVA, L, W, perimeter, Q1-Q4 quadrant volume growth (cm^3),
    growth specifically inside the originally flattened quadrant, and overlaid 2D Level 3 contours.
    """
    ib = cran_before["indices"]
    ia = cran_after["indices"]
    lb = cran_before["linear_measurements_mm"]
    la = cran_after["linear_measurements_mm"]
    qb = cran_before["quadrant_volumes"]
    qa = cran_after["quadrant_volumes"]

    delta_ci = round(ia["CI_unrounded"] - ib["CI_unrounded"], 2)
    delta_cvai = round(ia["CVAI_unrounded"] - ib["CVAI_unrounded"], 2)
    delta_cva = round(ia["CVA_unrounded"] - ib["CVA_unrounded"], 2)
    delta_l = round(la["L_length_ap"] - lb["L_length_ap"], 2)
    delta_w = round(la["W_width_ml"] - lb["W_width_ml"], 2)
    delta_perim = round(la["perimeter_slice"] - lb["perimeter_slice"], 2)

    q_deltas = {
        "Q1_front_left_cm3": round(qa["Q1_front_left_cm3"] - qb["Q1_front_left_cm3"], 1),
        "Q2_front_right_cm3": round(qa["Q2_front_right_cm3"] - qb["Q2_front_right_cm3"], 1),
        "Q3_back_left_cm3": round(qa["Q3_back_left_cm3"] - qb["Q3_back_left_cm3"], 1),
        "Q4_back_right_cm3": round(qa["Q4_back_right_cm3"] - qb["Q4_back_right_cm3"], 1),
        "total_vault_volume_cm3": round(qa["total_vault_volume_cm3"] - qb["total_vault_volume_cm3"], 1),
        "delta_PSR": round(qa["posterior_symmetry_ratio"] - qb["posterior_symmetry_ratio"], 3),
        "delta_ASR": round(qa["anterior_symmetry_ratio"] - qb["anterior_symmetry_ratio"], 3),
    }

    flat_code = qb.get("flattened_quadrant_code", "Q4")
    flat_key_map = {
        "Q1": "Q1_front_left_cm3",
        "Q2": "Q2_front_right_cm3",
        "Q3": "Q3_back_left_cm3",
        "Q4": "Q4_back_right_cm3",
    }
    flat_key = flat_key_map.get(flat_code, "Q4_back_right_cm3")
    flat_growth_cm3 = q_deltas[flat_key]
    flat_before = qb[flat_key]
    flat_growth_pct = round(100.0 * flat_growth_cm3 / max(flat_before, 1e-3), 1)

    # Center both 2D contours around (0, 0) in canonical space for direct visual overlay
    cb = np.array(cran_before["geometry_2d"]["contour_xy"], dtype=np.float64) - np.array(cran_before["geometry_2d"]["center"])
    ca = np.array(cran_after["geometry_2d"]["contour_xy"], dtype=np.float64) - np.array(cran_after["geometry_2d"]["center"])
    rb = np.array(cran_before["geometry_2d"].get("contour_radii_mm", np.linalg.norm(cb, axis=1)))
    ra = np.array(cran_after["geometry_2d"].get("contour_radii_mm", np.linalg.norm(ca, axis=1)))
    radial_growth = np.round(ra - rb, 2).tolist()

    gb = cran_before["classification"]["cvai_grade"]
    ga = cran_after["classification"]["cvai_grade"]

    if delta_cvai <= -1.0:
        verdict = (
            f"Выраженная положительная динамика: индекс асимметрии CVAI снизился на {abs(delta_cvai):.2f} п.п. "
            f"(с {ib['CVAI']:.2f}% до {ia['CVAI']:.2f}%), переход категории: «{gb['code']}» → «{ga['code']}»."
        )
        trend = "IMPROVED"
    elif delta_cvai < -0.25:
        verdict = (
            f"Умеренная положительная динамика: CVAI уменьшился на {abs(delta_cvai):.2f} п.п. "
            f"(CVA изменилась на {delta_cva:+.2f} мм)."
        )
        trend = "IMPROVED"
    elif abs(delta_cvai) <= 0.25:
        verdict = (
            f"Стабильные показатели асимметрии (ΔCVAI = {delta_cvai:+.2f} п.п., ΔCI = {delta_ci:+.2f} п.п.)."
        )
        trend = "STABLE"
    else:
        verdict = (
            f"Отмечается нарастание асимметрии (ΔCVAI = {delta_cvai:+.2f} п.п., ΔCVA = {delta_cva:+.2f} мм) — требуется очная оценка врача-ортезиста."
        )
        trend = "WORSENED"

    return {
        "scan_before_id": scan_before_id,
        "scan_after_id": scan_after_id,
        "trend": trend,
        "clinical_verdict": verdict,
        "before": {
            "CI": ib["CI"],
            "CVA_mm": ib["CVA_mm"],
            "CVAI": ib["CVAI"],
            "grade_code": gb["code"],
            "grade_label": gb["label"],
            "L_mm": lb["L_length_ap"],
            "W_mm": lb["W_width_ml"],
            "perimeter_mm": lb["perimeter_slice"],
            "quadrant_volumes": qb,
        },
        "after": {
            "CI": ia["CI"],
            "CVA_mm": ia["CVA_mm"],
            "CVAI": ia["CVAI"],
            "grade_code": ga["code"],
            "grade_label": ga["label"],
            "L_mm": la["L_length_ap"],
            "W_mm": la["W_width_ml"],
            "perimeter_mm": la["perimeter_slice"],
            "quadrant_volumes": qa,
        },
        "deltas": {
            "delta_CI_pp": delta_ci,
            "delta_CVAI_pp": delta_cvai,
            "delta_CVA_mm": delta_cva,
            "delta_L_mm": delta_l,
            "delta_W_mm": delta_w,
            "delta_perimeter_mm": delta_perim,
            "quadrant_volume_deltas_cm3": q_deltas,
            "flattened_zone_code": flat_code,
            "flattened_zone_label": qb.get("flattened_quadrant_label", ""),
            "flattened_zone_growth_cm3": flat_growth_cm3,
            "flattened_zone_growth_pct": flat_growth_pct,
        },
        "overlay_2d": {
            "contour_before_xy": np.round(cb[::2], 2).tolist(),
            "contour_after_xy": np.round(ca[::2], 2).tolist(),
            "radial_growth_mm": radial_growth[::2],
        },
    }
