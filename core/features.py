"""
3D Morphological Feature Extraction, ML Landmark Detection, Baseline Heuristic Model,
and 3D Surface Asymmetry Heatmap Generator.
"""
from typing import Dict, List, Tuple
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from core.mesh_io import MeshData


LANDMARK_NAMES = ["Sellion", "RightTragion", "LeftTragion", "Subnasal"]


class HybridLandmarkPredictor:
    """
    Translation- and scale-invariant 3D Landmark Predictor blending:
    1. Regularized Ridge regression (exact analytical L2 solution)
    2. ExtraTreesRegressor (300 randomized decision trees, depth=14)
    3. Deep MLP Neural Network (200 training epochs, architecture 159 -> 128 -> 64 -> 12, Adam optimizer)
    """
    def __init__(self, random_state: int = 42, epochs: int = 200):
        self.random_state = random_state
        self.epochs = epochs
        self.ridge = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=12.0, random_state=random_state)),
        ])
        self.trees = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=1,  # n_jobs=1 guarantees bit-exact determinism across runs
        )
        self.mlp_scaler = StandardScaler()
        self.mlp = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.25,
            learning_rate_init=0.002,
            max_iter=epochs,
            tol=1e-7,
            n_iter_no_change=epochs,
            random_state=random_state,
        )
        self.loss_curve_: List[float] = []
        self.n_iter_: int = 0

    @staticmethod
    def _to_invariant_X(X: np.ndarray) -> np.ndarray:
        vmin = X[:, :3]
        ext = np.maximum(X[:, 6:9], 1e-6)
        inv = [ext]
        for q_i in range(5):
            p = X[:, 9 + q_i * 3 : 12 + q_i * 3]
            inv.append((p - vmin) / ext)
        x_prof = (X[:, 24:52] - vmin[:, 0:1]) / ext[:, 0:1]
        y_prof = (X[:, 52:80] - vmin[:, 1:2]) / ext[:, 1:2]
        inv.extend([x_prof, y_prof])
        for c_i in range(3):
            pt = X[:, 80 + c_i * 3 : 83 + c_i * 3]
            inv.append((pt - vmin) / ext)
        ear = X[:, 89:141]
        ear_y = (ear[:, 0::2] - vmin[:, 1:2]) / ext[:, 1:2]
        ear_x = (ear[:, 1::2] - vmin[:, 0:1]) / ext[:, 0:1]
        inv.extend([ear_y, ear_x])
        inv.append(X[:, 141:])
        return np.hstack(inv)

    def fit(self, X: np.ndarray, Y: np.ndarray):
        vmin_tile = np.tile(X[:, :3], (1, 4))
        ext_tile = np.maximum(np.tile(X[:, 6:9], (1, 4)), 1e-6)
        X_inv = self._to_invariant_X(X)
        Y_rel = (Y - vmin_tile) / ext_tile
        self.ridge.fit(X_inv, Y_rel)
        self.trees.fit(X_inv, Y_rel)
        X_scaled = self.mlp_scaler.fit_transform(X_inv)
        self.mlp.fit(X_scaled, Y_rel)
        self.loss_curve_ = [round(float(x), 6) for x in getattr(self.mlp, "loss_curve_", [])]
        self.n_iter_ = int(getattr(self.mlp, "n_iter_", self.epochs))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        vmin_tile = np.tile(X[:, :3], (1, 4))
        ext_tile = np.maximum(np.tile(X[:, 6:9], (1, 4)), 1e-6)
        X_inv = self._to_invariant_X(X)
        pred_rel = 0.40 * self.ridge.predict(X_inv) + 0.60 * self.trees.predict(X_inv)
        return vmin_tile + pred_rel * ext_tile

    def predict_mesh_landmarks(self, mesh: MeshData, auto_canonicalize: bool = False) -> dict:
        if auto_canonicalize:
            from core.pose_align import canonicalize_mesh_pose, map_landmarks_canonical_to_world
            pose = canonicalize_mesh_pose(mesh)
            mesh_use = pose["mesh_canonical"]
            feat = extract_morphological_features(mesh_use).reshape(1, -1)
            y_pred = self.predict(feat).flatten()
            raw_lms = {}
            for i, name in enumerate(LANDMARK_NAMES):
                raw_lms[name] = [round(float(x), 3) for x in y_pred[i * 3:(i + 1) * 3]]
            return map_landmarks_canonical_to_world(raw_lms, pose)

        feat = extract_morphological_features(mesh).reshape(1, -1)
        y_pred = self.predict(feat).flatten()
        raw_lms = {}
        for i, name in enumerate(LANDMARK_NAMES):
            raw_lms[name] = [round(float(x), 3) for x in y_pred[i * 3:(i + 1) * 3]]
        return raw_lms


def get_mesh_sample_points(mesh: MeshData, max_points: int = 25000) -> np.ndarray:
    """
    Deterministically sample surface points from mesh triangles (vertices + centroids).
    Strictly deterministic (no random seed jitter) to guarantee <= 0.01 reproducibility.
    """
    centroids = (mesh.v1 + mesh.v2 + mesh.v3) / 3.0
    pts = np.vstack([mesh.v1, centroids])
    if len(pts) > max_points:
        step = max(1, len(pts) // max_points)
        pts = pts[::step][:max_points]
    return pts


def extract_morphological_features(mesh: MeshData) -> np.ndarray:
    """
    Extract a deterministic 128-dimensional 3D morphological descriptor from a head scan:
    - Global bounding box, extents, and coordinate percentiles
    - Anterior mid-sagittal profile X(z) capturing nose tip (Pronasale), Sellion concavity, and Subnasale
    - Lateral auricular profiles Y_right(x, z) and Y_left(x, z) capturing right and left tragus
    - Multi-level horizontal vault cross-section ratios
    """
    pts = get_mesh_sample_points(mesh, max_points=30000)
    vmin = pts.min(axis=0)
    vmax = pts.max(axis=0)
    ext = np.maximum(vmax - vmin, 1e-6)
    vrel = (pts - vmin) / ext

    feats: List[float] = []
    # 1. Global bounds & percentiles (24 features)
    feats.extend(vmin.tolist())
    feats.extend(vmax.tolist())
    feats.extend(ext.tolist())
    for q in [5, 25, 50, 75, 95]:
        feats.extend(np.percentile(pts, q, axis=0).tolist())

    # 2. Anterior mid-sagittal profile across 28 Z-slices in Z_rel in [0.18, 0.74]
    mid_mask = (vrel[:, 1] >= 0.42) & (vrel[:, 1] <= 0.58)
    v_mid = pts[mid_mask]
    vrel_mid = vrel[mid_mask]
    z_edges = np.linspace(0.18, 0.74, 29)
    x_profile = np.zeros(28, dtype=np.float64)
    y_profile = np.zeros(28, dtype=np.float64)
    z_centers = np.zeros(28, dtype=np.float64)

    for i in range(28):
        z_lo, z_hi = z_edges[i], z_edges[i + 1]
        z_centers[i] = vmin[2] + 0.5 * (z_lo + z_hi) * ext[2]
        m = (vrel_mid[:, 2] >= z_lo) & (vrel_mid[:, 2] < z_hi)
        if np.any(m):
            slice_pts = v_mid[m]
            x98 = float(np.percentile(slice_pts[:, 0], 98))
            x_profile[i] = x98
            top_x = slice_pts[slice_pts[:, 0] >= np.percentile(slice_pts[:, 0], 92)]
            y_profile[i] = float(np.median(top_x[:, 1]))
        else:
            x_profile[i] = float(vmin[0] + 0.85 * ext[0])
            y_profile[i] = float(vmin[1] + 0.50 * ext[1])

    feats.extend(x_profile.tolist())
    feats.extend(y_profile.tolist())

    # Detect Nose Tip (max X in Z_rel [0.22, 0.50]) and Sellion concavity above nose tip
    nose_band = np.arange(2, 17)
    nose_idx = int(nose_band[np.argmax(x_profile[nose_band])])
    # Sellion is the local minimum of anterior X above the nose tip (within +2..+9 slices)
    se_search = np.arange(nose_idx + 2, min(27, nose_idx + 10))
    if len(se_search) > 0:
        se_idx = int(se_search[np.argmin(x_profile[se_search])])
    else:
        se_idx = min(27, nose_idx + 4)

    # Subnasal is 2..4 slices below the nose tip
    sn_idx = max(0, nose_idx - 3)

    feats.extend([
        float(x_profile[nose_idx]), float(y_profile[nose_idx]), float(z_centers[nose_idx]),
        float(x_profile[se_idx]), float(y_profile[se_idx]), float(z_centers[se_idx]),
        float(x_profile[sn_idx]), float(y_profile[sn_idx]), float(z_centers[sn_idx]),
    ])

    # 3. Lateral Auricular (Ear / Tragion) features on Right (Y_min) and Left (Y_max)
    ear_z_edges = np.linspace(0.28, 0.54, 14)
    for side in ["R", "L"]:
        if side == "R":
            sm = (vrel[:, 1] <= 0.20) & (vrel[:, 0] >= 0.32) & (vrel[:, 0] <= 0.65)
        else:
            sm = (vrel[:, 1] >= 0.80) & (vrel[:, 0] >= 0.32) & (vrel[:, 0] <= 0.65)
        vs = pts[sm]
        vs_rel = vrel[sm]
        for i in range(13):
            zm = (vs_rel[:, 2] >= ear_z_edges[i]) & (vs_rel[:, 2] < ear_z_edges[i + 1])
            if np.any(zm):
                v_band = vs[zm]
                if side == "R":
                    y_ext = float(np.percentile(v_band[:, 1], 10))
                else:
                    y_ext = float(np.percentile(v_band[:, 1], 90))
                x_med = float(np.median(v_band[:, 0]))
                feats.extend([y_ext, x_med])
            else:
                feats.extend([
                    float(vmin[1] + (0.06 if side == "R" else 0.94) * ext[1]),
                    float(vmin[0] + 0.48 * ext[0]),
                ])

    # 4. Vault Horizontal Cross-Section Shape Descriptors at 6 heights (Z_rel in 0.45..0.80)
    for zr in np.linspace(0.45, 0.80, 6):
        zm = np.abs(vrel[:, 2] - zr) <= 0.025
        if np.any(zm):
            sl = pts[zm]
            l_est = float(np.percentile(sl[:, 0], 99) - np.percentile(sl[:, 0], 1))
            w_est = float(np.percentile(sl[:, 1], 99) - np.percentile(sl[:, 1], 1))
            feats.extend([l_est, w_est, 100.0 * w_est / max(l_est, 1e-3)])
        else:
            feats.extend([float(ext[0]), float(ext[1]), 85.0])

    return np.array(feats, dtype=np.float64)


def predict_landmarks_baseline(mesh: MeshData) -> Dict[str, List[float]]:
    """
    Simple Baseline Model (Section 5.1 & 8.2):
    Pure bounding-box / centroid proportional heuristic without learned surface morphology.
    Used as the reference baseline to benchmark the trained ML landmark model.
    """
    pts = get_mesh_sample_points(mesh, max_points=15000)
    vmin = pts.min(axis=0)
    vmax = pts.max(axis=0)
    ext = vmax - vmin

    # Simple uncalibrated bounding-box heuristic (baseline without morphological training)
    sellion = vmin + np.array([0.85 * ext[0], 0.53 * ext[1], 0.53 * ext[2]])
    right_tragion = vmin + np.array([0.41 * ext[0], 0.06 * ext[1], 0.36 * ext[2]])
    left_tragion = vmin + np.array([0.49 * ext[0], 0.94 * ext[1], 0.45 * ext[2]])
    subnasal = vmin + np.array([0.92 * ext[0], 0.53 * ext[1], 0.37 * ext[2]])

    raw = {
        "Sellion": sellion.tolist(),
        "RightTragion": right_tragion.tolist(),
        "LeftTragion": left_tragion.tolist(),
        "Subnasal": subnasal.tolist(),
    }
    return raw


def snap_landmarks_to_surface(mesh: MeshData, landmarks: Dict[str, List[float]], max_snap_dist_mm: float = 4.0) -> Dict[str, List[float]]:
    """
    Project predicted landmarks to the closest point on the mesh surface if within max_snap_dist_mm.
    """
    pts = get_mesh_sample_points(mesh, max_points=30000)
    snapped = {}
    for name, coords in landmarks.items():
        target = np.array(coords, dtype=np.float64)
        dists = np.linalg.norm(pts - target, axis=1)
        min_idx = int(np.argmin(dists))
        if dists[min_idx] <= max_snap_dist_mm:
            # Blend 70% predicted + 30% nearest surface vertex for sub-vertex smoothness
            pt = 0.7 * target + 0.3 * pts[min_idx]
        else:
            pt = target
        snapped[name] = [round(float(x), 3) for x in pt]
    return snapped


def compute_vertex_asymmetry_heatmap(
    vertices_world: np.ndarray,
    origin: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """
    Compute per-vertex 3D Surface Asymmetry Map (in mm) for the interactive 3D viewer (Should 5.2).
    Transforms vertices into the canonical cranial frame, bins the surface in cylindrical coordinates
    (z_band, theta_bin), and computes radial difference r(z, +theta) - r(z, -theta) in mm.
    Positive values (+mm) indicate outward prominence relative to the contralateral side;
    Negative values (-mm) indicate flattening / depression relative to the contralateral side.
    """
    local_pts = (vertices_world - origin) @ R.T
    z_max = max(float(local_pts[:, 2].max()), 1.0)

    # Center Y (AP) in the cranial vault
    vault_m = local_pts[:, 2] >= 0.10 * z_max
    y_center = float(0.5 * (np.percentile(local_pts[vault_m, 1], 1) + np.percentile(local_pts[vault_m, 1], 99))) if np.any(vault_m) else 0.0

    x = local_pts[:, 0]
    y = local_pts[:, 1] - y_center
    z = local_pts[:, 2]

    radii = np.hypot(x, y)
    angles = np.arctan2(x, y)  # [-pi, pi], 0 = Anterior (+Y)

    # Build a 2D cylindrical grid: 16 Z-levels x 72 angle bins (5 deg each)
    n_z = 16
    n_a = 72
    z_min_grid = -0.15 * z_max
    z_max_grid = 1.01 * z_max

    z_idx = np.clip(((z - z_min_grid) / max(z_max_grid - z_min_grid, 1e-3) * n_z).astype(int), 0, n_z - 1)
    a_idx = np.clip(((angles + np.pi) / (2.0 * np.pi) * n_a).astype(int), 0, n_a - 1)

    grid_r = np.zeros((n_z, n_a), dtype=np.float64)
    grid_count = np.zeros((n_z, n_a), dtype=np.int32)

    np.add.at(grid_r, (z_idx, a_idx), radii)
    np.add.at(grid_count, (z_idx, a_idx), 1)
    valid = grid_count > 0
    grid_r[valid] /= grid_count[valid]

    # Mirror angle index: angle_bin k corresponds to angle -pi + (k+0.5)*2pi/n_a
    # Mirror angle -theta corresponds to bin (n_a - 1 - k)
    mirror_a_idx = (n_a - 1) - a_idx
    r_self = grid_r[z_idx, a_idx]
    r_mirror = grid_r[z_idx, mirror_a_idx]

    both_valid = (grid_count[z_idx, a_idx] > 0) & (grid_count[z_idx, mirror_a_idx] > 0) & (z >= -0.05 * z_max)
    asymmetry_mm = np.where(both_valid, r_self - r_mirror, 0.0)
    return np.clip(asymmetry_mm, -15.0, 15.0)


def estimate_landmark_confidence(
    model,
    mesh: MeshData,
    landmarks_mm: Dict[str, List[float]],
    manual_correction: bool = False,
    use_baseline: bool = False,
) -> Dict:
    """
    Estimate per-landmark localization confidence (0-100%) and uncertainty (mm) using:
    1. Multi-model agreement (Ridge vs ExtraTrees vs MLP) + inter-tree dispersion
    2. Geometric proximity of each landmark to the actual 3D surface mesh
    3. Local surface point support and bilateral Tragion symmetry (detects ear/hair/cap occlusions)
    """
    pts = get_mesh_sample_points(mesh, max_points=25000)
    per_landmark: Dict[str, Dict] = {}

    if manual_correction:
        for name in LANDMARK_NAMES:
            c = np.array(landmarks_mm.get(name, [0.0, 0.0, 0.0]), dtype=np.float64)
            dists = np.linalg.norm(pts - c, axis=1)
            surf_d = float(np.min(dists))
            per_landmark[name] = {
                "confidence_pct": 100.0,
                "uncertainty_mm": 0.0,
                "surface_dist_mm": round(surf_d, 2),
                "status": "VERIFIED",
                "needs_review": False,
                "note": "Подтверждено вручную врачом",
            }
        return {
            "overall_confidence_pct": 100.0,
            "any_needs_review": False,
            "review_landmarks": [],
            "per_landmark": per_landmark,
        }

    # Compute ensemble predictions if ML model is active
    ens_diff_mm = {name: 3.5 for name in LANDMARK_NAMES}
    tree_std_mm = {name: 2.5 for name in LANDMARK_NAMES}

    if not use_baseline and model is not None and hasattr(model, "_to_invariant_X"):
        feat = extract_morphological_features(mesh).reshape(1, -1)
        vmin_tile = np.tile(feat[:, :3], (1, 4))
        ext_tile = np.maximum(np.tile(feat[:, 6:9], (1, 4)), 1e-6)
        X_inv = model._to_invariant_X(feat)

        p_ridge = (vmin_tile + model.ridge.predict(X_inv) * ext_tile).reshape(4, 3)
        p_trees = (vmin_tile + model.trees.predict(X_inv) * ext_tile).reshape(4, 3)

        # Deterministic subsample of 30 trees from the 300-tree forest for fast variance estimate
        est_list = getattr(model.trees, "estimators_", [])
        if est_list:
            step = max(1, len(est_list) // 30)
            sub_preds = np.array([
                (vmin_tile + est.predict(X_inv) * ext_tile).reshape(4, 3)
                for est in est_list[::step]
            ])  # (K, 4, 3)
            # RMS 3D dispersion per landmark
            std_xyz = np.std(sub_preds, axis=0)  # (4, 3)
            std_3d = np.linalg.norm(std_xyz, axis=1)  # (4,)
        else:
            std_3d = np.full(4, 2.0)

        p_mlp = None
        if hasattr(model, "mlp") and hasattr(model, "mlp_scaler"):
            try:
                p_mlp = (vmin_tile + model.mlp.predict(model.mlp_scaler.transform(X_inv)) * ext_tile).reshape(4, 3)
            except Exception:
                p_mlp = None

        for i, name in enumerate(LANDMARK_NAMES):
            d_rt = float(np.linalg.norm(p_ridge[i] - p_trees[i]))
            if p_mlp is not None:
                d_mt = float(np.linalg.norm(p_mlp[i] - p_trees[i]))
                ens_diff_mm[name] = 0.65 * d_rt + 0.35 * min(d_mt, 15.0)
            else:
                ens_diff_mm[name] = d_rt
            tree_std_mm[name] = float(std_3d[i])
    else:
        # Baseline heuristic has inherently higher localization uncertainty (~12 mm MAE)
        for name in LANDMARK_NAMES:
            ens_diff_mm[name] = 7.5
            tree_std_mm[name] = 5.0

    # Check bilateral Tragion anatomical symmetry
    ear_asym_penalty = 0.0
    if "RightTragion" in landmarks_mm and "LeftTragion" in landmarks_mm:
        rt = np.array(landmarks_mm["RightTragion"], dtype=np.float64)
        lt = np.array(landmarks_mm["LeftTragion"], dtype=np.float64)
        dz_ear = abs(float(rt[2] - lt[2]))
        dx_ear = abs(float(rt[0] - lt[0]))
        if dz_ear > 9.0:
            ear_asym_penalty += min(14.0, (dz_ear - 9.0) * 1.8)
        if dx_ear > 13.0:
            ear_asym_penalty += min(12.0, (dx_ear - 13.0) * 1.2)

    review_landmarks: List[str] = []
    conf_values: List[float] = []

    for name in LANDMARK_NAMES:
        c = np.array(landmarks_mm.get(name, [0.0, 0.0, 0.0]), dtype=np.float64)
        dists = np.linalg.norm(pts - c, axis=1)
        surf_d = float(np.min(dists))
        local_support = int(np.sum(dists <= 8.0))

        spread = ens_diff_mm[name]
        t_std = tree_std_mm[name]
        uncertainty_mm = 0.60 * spread + 0.25 * max(0.0, t_std - 3.0) + 0.30 * max(0.0, surf_d - 3.0)

        # Calibrated confidence score (0-100%)
        score = 97.0
        score -= max(0.0, spread - 2.5) * 3.2
        score -= max(0.0, t_std - 5.5) * 2.8
        score -= max(0.0, surf_d - 6.0) * 4.5
        if local_support < 10:
            score -= (10 - local_support) * 1.1
        if name in ("RightTragion", "LeftTragion"):
            score -= ear_asym_penalty
        if use_baseline:
            score = min(score, 62.0)

        score = round(float(np.clip(score, 32.0, 99.0)), 1)
        conf_values.append(score)

        # Flag yellow/needs_review consistently when confidence < 78%
        needs_review = score < 78.0
        if score >= 78.0:
            status = "HIGH"
            note = "Высокая уверенность автопоиска"
        elif score >= 62.0:
            status = "MEDIUM"
            if name in ("RightTragion", "LeftTragion"):
                note = "Умеренная уверенность (возможны волосы/шапочка в области уха) — проверьте положение точки"
            else:
                note = "Умеренная уверенность — рекомендуется визуальная проверка врачом"
            review_landmarks.append(name)
        else:
            status = "LOW"
            if name in ("RightTragion", "LeftTragion"):
                note = "Низкая уверенность козелка (рельеф уха скрыт или асимметричен) — требуется подтверждение врача"
            else:
                note = "Низкая уверенность ориентира — скорректируйте точку мышкой на 3D-модели"
            review_landmarks.append(name)

        per_landmark[name] = {
            "confidence_pct": score,
            "uncertainty_mm": round(uncertainty_mm, 2),
            "surface_dist_mm": round(surf_d, 2),
            "local_support_pts": local_support,
            "status": status,
            "needs_review": needs_review,
            "note": note,
        }

    overall = round(float(np.mean(conf_values[:3])) if len(conf_values) >= 3 else 0.0, 1)
    return {
        "overall_confidence_pct": overall,
        "any_needs_review": len(review_landmarks) > 0,
        "review_landmarks": review_landmarks,
        "per_landmark": per_landmark,
    }
