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
        self.scaler = StandardScaler()
        self.ridge = Ridge(alpha=12.0, random_state=random_state)
        self.trees = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=1,  # n_jobs=1 guarantees bit-exact determinism across runs
        )
        self.mlp = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.05,
            learning_rate_init=0.003,
            max_iter=epochs,
            tol=1e-7,
            n_iter_no_change=epochs,  # Run full 200 epochs deterministically
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
        X_scaled = self.scaler.fit_transform(X_inv)
        Y_rel = (Y - vmin_tile) / ext_tile
        self.ridge.fit(X_scaled, Y_rel)
        self.trees.fit(X_inv, Y_rel)
        self.mlp.fit(X_scaled, Y_rel)
        self.loss_curve_ = [round(float(x), 6) for x in getattr(self.mlp, "loss_curve_", [])]
        self.n_iter_ = int(getattr(self.mlp, "n_iter_", self.epochs))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        vmin_tile = np.tile(X[:, :3], (1, 4))
        ext_tile = np.maximum(np.tile(X[:, 6:9], (1, 4)), 1e-6)
        X_inv = self._to_invariant_X(X)
        X_scaled = self.scaler.transform(X_inv)
        pred_rel = (
            0.35 * self.ridge.predict(X_scaled)
            + 0.50 * self.trees.predict(X_inv)
            + 0.15 * self.mlp.predict(X_scaled)
        )
        return vmin_tile + pred_rel * ext_tile

    def predict_mesh_landmarks(self, mesh: MeshData) -> dict:
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
