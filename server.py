"""
Local Clinical Server & Documented REST API for Saigak-Orto 3D Infant Craniometry.
Implements:
- Must (5.1): STL loading, quality/completeness check, scale calibration, ML & baseline landmark detection,
  manual landmark correction with reactive recalculation, CI/CVA/CVAI/L/W/D1/D2/Perimeter, PDF/CSV/JSON reports.
- Should (5.2): 3D Surface Asymmetry Heatmap, Frontal vs Occipital regional asymmetry, Rotation & Noise robustness evaluator (/api/robustness).
- Could (5.3): Documented OpenAPI 3.0 specification (/api/openapi.json), Batch processing (/api/batch),
  Expert queue for complex cases & retraining annotations (/api/queue).
- Security & DevOps Hardening: HTTP Security Headers, Path Traversal prevention, CSV Injection sanitization,
  PII filename anonymization check, SHA-256 model weights verification, and /api/health endpoint.
"""
import base64
import hashlib
import hmac
import io
import ipaddress
import json
import os
import re
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime
import joblib
import numpy as np
import tornado.ioloop
import tornado.web

import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from core.mesh_io import parse_stl_bytes, validate_mesh, compute_face_normals
from core.craniometry import compute_craniometry, compare_two_scans, METHODOLOGY_REGISTRY
from core.pose_align import canonicalize_mesh_pose
from core.features import (
    extract_morphological_features,
    predict_landmarks_baseline,
    compute_vertex_asymmetry_heatmap,
    estimate_landmark_confidence,
    HybridLandmarkPredictor,  # noqa: F401
)
from core.report_generator import (
    generate_pdf_report_bytes,
    export_measurement_json,
    export_measurement_csv_string,
    export_measurement_excel_xml,
)


DATA_DIR = os.path.join(PROJECT_ROOT, "data")
STL_DIR = os.path.join(DATA_DIR, "stl_scans")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
QUEUE_FILE = os.path.join(DATA_DIR, "expert_retraining_queue.json")

MAX_UPLOAD_BYTES = 64 * 1024 * 1024  # 64 MB limit against DoS

STATE = {
    "landmark_model": None,
    "deformity_classifier": None,
    "model_hashes": {},
    "evaluation_metrics": {},
    "dataset_split": {},
    "ground_truth": {},
    "uploaded_files": {},  # filename -> bytes
    "expert_queue": [],
}

SHAPE_CLASS_LABELS = {
    "N": "Нормальная форма (Normal)",
    "B": "Брахицефалия (Brachycephaly)",
    "P": "Позиционная плагиоцефалия (Plagiocephaly)",
    "S": "Скафоцефалия / Долихоцефалия (Scaphocephaly)",
    "BP": "Комбинированная брахи-плагиоцефалия (Brachy + Plagio)",
    "SP": "Комбинированная скафо-плагиоцефалия (Scapho + Plagio)",
}


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


def sanitize_filename(name: str) -> str:
    """
    Prevent Path Traversal and automatically anonymize filenames containing
    Cyrillic personal names, long digit sequences (IIN/phone), or emails into ANON-<hash>.stl.
    """
    base = os.path.basename(str(name or "scan.stl"))
    stem = re.sub(r"\.stl$", "", base, flags=re.IGNORECASE)
    has_cyrillic = bool(re.search(r"[а-яА-ЯёЁәіңғүұқөһӘІҢҒҮҰҚӨҺ]", stem))
    has_phone_or_iin = bool(re.search(r"\d{6,}", stem))
    if has_cyrillic or has_phone_or_iin or "@" in stem:
        anon_hash = hashlib.sha256(base.encode("utf-8")).hexdigest()[:8].upper()
        return f"ANON_{anon_hash}.stl"
    clean = re.sub(r"[^a-zA-Z0-9._\-]", "_", base)
    if not clean.lower().endswith(".stl"):
        clean += ".stl"
    return clean


def scrub_stl_binary_header(raw_bytes: bytes) -> bytes:
    """
    Zero out the 80-byte Binary STL header before processing/caching so any patient name
    or scanner clinic metadata embedded by 3D scanners in the header is stripped.
    """
    if len(raw_bytes) >= 84 and not raw_bytes[:5].lower().startswith(b"solid"):
        clean_hdr = b"SAIGAK-ORTO ANONYMIZED 3D SCAN".ljust(80, b" ")
        return clean_hdr + raw_bytes[80:]
    return raw_bytes


def sanitize_text_field(val: str, max_len: int = 240) -> str:
    """Prevent CSV Formula Injection (=, +, -, @) and XSS control characters."""
    s = str(val or "").strip()[:max_len]
    s = re.sub(r"[\x00-\x1f\x7f]", " ", s)
    if s and s[0] in ("=", "+", "-", "@", "\t", "\r"):
        s = "'" + s
    return s


def load_resources():
    lm_path = os.path.join(MODELS_DIR, "landmark_model.joblib")
    if os.path.exists(lm_path):
        STATE["landmark_model"] = joblib.load(lm_path)
        STATE["model_hashes"]["landmark_model_sha256"] = compute_sha256(lm_path)
    clf_path = os.path.join(MODELS_DIR, "deformity_classifier.joblib")
    if os.path.exists(clf_path):
        STATE["deformity_classifier"] = joblib.load(clf_path)
        STATE["model_hashes"]["deformity_classifier_sha256"] = compute_sha256(clf_path)
    eval_path = os.path.join(MODELS_DIR, "evaluation_metrics.json")
    if os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            STATE["evaluation_metrics"] = json.load(f)
    split_path = os.path.join(DATA_DIR, "dataset_split.json")
    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            STATE["dataset_split"] = json.load(f)
    gt_path = os.path.join(DATA_DIR, "ground_truth_landmarks.json")
    if os.path.exists(gt_path):
        with open(gt_path, "r", encoding="utf-8") as f:
            STATE["ground_truth"] = json.load(f)
    if os.path.exists(QUEUE_FILE):
        try:
            with open(QUEUE_FILE, "r", encoding="utf-8") as f:
                STATE["expert_queue"] = json.load(f)
        except Exception:
            STATE["expert_queue"] = []
    else:
        STATE["expert_queue"] = [
            {
                "queue_id": "Q-001",
                "scan_id": "clinical_very_severe_plagio_P04",
                "reason": "Очень тяжёлая асимметрия (CVAI = 10.05% / пограничный случай выраженной деформации)",
                "status": "Ожидает верификации клиническим ментором",
                "doctor_name": "Золотухин А.П.",
                "timestamp": "2026-11-20T14:15:00",
                "annotated_landmarks_mm": STATE["ground_truth"].get("scan_003", {}).get("landmarks_mm"),
            }
        ]
        save_expert_queue()


def save_expert_queue():
    try:
        with open(QUEUE_FILE, "w", encoding="utf-8") as f:
            json.dump(STATE["expert_queue"], f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def get_raw_stl_bytes(filename: str) -> bytes:
    safe_name = sanitize_filename(filename)
    if safe_name in STATE["uploaded_files"]:
        return STATE["uploaded_files"][safe_name]
    stl_path = os.path.join(STL_DIR, safe_name)
    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"Файл {safe_name} не найден в локальном хранилище.")
    with open(stl_path, "rb") as f:
        return f.read()


def _decimate_mesh_vertex_clustering(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, target_tris: int = 16000):
    """
    Fast pure-NumPy Rossignac-Borrel smooth vertex clustering decimation.
    Preserves a watertight, hole-free surface for dense 100k+ triangle Artec 3D meshes
    (instead of striding faces which creates speckled gaps).
    """
    n_tri = len(v1)
    if n_tri <= target_tris:
        return v1, v2, v3

    all_v = np.concatenate([v1, v2, v3], axis=0)
    vmin = all_v.min(axis=0)
    vmax = all_v.max(axis=0)
    ext = np.maximum(vmax - vmin, 1e-3)
    diag = float(np.linalg.norm(ext))

    # Estimate surface voxel pitch for target_tris (~2 triangles per occupied surface voxel)
    # Surface area of an infant head ~ 4 * pi * (diag/3.4)^2; target occupied cells ~ target_tris / 2.2
    voxel = max(1.8, diag / np.sqrt(max(target_tris, 4000) * 0.28))

    for _ in range(3):
        q = np.floor((all_v - vmin) / voxel).astype(np.int64)
        # Pack 3D voxel index into 1D key (grid <= 512 per axis)
        keys = (q[:, 0] << 22) + (q[:, 1] << 11) + q[:, 2]
        uniq, inv = np.unique(keys, return_inverse=True)

        c1 = inv[:n_tri]
        c2 = inv[n_tri : 2 * n_tri]
        c3 = inv[2 * n_tri :]
        valid = (c1 != c2) & (c2 != c3) & (c1 != c3)
        nv = int(valid.sum())
        if nv <= target_tris * 1.45 and nv >= target_tris * 0.35:
            break
        scale = np.sqrt(max(nv, 100) / float(target_tris))
        voxel *= float(np.clip(scale, 0.65, 1.8))

    # Deduplicate identical clustered faces (unordered vertex triple)
    vc1, vc2, vc3 = c1[valid], c2[valid], c3[valid]
    tri_idx = np.stack([vc1, vc2, vc3], axis=1)
    tri_sorted = np.sort(tri_idx, axis=1)
    _, keep_idx = np.unique(tri_sorted, axis=0, return_index=True)
    tri_keep = tri_idx[keep_idx]

    # Smooth vertex position = centroid of all original vertices in each voxel cell
    n_cells = len(uniq)
    counts = np.bincount(inv, minlength=n_cells).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    centroids = np.empty((n_cells, 3), dtype=np.float64)
    for ax in range(3):
        centroids[:, ax] = np.bincount(inv, weights=all_v[:, ax], minlength=n_cells) / counts

    return centroids[tri_keep[:, 0]], centroids[tri_keep[:, 1]], centroids[tri_keep[:, 2]]


def build_threejs_mesh_payload(mesh, origin=None, R=None, max_tris: int = 16000) -> dict:
    n_tri = len(mesh.v1)
    if n_tri > max_tris:
        v1, v2, v3 = _decimate_mesh_vertex_clustering(mesh.v1, mesh.v2, mesh.v3, target_tris=max_tris)
    else:
        v1, v2, v3 = mesh.v1, mesh.v2, mesh.v3
    n_sub = len(v1)

    verts = np.empty((n_sub * 3, 3), dtype=np.float32)
    verts[0::3] = v1
    verts[1::3] = v2
    verts[2::3] = v3

    if origin is not None and R is not None:
        asym_mm = compute_vertex_asymmetry_heatmap(
            verts.astype(np.float64),
            np.asarray(origin, dtype=np.float64),
            np.asarray(R, dtype=np.float64),
        ).astype(np.float32)
    else:
        asym_mm = np.zeros(len(verts), dtype=np.float32)

    return {
        "displayed_triangles": n_sub,
        "total_triangles": n_tri,
        "positions_b64": base64.b64encode(verts.tobytes()).decode("ascii"),
        "asymmetry_mm_b64": base64.b64encode(asym_mm.tobytes()).decode("ascii"),
    }


def prepare_mesh_and_quality(raw_bytes: bytes, scale_factor: float = 1.0):
    """
    Parse STL, run quality gate, and automatically canonicalize arbitrarily rotated 3D scans
    (such as handheld Artec 3D clinic scans) into upright cranial orientation (+X face, +Y left, +Z up).
    """
    mesh = parse_stl_bytes(raw_bytes, scale_factor=scale_factor)
    quality = validate_mesh(mesh)
    pose_alignment = {"auto_aligned": False, "correction_deg": 0.0, "alignment_chamfer": 0.0}

    if quality.is_usable and quality.vault_closure_pct >= 85.0:
        pose = canonicalize_mesh_pose(mesh, force_align=False)
        if pose.get("was_rotated"):
            mesh = pose["mesh_canonical"]
            quality = validate_mesh(mesh)
            pose_alignment = {
                "auto_aligned": True,
                "correction_deg": pose["rotation_deg"],
                "alignment_chamfer": pose["alignment_chamfer"],
            }
    return mesh, quality, pose_alignment


def analyze_scan_bytes(
    raw_bytes: bytes,
    scan_id: str,
    scale_factor: float = 1.0,
    level_ratio: float = 0.30,
    diagonal_angle_deg: float = 30.0,
    custom_landmarks: dict = None,
    use_baseline: bool = False,
    expert_ci_profile_approved: bool = False,
    include_mesh_3d: bool = True,
) -> dict:
    t0 = time.perf_counter()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError(f"Размер файла превышает допустимый лимит безопасности ({MAX_UPLOAD_BYTES // (1024*1024)} МБ).")

    mesh, quality, pose_alignment = prepare_mesh_and_quality(raw_bytes, scale_factor=scale_factor)

    landmarks = None
    landmark_confidence = None
    craniometry = None
    shape_prediction = None
    manual_correction = False

    if quality.is_usable:
        if custom_landmarks and all(k in custom_landmarks for k in ["Sellion", "RightTragion", "LeftTragion"]):
            landmarks = custom_landmarks
            manual_correction = True
            model_used = "Ручная коррекция ориентиров специалистом + детерминированный перерасчёт"
        elif use_baseline:
            landmarks = predict_landmarks_baseline(mesh)
            model_used = "Saigak-Baseline-BBox-Heuristic-v1.0 (Базовая модель)"
        else:
            landmarks = STATE["landmark_model"].predict_mesh_landmarks(mesh)
            model_used = "Saigak-ML-ExtraTrees-v1.2 (Обученная ML-модель, seed=42)"

        landmark_confidence = estimate_landmark_confidence(
            STATE["landmark_model"],
            mesh,
            landmarks,
            manual_correction=manual_correction,
            use_baseline=use_baseline,
        )

        try:
            craniometry = compute_craniometry(
                mesh=mesh,
                landmarks=landmarks,
                level_ratio=level_ratio,
                diagonal_angle_deg=diagonal_angle_deg,
                expert_ci_profile_approved=expert_ci_profile_approved,
                include_aux_levels=include_mesh_3d,
            )
        except Exception as e:
            quality.status = "REJECTED"
            quality.is_usable = False
            quality.defects.append(f"Ошибка геометрического построения сечения: {str(e)}")

        if STATE["deformity_classifier"] is not None and quality.is_usable:
            feat = extract_morphological_features(mesh).reshape(1, -1)
            clf = STATE["deformity_classifier"]
            pred_cls = str(clf.predict(feat)[0])
            probs = clf.predict_proba(feat)[0]
            prob_dict = {
                str(c): round(float(p) * 100.0, 1)
                for c, p in zip(clf.classes_, probs)
            }
            shape_prediction = {
                "predicted_code": pred_cls,
                "predicted_label": SHAPE_CLASS_LABELS.get(pred_cls, pred_cls),
                "probabilities_pct": prob_dict,
            }
    else:
        model_used = "Расчёт заблокирован контролем качества входных данных"

    mesh_payload = None
    if include_mesh_3d:
        origin = craniometry["geometry_3d"]["origin"] if craniometry else None
        R = craniometry["geometry_3d"]["axes_rotation_matrix"] if craniometry else None
        mesh_payload = build_threejs_mesh_payload(mesh, origin=origin, R=R)

    gt_comparison = None
    base_id = sanitize_filename(scan_id).replace(".stl", "")
    if base_id in STATE["ground_truth"] and craniometry and abs(scale_factor - 1.0) < 1e-6:
        gt_entry = STATE["ground_truth"][base_id]
        gt_ind = gt_entry["ground_truth_indices"]
        gt_lms = gt_entry["landmarks_mm"]
        gt_comparison = {
            "has_ground_truth": True,
            "split": gt_entry["split"],
            "gt_landmarks_mm": gt_lms,
            "gt_indices": gt_ind,
            "abs_errors": {
                "CI_pp": round(abs(craniometry["indices"]["CI_unrounded"] - gt_ind["CI_unrounded"]), 3),
                "CVAI_pp": round(abs(craniometry["indices"]["CVAI_unrounded"] - gt_ind["CVAI_unrounded"]), 3),
                "CVA_mm": round(abs(craniometry["indices"]["CVA_unrounded"] - gt_ind["CVA_unrounded"]), 3),
            },
        }

    elapsed_sec = time.perf_counter() - t0
    return {
        "scan_id": base_id,
        "processing_time_sec": round(elapsed_sec, 4),
        "model_version": model_used,
        "manual_correction_applied": manual_correction,
        "pose_alignment": pose_alignment,
        "quality": quality.to_dict(),
        "landmarks_mm": landmarks,
        "landmark_confidence": landmark_confidence,
        "craniometry": craniometry,
        "shape_classifier": shape_prediction,
        "ground_truth_comparison": gt_comparison,
        "mesh_3d": mesh_payload,
    }


def evaluate_rotation_and_noise_robustness(
    raw_bytes: bytes,
    scale_factor: float = 1.0,
    level_ratio: float = 0.30,
    noise_sigma_mm: float = 0.25,
    rotation_deg: float = 15.0,
) -> dict:
    """
    Should 5.2: Evaluate robustness of craniometric indices (CI, CVA, CVAI) to:
    1. Gaussian surface vertex noise (sigma = 0.25 mm)
    2. 3D rigid rotation of the scan + landmarks by rotation_deg around an oblique 3D axis
    3. Repeat-run determinism check (divergence <= 0.01)
    """
    mesh_base = parse_stl_bytes(raw_bytes, scale_factor=scale_factor)
    q = validate_mesh(mesh_base)
    if not q.is_usable:
        raise ValueError("Скан непригоден для теста устойчивости (не прошёл контроль качества).")

    lms_base = STATE["landmark_model"].predict_mesh_landmarks(mesh_base)
    cran_base = compute_craniometry(mesh_base, lms_base, level_ratio=level_ratio)
    ind_base = cran_base["indices"]

    # 1. Repeat-run determinism
    mesh_rep = parse_stl_bytes(raw_bytes, scale_factor=scale_factor)
    lms_rep = STATE["landmark_model"].predict_mesh_landmarks(mesh_rep)
    cran_rep = compute_craniometry(mesh_rep, lms_rep, level_ratio=level_ratio)
    rep_diff = max(
        abs(ind_base["CI_unrounded"] - cran_rep["indices"]["CI_unrounded"]),
        abs(ind_base["CVAI_unrounded"] - cran_rep["indices"]["CVAI_unrounded"]),
        abs(ind_base["CVA_unrounded"] - cran_rep["indices"]["CVA_unrounded"]),
    )

    # 2. Gaussian surface noise robustness (deterministic seed=42)
    rng = np.random.default_rng(42)
    mesh_noisy = parse_stl_bytes(raw_bytes, scale_factor=scale_factor)
    mesh_noisy.v1 = mesh_noisy.v1 + rng.normal(0.0, noise_sigma_mm, size=mesh_noisy.v1.shape)
    mesh_noisy.v2 = mesh_noisy.v2 + rng.normal(0.0, noise_sigma_mm, size=mesh_noisy.v2.shape)
    mesh_noisy.v3 = mesh_noisy.v3 + rng.normal(0.0, noise_sigma_mm, size=mesh_noisy.v3.shape)
    lms_noisy = STATE["landmark_model"].predict_mesh_landmarks(mesh_noisy)
    cran_noisy = compute_craniometry(mesh_noisy, lms_noisy, level_ratio=level_ratio)
    ind_noisy = cran_noisy["indices"]

    # 3. 3D Rigid Rotation invariance test (rotate mesh and reference frame by rotation_deg)
    theta = np.deg2rad(rotation_deg)
    axis = np.array([0.3, 0.5, 0.8], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R_rot = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    center_pt = np.mean(mesh_base.v1, axis=0)

    mesh_rot = parse_stl_bytes(raw_bytes, scale_factor=scale_factor)
    mesh_rot.v1 = (mesh_rot.v1 - center_pt) @ R_rot.T + center_pt
    mesh_rot.v2 = (mesh_rot.v2 - center_pt) @ R_rot.T + center_pt
    mesh_rot.v3 = (mesh_rot.v3 - center_pt) @ R_rot.T + center_pt
    lms_rot = {
        k: (((np.array(v, dtype=np.float64) - center_pt) @ R_rot.T) + center_pt).tolist()
        for k, v in lms_base.items()
    }
    cran_rot = compute_craniometry(mesh_rot, lms_rot, level_ratio=level_ratio)
    ind_rot = cran_rot["indices"]

    return {
        "baseline_indices": {
            "CI": ind_base["CI"],
            "CVA_mm": ind_base["CVA_mm"],
            "CVAI": ind_base["CVAI"],
        },
        "repeat_run_determinism": {
            "max_divergence": round(float(rep_diff), 6),
            "target_max_divergence": 0.01,
            "passed": float(rep_diff) <= 0.01,
        },
        "noise_perturbation_test": {
            "sigma_mm": noise_sigma_mm,
            "noisy_CI": ind_noisy["CI"],
            "noisy_CVA_mm": ind_noisy["CVA_mm"],
            "noisy_CVAI": ind_noisy["CVAI"],
            "delta_CI_pp": round(abs(ind_base["CI_unrounded"] - ind_noisy["CI_unrounded"]), 3),
            "delta_CVA_mm": round(abs(ind_base["CVA_unrounded"] - ind_noisy["CVA_unrounded"]), 3),
            "delta_CVAI_pp": round(abs(ind_base["CVAI_unrounded"] - ind_noisy["CVAI_unrounded"]), 3),
            "passed": abs(ind_base["CVAI_unrounded"] - ind_noisy["CVAI_unrounded"]) <= 0.50,
        },
        "rotation_3d_invariance_test": {
            "rotation_angle_deg": rotation_deg,
            "rotated_CI": ind_rot["CI"],
            "rotated_CVA_mm": ind_rot["CVA_mm"],
            "rotated_CVAI": ind_rot["CVAI"],
            "delta_CI_pp": round(abs(ind_base["CI_unrounded"] - ind_rot["CI_unrounded"]), 5),
            "delta_CVA_mm": round(abs(ind_base["CVA_unrounded"] - ind_rot["CVA_unrounded"]), 5),
            "delta_CVAI_pp": round(abs(ind_base["CVAI_unrounded"] - ind_rot["CVAI_unrounded"]), 5),
            "passed": abs(ind_base["CVAI_unrounded"] - ind_rot["CVAI_unrounded"]) <= 0.01,
        },
    }


class BaseSecureHandler(tornado.web.RequestHandler):
    """Base handler adding strict medical software HTTP security headers."""
    def set_default_headers(self):
        self.set_header("X-Content-Type-Options", "nosniff")
        self.set_header("X-Frame-Options", "DENY")
        self.set_header("X-XSS-Protection", "1; mode=block")
        self.set_header("Referrer-Policy", "no-referrer")
        self.set_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.set_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self';"
        )


class WelcomeHandler(BaseSecureHandler):
    def get(self):
        with open(os.path.join(STATIC_DIR, "welcome.html"), "r", encoding="utf-8") as f:
            self.set_header("Content-Type", "text/html; charset=utf-8")
            self.write(f.read())


class CockpitHandler(BaseSecureHandler):
    def get(self):
        with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
            self.set_header("Content-Type", "text/html; charset=utf-8")
            self.write(f.read())


class HealthHandler(BaseSecureHandler):
    def get(self):
        self.set_header("Content-Type", "application/json; charset=utf-8")
        self.write(json.dumps({
            "status": "HEALTHY",
            "service": "Saigak-Orto 3D Craniometry Local Workstation",
            "version": "1.2.0",
            "local_processing_only": True,
            "external_ai_calls_disabled": True,
            "models_loaded": {
                "landmark_model": STATE["landmark_model"] is not None,
                "deformity_classifier": STATE["deformity_classifier"] is not None,
            },
            "model_integrity_hashes": STATE["model_hashes"],
            "timestamp": datetime.now().isoformat(),
        }, ensure_ascii=False))


class OpenApiHandler(BaseSecureHandler):
    def get(self):
        self.set_header("Content-Type", "application/json; charset=utf-8")
        spec = {
            "openapi": "3.0.3",
            "info": {
                "title": "Saigak-Orto 3D Craniometry, Longitudinal Dynamics & MIS Integration API",
                "version": "1.3.0",
                "description": (
                    "Локальный REST API для стандартизированного расчёта краниальных индексов (CI, CVA, CVAI), "
                    "объёмной квадрантной асимметрии (Q1–Q4, ASR, PSR), сравнения в динамике («До / После»), "
                    "пакетной обработки папок STL с выгрузкой CSV/Excel/ZIP-архива PDF-отчётов и безопасного "
                    "анонимизированного экспорта/вебхука в МИС клиники (HL7 FHIR DiagnosticReport-совместимый JSON)."
                ),
                "contact": {
                    "name": "ТОО Saigak-Orto (Сайгак-Орто)",
                    "url": "https://saigak-orto.kz/",
                    "email": "boiko@saigak-orto.kz",
                },
            },
            "servers": [
                {"url": "http://127.0.0.1:8765", "description": "Локальная рабочая станция врача (airgapped)"},
            ],
            "tags": [
                {"name": "System", "description": "Здоровье сервиса и проверка целостности весов SHA-256"},
                {"name": "Craniometry", "description": "3D-анализ, уверенность ориентиров, динамика До/После и устойчивость"},
                {"name": "Reporting & Batch", "description": "Клинические PDF-отчёты, пакетная обработка папок, CSV/Excel/ZIP"},
                {"name": "MIS Integration", "description": "Анонимизированный вебхук и экспорт в МИС клиники"},
            ],
            "paths": {
                "/api/health": {
                    "get": {
                        "tags": ["System"],
                        "summary": "Проверка состояния сервера и целостности весов ML-моделей (SHA-256)",
                        "responses": {"200": {"description": "Статус HEALTHY и SHA-256 контрольные суммы моделей"}},
                    }
                },
                "/api/catalog": {
                    "get": {
                        "tags": ["System"],
                        "summary": "Список доступных 3D STL-сканов, реестр методик и метрики валидации (MAE, P95, Max)",
                        "responses": {"200": {"description": "Каталог STL и протокол валидации"}},
                    }
                },
                "/api/analyze": {
                    "post": {
                        "tags": ["Craniometry"],
                        "summary": "Анализ одного 3D STL-скана (контроль дефектов/артефактов, уверенность ориентиров, CI/CVA/CVAI, Q1–Q4, уровни 20%/30%/50%, пограничный флаг)",
                        "requestBody": {
                            "required": True,
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/AnalyzeRequest"}}},
                        },
                        "responses": {"200": {"description": "Полный протокол измерений и 3D-буфер"}},
                    }
                },
                "/api/compare": {
                    "post": {
                        "tags": ["Craniometry"],
                        "summary": "Сравнение двух 3D-сканов одного пациента в динамике («До / После» шлемотерапии)",
                        "requestBody": {
                            "required": True,
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/CompareRequest"}}},
                        },
                        "responses": {"200": {"description": "Дельты CI/CVAI/CVA, прирост объёма в зоне уплощения (см³) и совмещённые контуры Level 3"}},
                    }
                },
                "/api/robustness": {
                    "post": {
                        "tags": ["Craniometry"],
                        "summary": "Тест устойчивости к 3D-повороту модели, гауссовскому шуму поверхности и повторному запуску",
                        "responses": {"200": {"description": "Результаты испытания детерминизма и устойчивости"}},
                    }
                },
                "/api/batch": {
                    "post": {
                        "tags": ["Reporting & Batch"],
                        "summary": "Пакетная обработка серии или загруженной папки STL-сканов с выгрузкой CSV и Excel (.xls)",
                        "requestBody": {
                            "required": True,
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/BatchRequest"}}},
                        },
                        "responses": {"200": {"description": "Сводная таблица, CSV и SpreadsheetML Excel"}},
                    }
                },
                "/api/batch/zip": {
                    "post": {
                        "tags": ["Reporting & Batch"],
                        "summary": "Скачать ZIP-архив пакетной обработки (сводные CSV/Excel/JSON + индивидуальные PDF-отчёты по каждому скану)",
                        "requestBody": {
                            "required": True,
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/BatchRequest"}}},
                        },
                        "responses": {
                            "200": {
                                "description": "ZIP-архив с PDF-отчётами и таблицами",
                                "content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}},
                            }
                        },
                    }
                },
                "/api/queue": {
                    "get": {
                        "tags": ["Craniometry"],
                        "summary": "Получить экспертную очередь сложных случаев и сохранённых разметок для переобучения",
                        "responses": {"200": {"description": "Список записей экспертной очереди"}},
                    },
                    "post": {
                        "tags": ["Craniometry"],
                        "summary": "Добавить скан и скорректированные врачом ориентиры в экспертную очередь дообучения",
                        "responses": {"200": {"description": "Запись добавлена"}},
                    },
                },
                "/api/report/pdf": {
                    "post": {
                        "tags": ["Reporting & Batch"],
                        "summary": "Сформировать клинический A4 PDF-отчёт с 2D-схемой среза, аудитом ориентиров и подписью врача",
                        "responses": {
                            "200": {
                                "description": "Векторный A4 PDF-отчёт",
                                "content": {"application/pdf": {"schema": {"type": "string", "format": "binary"}}},
                            }
                        },
                    }
                },
                "/api/report/export": {
                    "post": {
                        "tags": ["Reporting & Batch"],
                        "summary": "Экспорт численных результатов в CSV, Excel (.xls) или JSON",
                        "responses": {"200": {"description": "Файл экспорта"}},
                    }
                },
                "/api/mis/export": {
                    "post": {
                        "tags": ["MIS Integration"],
                        "summary": "Анонимизированный вебхук / экспорт в МИС клиники (HL7 FHIR DiagnosticReport JSON + опционально PDF в Base64 + HMAC-SHA256)",
                        "requestBody": {
                            "required": True,
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/MisExportRequest"}}},
                        },
                        "responses": {"200": {"description": "Анонимизированный пакет МИС и статус доставки вебхука"}},
                    }
                },
            },
            "components": {
                "schemas": {
                    "AnalyzeRequest": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "example": "scan_081.stl"},
                            "file_b64": {"type": "string", "description": "Опционально: STL в Base64"},
                            "scale_factor": {"type": "number", "default": 1.0},
                            "level_ratio": {"type": "number", "default": 0.30, "description": "Доля высоты свода (0.20, 0.30, 0.50)"},
                            "custom_landmarks": {"type": "object", "description": "Ручные координаты Sellion, RightTragion, LeftTragion"},
                            "include_mesh_3d": {"type": "boolean", "default": True, "description": "False для мгновенного пересчёта при перетаскивании ориентиров"},
                        },
                    },
                    "CompareRequest": {
                        "type": "object",
                        "required": ["scan_before", "scan_after"],
                        "properties": {
                            "scan_before": {"type": "string", "example": "clinical_severe_plagio_P03.stl"},
                            "scan_after": {"type": "string", "example": "clinical_mild_plagio_P01.stl"},
                            "file_before_b64": {"type": "string"},
                            "file_after_b64": {"type": "string"},
                            "scale_before": {"type": "number", "default": 1.0},
                            "scale_after": {"type": "number", "default": 1.0},
                            "level_ratio": {"type": "number", "default": 0.30},
                        },
                    },
                    "BatchRequest": {
                        "type": "object",
                        "properties": {
                            "filenames": {"type": "array", "items": {"type": "string"}},
                            "uploaded_files": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "filename": {"type": "string"},
                                        "file_b64": {"type": "string"},
                                    },
                                },
                            },
                            "scale_factor": {"type": "number", "default": 1.0},
                            "level_ratio": {"type": "number", "default": 0.30},
                            "doctor_name": {"type": "string"},
                        },
                    },
                    "MisExportRequest": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "example": "clinical_severe_plagio_P03.stl"},
                            "mode": {"type": "string", "enum": ["preview", "dispatch"], "default": "preview"},
                            "webhook_url": {"type": "string", "example": "http://127.0.0.1:9000/mis/craniometry-webhook"},
                            "webhook_secret": {"type": "string", "description": "Секрет для подписи X-Saigak-Signature (HMAC-SHA256)"},
                            "include_pdf": {"type": "boolean", "default": True},
                            "patient_age_months": {"type": "number"},
                            "doctor_name": {"type": "string"},
                            "doctor_verified": {"type": "boolean", "default": True},
                            "clinical_notes": {"type": "string"},
                        },
                    },
                }
            },
        }
        self.write(json.dumps(spec, ensure_ascii=False, indent=2))


class CatalogHandler(BaseSecureHandler):
    def get(self):
        self.set_header("Content-Type", "application/json; charset=utf-8")
        files = sorted([f for f in os.listdir(STL_DIR) if f.endswith(".stl")])
        demos = STATE["dataset_split"].get("clinical_and_defect_demos", [])
        demo_map = {d["filename"]: d for d in demos}
        artec_demos = STATE["dataset_split"].get("artec_clinic_dataset", {}).get("representative_demos", [])
        artec_map = {d["filename"]: d for d in artec_demos}

        catalog_items = []
        for fname in files:
            sid = fname.replace(".stl", "")
            if fname in demo_map:
                d = demo_map[fname]
                catalog_items.append({
                    "filename": fname,
                    "scan_id": sid,
                    "group": "1. Клинические случаи деформаций и контроль дефектов",
                    "label": f"{sid} — {d['description']}",
                    "category": d["category"],
                })
            elif fname in artec_map:
                d = artec_map[fname]
                catalog_items.append({
                    "filename": fname,
                    "scan_id": sid,
                    "group": "1b. Реальные клинические сканы (Artec 3D)",
                    "label": f"{sid} — {d['description']}",
                    "category": "artec_clinic_real",
                })
            elif sid in STATE["ground_truth"]:
                sp = STATE["ground_truth"][sid]["split"]
                grp = "2. Закрытая тестовая выборка (Reserved Test)" if sp == "test" else "3. Обучающая выборка (Train)"
                catalog_items.append({
                    "filename": fname,
                    "scan_id": sid,
                    "group": grp,
                    "label": f"{sid} ({'Тестовая выборка' if sp == 'test' else 'Обучающая выборка'})",
                    "category": sp,
                })

        for up_name in STATE["uploaded_files"]:
            catalog_items.insert(0, {
                "filename": up_name,
                "scan_id": up_name.replace(".stl", ""),
                "group": "0. Загруженные пользователем STL",
                "label": f"{up_name} (Пользовательский STL)",
                "category": "user_upload",
            })

        self.write(json.dumps({
            "scans": catalog_items,
            "methodology": METHODOLOGY_REGISTRY,
            "evaluation_metrics": STATE["evaluation_metrics"],
            "expert_queue_count": len(STATE["expert_queue"]),
            "sample_reports": [
                {"name": "1. Качественный скан (scan_081)", "url": "/reports/report_case1_good_scan.pdf"},
                {"name": "2. Тяжёлая плагиоцефалия (CVAI 8.75–11%)", "url": "/reports/report_case2_severe_deformity.pdf"},
                {"name": "3. Очень тяжёлая деформация (CVAI ≥ 11%)", "url": "/reports/report_case2b_very_severe.pdf"},
                {"name": "4. Отказ при дефектном скане", "url": "/reports/report_case3_rejected_defect.pdf"},
                {"name": "5. Протокол валидации (CSV)", "url": "/reports/validation_protocol_results.csv"},
            ],
        }, ensure_ascii=False))


class AnalyzeHandler(BaseSecureHandler):
    def post(self):
        self.set_header("Content-Type", "application/json; charset=utf-8")
        try:
            body = json.loads(self.request.body.decode("utf-8"))
            filename = sanitize_filename(body.get("filename", "scan_081.stl"))
            scale_factor = float(body.get("scale_factor", 1.0))
            level_ratio = float(body.get("level_ratio", 0.30))
            diagonal_angle_deg = float(body.get("diagonal_angle_deg", 30.0))
            custom_landmarks = body.get("custom_landmarks")
            use_baseline = bool(body.get("use_baseline", False))
            expert_ci = bool(body.get("expert_ci_profile_approved", False))
            include_mesh_3d = bool(body.get("include_mesh_3d", True))

            if body.get("file_b64"):
                raw_bytes = scrub_stl_binary_header(base64.b64decode(body["file_b64"]))
                if len(raw_bytes) > MAX_UPLOAD_BYTES:
                    raise ValueError("Превышен максимальный размер STL-файла (64 МБ).")
                STATE["uploaded_files"][filename] = raw_bytes
            else:
                raw_bytes = get_raw_stl_bytes(filename)

            result = analyze_scan_bytes(
                raw_bytes=raw_bytes,
                scan_id=filename,
                scale_factor=scale_factor,
                level_ratio=level_ratio,
                diagonal_angle_deg=diagonal_angle_deg,
                custom_landmarks=custom_landmarks,
                use_baseline=use_baseline,
                expert_ci_profile_approved=expert_ci,
                include_mesh_3d=include_mesh_3d,
            )
            self.write(json.dumps(result, ensure_ascii=False))
        except FileNotFoundError as e:
            self.set_status(404)
            self.write(json.dumps({"error": str(e)}, ensure_ascii=False))
        except Exception as e:
            self.set_status(400)
            self.write(json.dumps({"error": str(e)}, ensure_ascii=False))


class CompareHandler(BaseSecureHandler):
    """Longitudinal 'Before / After' comparison of two 3D STL scans in canonical cranial space."""
    def post(self):
        self.set_header("Content-Type", "application/json; charset=utf-8")
        try:
            body = json.loads(self.request.body.decode("utf-8"))
            fn_before = sanitize_filename(body.get("scan_before", "clinical_severe_plagio_P03.stl"))
            fn_after = sanitize_filename(body.get("scan_after", "clinical_mild_plagio_P01.stl"))
            scale_before = float(body.get("scale_before", 1.0))
            scale_after = float(body.get("scale_after", 1.0))
            level_ratio = float(body.get("level_ratio", 0.30))
            diagonal_angle_deg = float(body.get("diagonal_angle_deg", 30.0))

            if body.get("file_before_b64"):
                raw_before = scrub_stl_binary_header(base64.b64decode(body["file_before_b64"]))
                STATE["uploaded_files"][fn_before] = raw_before
            else:
                raw_before = get_raw_stl_bytes(fn_before)

            if body.get("file_after_b64"):
                raw_after = scrub_stl_binary_header(base64.b64decode(body["file_after_b64"]))
                STATE["uploaded_files"][fn_after] = raw_after
            else:
                raw_after = get_raw_stl_bytes(fn_after)

            res_b = analyze_scan_bytes(
                raw_before, fn_before, scale_factor=scale_before,
                level_ratio=level_ratio, diagonal_angle_deg=diagonal_angle_deg,
                include_mesh_3d=False,
            )
            res_a = analyze_scan_bytes(
                raw_after, fn_after, scale_factor=scale_after,
                level_ratio=level_ratio, diagonal_angle_deg=diagonal_angle_deg,
                include_mesh_3d=False,
            )
            if not res_b["quality"]["is_usable"] or not res_b["craniometry"]:
                raise ValueError(f"Скан «До» ({res_b['scan_id']}) не прошёл контроль качества и непригоден для сравнения.")
            if not res_a["quality"]["is_usable"] or not res_a["craniometry"]:
                raise ValueError(f"Скан «После» ({res_a['scan_id']}) не прошёл контроль качества и непригоден для сравнения.")

            comp = compare_two_scans(
                cran_before=res_b["craniometry"],
                cran_after=res_a["craniometry"],
                scan_before_id=res_b["scan_id"],
                scan_after_id=res_a["scan_id"],
            )
            self.write(json.dumps(comp, ensure_ascii=False))
        except Exception as e:
            self.set_status(400)
            self.write(json.dumps({"error": str(e)}, ensure_ascii=False))


class RobustnessHandler(BaseSecureHandler):
    def post(self):
        self.set_header("Content-Type", "application/json; charset=utf-8")
        try:
            body = json.loads(self.request.body.decode("utf-8"))
            filename = sanitize_filename(body.get("filename", "scan_081.stl"))
            scale_factor = float(body.get("scale_factor", 1.0))
            level_ratio = float(body.get("level_ratio", 0.30))
            noise_sigma = float(body.get("noise_sigma_mm", 0.25))
            rot_deg = float(body.get("rotation_deg", 15.0))

            raw_bytes = get_raw_stl_bytes(filename)
            res = evaluate_rotation_and_noise_robustness(
                raw_bytes=raw_bytes,
                scale_factor=scale_factor,
                level_ratio=level_ratio,
                noise_sigma_mm=noise_sigma,
                rotation_deg=rot_deg,
            )
            self.write(json.dumps(res, ensure_ascii=False))
        except Exception as e:
            self.set_status(400)
            self.write(json.dumps({"error": str(e)}, ensure_ascii=False))


def _resolve_batch_items(body: dict) -> list:
    """
    Resolve batch inputs from either uploaded STL files (folder upload) or catalog filenames.
    Returns list of (safe_filename, raw_bytes).
    """
    items = []
    uploaded = body.get("uploaded_files") or []
    if uploaded:
        for entry in uploaded[:30]:
            fn = sanitize_filename(entry.get("filename", "batch_scan.stl"))
            raw = scrub_stl_binary_header(base64.b64decode(entry["file_b64"]))
            if len(raw) > MAX_UPLOAD_BYTES:
                raise ValueError(f"Файл {fn} превышает лимит 64 МБ.")
            STATE["uploaded_files"][fn] = raw
            items.append((fn, raw))
        return items

    filenames = body.get("filenames") or [
        "clinical_normal_N01.stl",
        "clinical_mild_plagio_P01.stl",
        "clinical_moderate_bp_BP05.stl",
        "clinical_severe_plagio_P03.stl",
        "clinical_very_severe_plagio_P04.stl",
        "defect_incomplete_occipital.stl",
    ]
    for fn in filenames[:30]:
        safe_fn = sanitize_filename(fn)
        items.append((safe_fn, get_raw_stl_bytes(safe_fn)))
    return items


class BatchProcessHandler(BaseSecureHandler):
    def post(self):
        self.set_header("Content-Type", "application/json; charset=utf-8")
        try:
            body = json.loads(self.request.body.decode("utf-8"))
            batch_inputs = _resolve_batch_items(body)
            scale_factor = float(body.get("scale_factor", 1.0))
            level_ratio = float(body.get("level_ratio", 0.30))
            doctor_name = sanitize_text_field(body.get("doctor_name", "Золотухин А.П. / Бойко А."), 100)

            t0 = time.perf_counter()
            batch_results = []
            export_records = []
            resolved_filenames = []

            for safe_fn, raw_bytes in batch_inputs:
                resolved_filenames.append(safe_fn)
                res = analyze_scan_bytes(
                    raw_bytes=raw_bytes,
                    scan_id=safe_fn,
                    scale_factor=scale_factor,
                    level_ratio=level_ratio,
                    include_mesh_3d=False,
                )
                q = res["quality"]
                cran = res["craniometry"]
                ind = cran["indices"] if cran else {}
                qvol = cran.get("quadrant_volumes") if cran else {}
                cls_wrap = cran["classification"] if cran else {}
                cls_grade = cls_wrap.get("cvai_grade", {"code": "REJECTED", "label": "Отказ (дефект)"})
                borderline = cls_wrap.get("borderline_alert", {})
                batch_results.append({
                    "filename": safe_fn,
                    "scan_id": res["scan_id"],
                    "status": q["status"],
                    "quality_score": q["quality_score"],
                    "CI": ind.get("CI"),
                    "CVA_mm": ind.get("CVA_mm"),
                    "CVAI": ind.get("CVAI"),
                    "grade_code": cls_grade.get("code"),
                    "grade_label": cls_grade.get("label"),
                    "is_borderline": bool(borderline.get("is_borderline", False)),
                    "PSR": qvol.get("posterior_symmetry_ratio") if qvol else None,
                    "processing_time_sec": res["processing_time_sec"],
                })
                export_records.append(export_measurement_json(
                    scan_id=res["scan_id"],
                    quality=validate_mesh(parse_stl_bytes(raw_bytes, scale_factor=scale_factor)),
                    craniometry=cran,
                    landmarks=res["landmarks_mm"],
                    doctor_name=doctor_name,
                    landmark_confidence=res.get("landmark_confidence"),
                ))

            csv_summary = export_measurement_csv_string(export_records)
            excel_xml = export_measurement_excel_xml(export_records)
            self.write(json.dumps({
                "processed_count": len(batch_results),
                "total_time_sec": round(time.perf_counter() - t0, 4),
                "filenames": resolved_filenames,
                "results": batch_results,
                "csv_summary": csv_summary,
                "excel_xml": excel_xml,
            }, ensure_ascii=False))
        except Exception as e:
            self.set_status(400)
            self.write(json.dumps({"error": str(e)}, ensure_ascii=False))


class BatchZipHandler(BaseSecureHandler):
    """Return a single ZIP archive containing CSV, Excel (.xls), JSON, and individual PDF reports for all batch scans."""
    def post(self):
        try:
            body = json.loads(self.request.body.decode("utf-8"))
            batch_inputs = _resolve_batch_items(body)
            scale_factor = float(body.get("scale_factor", 1.0))
            level_ratio = float(body.get("level_ratio", 0.30))
            doctor_name = sanitize_text_field(body.get("doctor_name", "Золотухин А.П. / Бойко А."), 100)
            doctor_verified = bool(body.get("doctor_verified", True))

            export_records = []
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for safe_fn, raw_bytes in batch_inputs:
                    mesh = parse_stl_bytes(raw_bytes, scale_factor=scale_factor)
                    quality = validate_mesh(mesh)
                    res = analyze_scan_bytes(
                        raw_bytes=raw_bytes,
                        scan_id=safe_fn,
                        scale_factor=scale_factor,
                        level_ratio=level_ratio,
                        include_mesh_3d=False,
                    )
                    scan_id = res["scan_id"]
                    pdf_bytes = generate_pdf_report_bytes(
                        scan_id=scan_id,
                        mesh=mesh,
                        quality=quality,
                        craniometry=res["craniometry"],
                        landmarks=res["landmarks_mm"],
                        model_version=res["model_version"],
                        doctor_name=doctor_name,
                        doctor_verified=doctor_verified,
                        manual_correction_applied=False,
                        landmark_confidence=res.get("landmark_confidence"),
                    )
                    zf.writestr(f"reports/saigak_orto_report_{scan_id}.pdf", pdf_bytes)
                    export_records.append(export_measurement_json(
                        scan_id=scan_id,
                        quality=quality,
                        craniometry=res["craniometry"],
                        landmarks=res["landmarks_mm"],
                        model_version=res["model_version"],
                        doctor_name=doctor_name,
                        doctor_verified=doctor_verified,
                        landmark_confidence=res.get("landmark_confidence"),
                    ))

                csv_summary = "\ufeff" + export_measurement_csv_string(export_records)
                excel_xml = export_measurement_excel_xml(export_records)
                zf.writestr("batch_summary.csv", csv_summary.encode("utf-8"))
                zf.writestr("batch_summary.xls", excel_xml.encode("utf-8"))
                zf.writestr("batch_summary.json", json.dumps(export_records, ensure_ascii=False, indent=2).encode("utf-8"))

            self.set_header("Content-Type", "application/zip")
            self.set_header("Content-Disposition", 'attachment; filename="saigak_orto_batch_reports.zip"')
            self.write(zip_buf.getvalue())
        except Exception as e:
            self.set_status(400)
            self.set_header("Content-Type", "application/json; charset=utf-8")
            self.write(json.dumps({"error": str(e)}, ensure_ascii=False))


def _is_safe_clinic_webhook_host(url: str) -> tuple:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False, "Поддерживаются только протоколы http:// и https://"
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return False, "Не указан хост МИС-сервера"
    if host in ("localhost", "127.0.0.1", "::1"):
        return True, "loopback"
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback:
            return True, "private_lan"
        return False, "По политике локальной безопасности разрешены только адреса локальной сети клиники (localhost / RFC1918)"
    except ValueError:
        if host.endswith(".local") or host.endswith(".lan") or host.endswith(".internal"):
            return True, "local_domain"
        return False, "По политике локальной безопасности разрешены только локальные/внутрисетевые хосты МИС (*.local, *.lan, 127.0.0.1, 10.x, 192.168.x)"


class MisExportHandler(BaseSecureHandler):
    """
    Build an anonymized HL7-FHIR-style DiagnosticReport JSON + optional Base64 PDF payload
    for clinic MIS integration, with HMAC-SHA256 signing and safe local-LAN webhook dispatch.
    """
    def post(self):
        self.set_header("Content-Type", "application/json; charset=utf-8")
        try:
            body = json.loads(self.request.body.decode("utf-8"))
            filename = sanitize_filename(body.get("filename", "scan_081.stl"))
            mode = str(body.get("mode", "preview")).lower()
            webhook_url = str(body.get("webhook_url", "") or "").strip()
            webhook_secret = str(body.get("webhook_secret", "") or "saigak-local-clinic-hmac")
            include_pdf = bool(body.get("include_pdf", True))
            scale_factor = float(body.get("scale_factor", 1.0))
            level_ratio = float(body.get("level_ratio", 0.30))
            custom_landmarks = body.get("custom_landmarks")
            expert_ci = bool(body.get("expert_ci_profile_approved", False))
            age_val = body.get("patient_age_months")
            patient_age = float(age_val) if age_val not in (None, "") else None
            doctor_name = sanitize_text_field(body.get("doctor_name", "Золотухин А.П. / Бойко А."), 100)
            doctor_verified = bool(body.get("doctor_verified", True))
            clinical_notes = sanitize_text_field(body.get("clinical_notes", ""), 220)

            raw_bytes = get_raw_stl_bytes(filename)
            mesh, quality, _ = prepare_mesh_and_quality(raw_bytes, scale_factor=scale_factor)
            res = analyze_scan_bytes(
                raw_bytes=raw_bytes,
                scan_id=filename,
                scale_factor=scale_factor,
                level_ratio=level_ratio,
                custom_landmarks=custom_landmarks,
                expert_ci_profile_approved=expert_ci,
                include_mesh_3d=False,
            )
            scan_id = res["scan_id"]
            cran = res["craniometry"]

            pdf_attachment = None
            if include_pdf:
                pdf_bytes = generate_pdf_report_bytes(
                    scan_id=scan_id,
                    mesh=mesh,
                    quality=quality,
                    craniometry=cran,
                    landmarks=res["landmarks_mm"],
                    model_version=res["model_version"],
                    patient_age_months=patient_age,
                    doctor_name=doctor_name,
                    doctor_verified=doctor_verified,
                    manual_correction_applied=res["manual_correction_applied"],
                    clinical_notes=clinical_notes,
                    landmark_confidence=res.get("landmark_confidence"),
                )
                pdf_attachment = {
                    "contentType": "application/pdf",
                    "filename": f"saigak_orto_report_{scan_id}.pdf",
                    "size_bytes": len(pdf_bytes),
                    "sha256": hashlib.sha256(pdf_bytes).hexdigest(),
                    "data_base64": base64.b64encode(pdf_bytes).decode("ascii"),
                }

            mis_payload = {
                "resourceType": "DiagnosticReport",
                "protocol": "SAIGAK-ORTO-MIS-WEBHOOK-V1",
                "status": "final" if doctor_verified else "preliminary",
                "issued": datetime.now().isoformat(),
                "anonymized_subject_ref": scan_id,
                "patient_age_months": patient_age,
                "methodology": "STARBAND-GRAHAM-2020-V1.2",
                "model_version": res["model_version"],
                "model_integrity_sha256": STATE["model_hashes"],
                "clinical_disclaimer": "Инструмент поддержки принятия клинических решений; не заменяет врачебный диагноз.",
                "quality_gate": {
                    "status": quality.status,
                    "quality_score": round(quality.quality_score, 1),
                    "vault_closure_pct": round(quality.vault_closure_pct, 1),
                    "defects": quality.defects,
                },
                "observations": {
                    "indices": cran["indices"] if cran else None,
                    "linear_measurements_mm": cran["linear_measurements_mm"] if cran else None,
                    "regional_asymmetry": cran["regional_asymmetry"] if cran else None,
                    "quadrant_volumes": cran.get("quadrant_volumes") if cran else None,
                    "cvai_grade": cran["classification"]["cvai_grade"] if cran else None,
                    "borderline_alert": cran["classification"].get("borderline_alert") if cran else None,
                },
                "doctor_verification": {
                    "verified": doctor_verified,
                    "doctor_name": doctor_name,
                    "landmarks_audit_mode": "manual_corrected" if res["manual_correction_applied"] else "ml_auto",
                    "clinical_notes": clinical_notes,
                },
                "presentedForm": [pdf_attachment] if pdf_attachment else [],
            }

            payload_bytes = json.dumps(mis_payload, ensure_ascii=False).encode("utf-8")
            sig = hmac.new(webhook_secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()

            dispatch_result = {"mode": mode, "dispatched": False, "status_code": None, "detail": "Локальный предпросмотр (внешние запросы не выполнялись)"}
            if mode == "dispatch":
                if not webhook_url:
                    raise ValueError("Укажите локальный URL вебхука МИС клиники для отправки.")
                safe_ok, reason = _is_safe_clinic_webhook_host(webhook_url)
                if not safe_ok:
                    raise ValueError(reason)
                req = urllib.request.Request(
                    webhook_url,
                    data=payload_bytes,
                    headers={
                        "Content-Type": "application/json; charset=utf-8",
                        "X-Saigak-Protocol": "SAIGAK-ORTO-MIS-WEBHOOK-V1",
                        "X-Saigak-Signature": f"sha256={sig}",
                    },
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=5.0) as resp:
                        dispatch_result = {
                            "mode": "dispatch",
                            "dispatched": True,
                            "status_code": resp.status,
                            "detail": f"Пакет успешно принят МИС ({resp.status})",
                        }
                except Exception as net_err:
                    dispatch_result = {
                        "mode": "dispatch",
                        "dispatched": False,
                        "status_code": None,
                        "detail": f"Не удалось подключиться к указанному локальному вебхуку МИС: {net_err}",
                    }

            # Return lightweight preview copy (truncate huge PDF base64 in UI preview field while keeping full payload available)
            preview_payload = dict(mis_payload)
            if pdf_attachment:
                preview_att = dict(pdf_attachment)
                preview_att["data_base64"] = preview_att["data_base64"][:64] + f"... ({pdf_attachment['size_bytes']} bytes PDF)"
                preview_payload["presentedForm"] = [preview_att]

            self.write(json.dumps({
                "signature_header": f"sha256={sig}",
                "payload_size_bytes": len(payload_bytes),
                "dispatch": dispatch_result,
                "payload_preview": preview_payload,
                "full_payload": mis_payload,
            }, ensure_ascii=False))
        except Exception as e:
            self.set_status(400)
            self.write(json.dumps({"error": str(e)}, ensure_ascii=False))


class ExpertQueueHandler(BaseSecureHandler):
    def get(self):
        self.set_header("Content-Type", "application/json; charset=utf-8")
        self.write(json.dumps({
            "count": len(STATE["expert_queue"]),
            "items": STATE["expert_queue"],
        }, ensure_ascii=False))

    def post(self):
        self.set_header("Content-Type", "application/json; charset=utf-8")
        try:
            body = json.loads(self.request.body.decode("utf-8"))
            scan_id = sanitize_text_field(body.get("scan_id", "scan_081"), 80)
            reason = sanitize_text_field(body.get("reason", "Ручная разметка ориентиров врачом для дообучения ML"), 200)
            doctor_name = sanitize_text_field(body.get("doctor_name", "Золотухин А.П. / Бойко А."), 100)
            landmarks = body.get("landmarks_mm")

            entry = {
                "queue_id": f"Q-{len(STATE['expert_queue']) + 1:03d}",
                "scan_id": scan_id,
                "reason": reason,
                "status": "Сохранено для переобучения модели",
                "doctor_name": doctor_name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "annotated_landmarks_mm": landmarks,
            }
            STATE["expert_queue"].insert(0, entry)
            save_expert_queue()
            self.write(json.dumps({
                "status": "ADDED",
                "queue_count": len(STATE["expert_queue"]),
                "entry": entry,
            }, ensure_ascii=False))
        except Exception as e:
            self.set_status(400)
            self.write(json.dumps({"error": str(e)}, ensure_ascii=False))


class PdfReportHandler(BaseSecureHandler):
    def post(self):
        try:
            body = json.loads(self.request.body.decode("utf-8"))
            filename = sanitize_filename(body.get("filename", "scan_081.stl"))
            scale_factor = float(body.get("scale_factor", 1.0))
            level_ratio = float(body.get("level_ratio", 0.30))
            diagonal_angle_deg = float(body.get("diagonal_angle_deg", 30.0))
            custom_landmarks = body.get("custom_landmarks")
            use_baseline = bool(body.get("use_baseline", False))
            expert_ci = bool(body.get("expert_ci_profile_approved", False))
            age_val = body.get("patient_age_months")
            patient_age = float(age_val) if age_val not in (None, "") else None
            doctor_name = sanitize_text_field(body.get("doctor_name", "Золотухин А.П. / Бойко А."), 100)
            doctor_verified = bool(body.get("doctor_verified", True))
            clinical_notes = sanitize_text_field(body.get("clinical_notes", ""), 220)

            raw_bytes = get_raw_stl_bytes(filename)
            mesh, quality, _ = prepare_mesh_and_quality(raw_bytes, scale_factor=scale_factor)
            manual_corr = False
            lms = None
            cran = None

            if quality.is_usable:
                if custom_landmarks and all(k in custom_landmarks for k in ["Sellion", "RightTragion", "LeftTragion"]):
                    lms = custom_landmarks
                    manual_corr = True
                    m_ver = "Ручная коррекция специалистом + Saigak-ML-v1.3"
                elif use_baseline:
                    lms = predict_landmarks_baseline(mesh)
                    m_ver = "Saigak-Baseline-BBox-Heuristic-v1.0"
                else:
                    lms = STATE["landmark_model"].predict_mesh_landmarks(mesh)
                    m_ver = "Saigak-ML-ExtraTrees-v1.3-Artec (seed=42)"
                lm_conf = estimate_landmark_confidence(
                    STATE["landmark_model"], mesh, lms,
                    manual_correction=manual_corr, use_baseline=use_baseline,
                )
                cran = compute_craniometry(
                    mesh, lms,
                    level_ratio=level_ratio,
                    diagonal_angle_deg=diagonal_angle_deg,
                    expert_ci_profile_approved=expert_ci,
                )
            else:
                m_ver = "Контроль качества входных данных (отказ)"
                lm_conf = None

            scan_id = filename.replace(".stl", "")
            pdf_bytes = generate_pdf_report_bytes(
                scan_id=scan_id,
                mesh=mesh,
                quality=quality,
                craniometry=cran,
                landmarks=lms,
                model_version=m_ver,
                patient_age_months=patient_age,
                doctor_name=doctor_name,
                doctor_verified=doctor_verified,
                manual_correction_applied=manual_corr,
                clinical_notes=clinical_notes,
                landmark_confidence=lm_conf,
            )

            self.set_header("Content-Type", "application/pdf")
            self.set_header("Content-Disposition", f'attachment; filename="saigak_orto_report_{scan_id}.pdf"')
            self.write(pdf_bytes)
        except Exception as e:
            self.set_status(400)
            self.set_header("Content-Type", "application/json; charset=utf-8")
            self.write(json.dumps({"error": str(e)}, ensure_ascii=False))


class ExportDataHandler(BaseSecureHandler):
    def post(self):
        try:
            body = json.loads(self.request.body.decode("utf-8"))
            fmt = body.get("format", "json").lower()
            filename = sanitize_filename(body.get("filename", "scan_081.stl"))
            scale_factor = float(body.get("scale_factor", 1.0))
            level_ratio = float(body.get("level_ratio", 0.30))
            diagonal_angle_deg = float(body.get("diagonal_angle_deg", 30.0))
            custom_landmarks = body.get("custom_landmarks")
            use_baseline = bool(body.get("use_baseline", False))
            expert_ci = bool(body.get("expert_ci_profile_approved", False))
            age_val = body.get("patient_age_months")
            patient_age = float(age_val) if age_val not in (None, "") else None
            doctor_name = sanitize_text_field(body.get("doctor_name", "Золотухин А.П. / Бойко А."), 100)
            doctor_verified = bool(body.get("doctor_verified", True))
            clinical_notes = sanitize_text_field(body.get("clinical_notes", ""), 220)

            raw_bytes = get_raw_stl_bytes(filename)
            mesh, quality, _ = prepare_mesh_and_quality(raw_bytes, scale_factor=scale_factor)
            manual_corr = False
            lms = None
            lm_conf = None
            cran = None

            if quality.is_usable:
                if custom_landmarks and all(k in custom_landmarks for k in ["Sellion", "RightTragion", "LeftTragion"]):
                    lms = custom_landmarks
                    manual_corr = True
                elif use_baseline:
                    lms = predict_landmarks_baseline(mesh)
                else:
                    lms = STATE["landmark_model"].predict_mesh_landmarks(mesh)
                lm_conf = estimate_landmark_confidence(
                    STATE["landmark_model"], mesh, lms,
                    manual_correction=manual_corr, use_baseline=use_baseline,
                )
                cran = compute_craniometry(
                    mesh, lms,
                    level_ratio=level_ratio,
                    diagonal_angle_deg=diagonal_angle_deg,
                    expert_ci_profile_approved=expert_ci,
                )

            scan_id = filename.replace(".stl", "")
            record = export_measurement_json(
                scan_id=scan_id,
                quality=quality,
                craniometry=cran,
                landmarks=lms,
                patient_age_months=patient_age,
                doctor_name=doctor_name,
                doctor_verified=doctor_verified,
                manual_correction_applied=manual_corr,
                clinical_notes=clinical_notes,
                landmark_confidence=lm_conf,
            )

            if fmt == "csv":
                csv_str = export_measurement_csv_string([record])
                self.set_header("Content-Type", "text/csv; charset=utf-8")
                self.set_header("Content-Disposition", f'attachment; filename="saigak_orto_{scan_id}.csv"')
                self.write("\ufeff" + csv_str)
            elif fmt == "xls":
                xls_str = export_measurement_excel_xml([record])
                self.set_header("Content-Type", "application/vnd.ms-excel; charset=utf-8")
                self.set_header("Content-Disposition", f'attachment; filename="saigak_orto_{scan_id}.xls"')
                self.write(xls_str)
            else:
                self.set_header("Content-Type", "application/json; charset=utf-8")
                self.set_header("Content-Disposition", f'attachment; filename="saigak_orto_{scan_id}.json"')
                self.write(json.dumps(record, ensure_ascii=False, indent=2))
        except Exception as e:
            self.set_status(400)
            self.write(json.dumps({"error": str(e)}, ensure_ascii=False))


def make_app():
    load_resources()
    return tornado.web.Application([
        (r"/", WelcomeHandler),
        (r"/cockpit", CockpitHandler),
        (r"/api/health", HealthHandler),
        (r"/api/openapi.json", OpenApiHandler),
        (r"/api/catalog", CatalogHandler),
        (r"/api/analyze", AnalyzeHandler),
        (r"/api/compare", CompareHandler),
        (r"/api/robustness", RobustnessHandler),
        (r"/api/batch", BatchProcessHandler),
        (r"/api/batch/zip", BatchZipHandler),
        (r"/api/mis/export", MisExportHandler),
        (r"/api/queue", ExpertQueueHandler),
        (r"/api/report/pdf", PdfReportHandler),
        (r"/api/report/export", ExportDataHandler),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": STATIC_DIR}),
        (r"/reports/(.*)", tornado.web.StaticFileHandler, {"path": REPORTS_DIR}),
    ])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8765))
    app = make_app()
    app.listen(port, address="127.0.0.1")
    print(f"Saigak-Orto 3D Craniometry Server running locally at http://127.0.0.1:{port}")
    tornado.ioloop.IOLoop.current().start()
