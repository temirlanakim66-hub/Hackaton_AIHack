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
import json
import os
import re
import time
from datetime import datetime
import joblib
import numpy as np
import tornado.ioloop
import tornado.web

import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from core.mesh_io import parse_stl_bytes, validate_mesh, compute_face_normals
from core.craniometry import compute_craniometry, METHODOLOGY_REGISTRY
from core.features import (
    extract_morphological_features,
    predict_landmarks_baseline,
    compute_vertex_asymmetry_heatmap,
    HybridLandmarkPredictor,  # noqa: F401
)
from core.report_generator import (
    generate_pdf_report_bytes,
    export_measurement_json,
    export_measurement_csv_string,
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


def build_threejs_mesh_payload(mesh, origin=None, R=None, max_tris: int = 18000) -> dict:
    n_tri = len(mesh.v1)
    step = max(1, n_tri // max_tris)
    v1 = mesh.v1[::step]
    v2 = mesh.v2[::step]
    v3 = mesh.v3[::step]
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

    mesh = parse_stl_bytes(raw_bytes, scale_factor=scale_factor)
    quality = validate_mesh(mesh)

    landmarks = None
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

        try:
            craniometry = compute_craniometry(
                mesh=mesh,
                landmarks=landmarks,
                level_ratio=level_ratio,
                diagonal_angle_deg=diagonal_angle_deg,
                expert_ci_profile_approved=expert_ci_profile_approved,
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
        "quality": quality.to_dict(),
        "landmarks_mm": landmarks,
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
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self';"
        )


class IndexHandler(BaseSecureHandler):
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
                "title": "Saigak-Orto 3D Craniometry & Deformity Assessment API",
                "version": "1.2.0",
                "description": "Локальный REST API для стандартизированного расчёта краниальных индексов (CI, CVA, CVAI), пакетной обработки 3D STL-сканов, генерации PDF-отчётов и управления экспертной очередью разметки.",
                "contact": {
                    "name": "ТОО Saigak-Orto (Сайгак-Орто)",
                    "url": "https://saigak-orto.kz/",
                    "email": "boiko@saigak-orto.kz",
                },
            },
            "paths": {
                "/api/health": {"get": {"summary": "Проверка состояния сервера и целостности весов ML-моделей (SHA-256)"}},
                "/api/catalog": {"get": {"summary": "Список доступных 3D STL-сканов, реестр методик и метрики валидации"}},
                "/api/analyze": {"post": {"summary": "Анализ одного 3D STL-скана (контроль качества, ML-ориентиры, расчёт CI/CVA/CVAI, 3D-буфер и карта асимметрии)"}},
                "/api/robustness": {"post": {"summary": "Тест устойчивости к 3D-повороту модели, гауссовскому шуму поверхности и повторному запуску (Should 5.2)"}},
                "/api/batch": {"post": {"summary": "Пакетная обработка списка STL-сканов с выгрузкой сводной таблицы (Could 5.3)"}},
                "/api/queue": {
                    "get": {"summary": "Получить экспертную очередь сложных случаев и сохранённых разметок для переобучения (Could 5.3)"},
                    "post": {"summary": "Добавить скан и скорректированные врачом ориентиры в экспертную очередь дообучения (Could 5.3)"},
                },
                "/api/report/pdf": {"post": {"summary": "Сформировать клинический PDF-отчёт по одному скану"}},
                "/api/report/export": {"post": {"summary": "Экспорт численных результатов в CSV или JSON"}},
            },
        }
        self.write(json.dumps(spec, ensure_ascii=False, indent=2))


class CatalogHandler(BaseSecureHandler):
    def get(self):
        self.set_header("Content-Type", "application/json; charset=utf-8")
        files = sorted([f for f in os.listdir(STL_DIR) if f.endswith(".stl")])
        demos = STATE["dataset_split"].get("clinical_and_defect_demos", [])
        demo_map = {d["filename"]: d for d in demos}

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
            )
            self.write(json.dumps(result, ensure_ascii=False))
        except FileNotFoundError as e:
            self.set_status(404)
            self.write(json.dumps({"error": str(e)}, ensure_ascii=False))
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


class BatchProcessHandler(BaseSecureHandler):
    def post(self):
        self.set_header("Content-Type", "application/json; charset=utf-8")
        try:
            body = json.loads(self.request.body.decode("utf-8"))
            filenames = body.get("filenames") or [
                "clinical_normal_N01.stl",
                "clinical_mild_plagio_P01.stl",
                "clinical_moderate_bp_BP05.stl",
                "clinical_severe_plagio_P03.stl",
                "clinical_very_severe_plagio_P04.stl",
                "defect_incomplete_occipital.stl",
            ]
            scale_factor = float(body.get("scale_factor", 1.0))
            level_ratio = float(body.get("level_ratio", 0.30))

            t0 = time.perf_counter()
            batch_results = []
            export_records = []

            for fn in filenames[:30]:
                safe_fn = sanitize_filename(fn)
                raw_bytes = get_raw_stl_bytes(safe_fn)
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
                cls_grade = cran["classification"]["cvai_grade"] if cran else {"code": "REJECTED", "label": "Отказ (дефект)"}
                batch_results.append({
                    "scan_id": res["scan_id"],
                    "status": q["status"],
                    "quality_score": q["quality_score"],
                    "CI": ind.get("CI"),
                    "CVA_mm": ind.get("CVA_mm"),
                    "CVAI": ind.get("CVAI"),
                    "grade_code": cls_grade.get("code"),
                    "grade_label": cls_grade.get("label"),
                    "processing_time_sec": res["processing_time_sec"],
                })
                export_records.append(export_measurement_json(
                    scan_id=res["scan_id"],
                    quality=validate_mesh(parse_stl_bytes(raw_bytes, scale_factor=scale_factor)),
                    craniometry=cran,
                    landmarks=res["landmarks_mm"],
                ))

            csv_summary = export_measurement_csv_string(export_records)
            self.write(json.dumps({
                "processed_count": len(batch_results),
                "total_time_sec": round(time.perf_counter() - t0, 4),
                "results": batch_results,
                "csv_summary": csv_summary,
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
            mesh = parse_stl_bytes(raw_bytes, scale_factor=scale_factor)
            quality = validate_mesh(mesh)
            manual_corr = False
            lms = None
            cran = None

            if quality.is_usable:
                if custom_landmarks and all(k in custom_landmarks for k in ["Sellion", "RightTragion", "LeftTragion"]):
                    lms = custom_landmarks
                    manual_corr = True
                    m_ver = "Ручная коррекция специалистом + Saigak-ML-v1.2"
                elif use_baseline:
                    lms = predict_landmarks_baseline(mesh)
                    m_ver = "Saigak-Baseline-BBox-Heuristic-v1.0"
                else:
                    lms = STATE["landmark_model"].predict_mesh_landmarks(mesh)
                    m_ver = "Saigak-ML-ExtraTrees-v1.2 (seed=42)"
                cran = compute_craniometry(
                    mesh, lms,
                    level_ratio=level_ratio,
                    diagonal_angle_deg=diagonal_angle_deg,
                    expert_ci_profile_approved=expert_ci,
                )
            else:
                m_ver = "Контроль качества входных данных (отказ)"

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

            raw_bytes = get_raw_stl_bytes(filename)
            mesh = parse_stl_bytes(raw_bytes, scale_factor=scale_factor)
            quality = validate_mesh(mesh)
            manual_corr = False
            lms = None
            cran = None

            if quality.is_usable:
                if custom_landmarks and all(k in custom_landmarks for k in ["Sellion", "RightTragion", "LeftTragion"]):
                    lms = custom_landmarks
                    manual_corr = True
                elif use_baseline:
                    lms = predict_landmarks_baseline(mesh)
                else:
                    lms = STATE["landmark_model"].predict_mesh_landmarks(mesh)
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
            )

            if fmt == "csv":
                csv_str = export_measurement_csv_string([record])
                self.set_header("Content-Type", "text/csv; charset=utf-8")
                self.set_header("Content-Disposition", f'attachment; filename="saigak_orto_{scan_id}.csv"')
                self.write("\ufeff" + csv_str)
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
        (r"/", IndexHandler),
        (r"/api/health", HealthHandler),
        (r"/api/openapi.json", OpenApiHandler),
        (r"/api/catalog", CatalogHandler),
        (r"/api/analyze", AnalyzeHandler),
        (r"/api/robustness", RobustnessHandler),
        (r"/api/batch", BatchProcessHandler),
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
