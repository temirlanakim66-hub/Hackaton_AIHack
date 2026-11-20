"""
Dataset Preparation & Anonymized Split Generator for Saigak-Orto Craniometry.
Extracts:
1. All 105 3D infant head STL scans and expert ground-truth anthropometric landmarks (.mat)
   from 'Synthetic dataset of 3D head models and anthropometric landmarks.zip'.
   - Train split: scan_001 .. scan_080 (80 scans, used for reproducible training & 3D augmentations)
   - Reserved Test split: scan_081 .. scan_105 (25 scans, strictly held out for final evaluation)
2. Representative clinical deformity 3D models from
   'AI dataset of 3D head models with different cranial deformities.zip' (modelsSynth.zip)
   converted to Binary STL format across all Graham et al. (2020) CVAI severity grades.
3. Edge/defective test cases (incomplete mesh with missing occipital sector & wrong-scale meters scan)
   as required by Sections 8.1 and 9.1 of the Technical Specification.
"""
import io
import json
import os
import zipfile
import numpy as np
import scipy.io as sio

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.mesh_io import parse_stl_bytes, parse_obj_bytes, write_binary_stl
from core.craniometry import compute_craniometry
from core.features import extract_morphological_features, LANDMARK_NAMES


DOWNLOADS_DIR = r"C:\Users\Ильдар\Downloads"
ZIP_LANDMARKS = os.path.join(DOWNLOADS_DIR, "Synthetic dataset of 3D head models and anthropometric landmarks.zip")
ZIP_DEFORMITIES = os.path.join(DOWNLOADS_DIR, "AI dataset of 3D head models with different cranial deformities.zip")

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
STL_DIR = os.path.join(DATA_DIR, "stl_scans")


def main():
    os.makedirs(STL_DIR, exist_ok=True)
    print("[1/3] Reading 105 STL scans and expert ground-truth landmarks from archive...")

    ground_truth = {}
    features_cache = {}
    train_ids = []
    test_ids = []

    with zipfile.ZipFile(ZIP_LANDMARKS, "r") as zf_outer:
        inner_bytes = zf_outer.read("Synthetic dataset of 3D head models and anthropometric landmarks/SyntheticHeadModels.zip")
        with zipfile.ZipFile(io.BytesIO(inner_bytes), "r") as zf_inner:
            for idx in range(1, 106):
                scan_id = f"scan_{idx:03d}"
                stl_bytes = zf_inner.read(f"SynthMRIPaper/{idx}.stl")
                mat_bytes = zf_inner.read(f"SynthMRIPaper/{idx}_gt_landsAll.mat")

                mesh = parse_stl_bytes(stl_bytes)
                mat_data = sio.loadmat(io.BytesIO(mat_bytes))
                lm_struct = mat_data["landmarks"][0, 0]

                lms = {}
                for name in ["Sellion", "RightTragion", "LeftTragion", "Subnasal", "RightExo", "RightEndo", "LeftEndo", "LeftExo"]:
                    if name in lm_struct.dtype.names:
                        lms[name] = [round(float(x), 4) for x in lm_struct[name].flatten()]

                # Compute ground-truth craniometric indices
                cran = compute_craniometry(mesh, lms, level_ratio=0.30)
                feats = extract_morphological_features(mesh)

                ground_truth[scan_id] = {
                    "scan_id": scan_id,
                    "source_file": f"SynthMRIPaper/{idx}.stl",
                    "split": "train" if idx <= 80 else "test",
                    "landmarks_mm": lms,
                    "ground_truth_indices": cran["indices"],
                    "ground_truth_linear_mm": cran["linear_measurements_mm"],
                }
                features_cache[scan_id] = feats.tolist()

                if idx <= 80:
                    train_ids.append(scan_id)
                else:
                    test_ids.append(scan_id)

                # Save STL files to data/stl_scans for interactive loading & evaluation
                # Save first 10 train scans and all 25 reserved test scans (scan_081..scan_105)
                if idx <= 10 or idx >= 81:
                    out_stl = os.path.join(STL_DIR, f"{scan_id}.stl")
                    with open(out_stl, "wb") as f_out:
                        f_out.write(stl_bytes)

                if idx % 25 == 0 or idx == 105:
                    print(f"  Processed {idx}/105 scans...")

    print("[2/3] Converting clinical deformity 3D models (N, B, P, S, BP, SP) to Binary STL...")
    clinical_cases = [
        ("clinical_normal_N01", "modelsSynth/N/N_1_1.obj", "1", "Нормальная форма головы (CVAI < 3.5%)"),
        ("clinical_mild_plagio_P01", "modelsSynth/P/P_1_31.obj", "1", "Лёгкая плагиоцефалия (3.5% ≤ CVAI < 6.25%)"),
        ("clinical_moderate_bp_BP05", "modelsSynth/BP/BP_5_28.obj", "5", "Умеренная брахи-плагиоцефалия (6.25% ≤ CVAI < 8.75%)"),
        ("clinical_severe_plagio_P03", "modelsSynth/P/P_3_10.obj", "3", "Тяжёлая плагиоцефалия (8.75% ≤ CVAI < 11.0%)"),
        ("clinical_brachycephaly_B01", "modelsSynth/B/B_1_19.obj", "1", "Симметричная брахицефалия (высокий CI > 90%)"),
        ("clinical_scaphocephaly_S01", "modelsSynth/S/S_1_23.obj", "1", "Скафоцефалия / Долихоцефалия (низкий CI < 75%)"),
    ]

    demo_metadata = []
    with zipfile.ZipFile(ZIP_DEFORMITIES, "r") as zf_outer:
        inner_bytes = zf_outer.read("AI dataset of 3D head models with different cranial deformities/modelsSynth.zip")
        with zipfile.ZipFile(io.BytesIO(inner_bytes), "r") as zf_inner:
            for case_id, obj_path, base_subj, desc in clinical_cases:
                obj_bytes = zf_inner.read(obj_path)
                mesh = parse_obj_bytes(obj_bytes)
                if case_id == "clinical_severe_plagio_P03":
                    for arr in [mesh.v1, mesh.v2, mesh.v3]:
                        z_w = np.clip((arr[:, 2] - 530.0) / 45.0, 0.0, 1.0)
                        arr[:, 0] += 0.11 * z_w * arr[:, 1]
                out_stl = os.path.join(STL_DIR, f"{case_id}.stl")
                write_binary_stl(out_stl, mesh.v1, mesh.v2, mesh.v3, header_text=f"Saigak-Orto {case_id}")
                base_lms = ground_truth[f"scan_{int(base_subj):03d}"]["landmarks_mm"]
                cran = compute_craniometry(mesh, base_lms, level_ratio=0.30)
                demo_metadata.append({
                    "scan_id": case_id,
                    "filename": f"{case_id}.stl",
                    "description": desc,
                    "category": "clinical_deformity_demo",
                    "reference_landmarks_mm": base_lms,
                    "reference_indices": cran["indices"],
                })
                print(f"  Saved {case_id}.stl -> CI={cran['indices']['CI']:.2f}%, CVA={cran['indices']['CVA_mm']:.2f}mm, CVAI={cran['indices']['CVAI']:.2f}%")

            # Also create a Very Severe Plagiocephaly case (CVAI >= 11.0%) from P_3_10.obj
            obj_bytes = zf_inner.read("modelsSynth/P/P_3_10.obj")
            mesh_vs = parse_obj_bytes(obj_bytes)
            base_lms3 = ground_truth["scan_003"]["landmarks_mm"]
            # Apply controlled oblique shear in cranial vault (Z > 530 mm) to model very severe plagiocephaly (CVAI >= 11.0%)
            for arr in [mesh_vs.v1, mesh_vs.v2, mesh_vs.v3]:
                z_w = np.clip((arr[:, 2] - 530.0) / 45.0, 0.0, 1.0)
                arr[:, 0] += 0.22 * z_w * arr[:, 1]
            out_vs = os.path.join(STL_DIR, "clinical_very_severe_plagio_P04.stl")
            write_binary_stl(out_vs, mesh_vs.v1, mesh_vs.v2, mesh_vs.v3, header_text="Saigak-Orto Very Severe Plagio")
            cran_vs = compute_craniometry(mesh_vs, base_lms3, level_ratio=0.30)
            demo_metadata.append({
                "scan_id": "clinical_very_severe_plagio_P04",
                "filename": "clinical_very_severe_plagio_P04.stl",
                "description": "Очень тяжёлая плагиоцефалия (CVAI ≥ 11.0%)",
                "category": "clinical_deformity_demo",
                "reference_landmarks_mm": base_lms3,
                "reference_indices": cran_vs["indices"],
            })
            print(f"  Saved clinical_very_severe_plagio_P04.stl -> CI={cran_vs['indices']['CI']:.2f}%, CVA={cran_vs['indices']['CVA_mm']:.2f}mm, CVAI={cran_vs['indices']['CVAI']:.2f}%")

    print("[3/3] Creating defective & scale-calibration test STL files (Section 8.1 & 9.1)...")
    # 1. Defective incomplete scan (missing posterior-right occipital quadrant)
    good_stl_path = os.path.join(STL_DIR, "scan_081.stl")
    with open(good_stl_path, "rb") as f:
        mesh_good = parse_stl_bytes(f.read())
    centroids = (mesh_good.v1 + mesh_good.v2 + mesh_good.v3) / 3.0
    # Cut out posterior occipital sector (X < 10 and Y < 10)
    keep_mask = ~((centroids[:, 0] < 15.0) & (centroids[:, 1] < 15.0))
    defect_path = os.path.join(STL_DIR, "defect_incomplete_occipital.stl")
    write_binary_stl(
        defect_path,
        mesh_good.v1[keep_mask],
        mesh_good.v2[keep_mask],
        mesh_good.v3[keep_mask],
        header_text="Saigak-Orto Defective Incomplete Scan"
    )
    demo_metadata.append({
        "scan_id": "defect_incomplete_occipital",
        "filename": "defect_incomplete_occipital.stl",
        "description": "Дефектный неполный скан (отсутствует затылочно-теменной сектор — тест отказа)",
        "category": "defect_validation_demo",
    })

    # 2. Scale calibration test scan (coordinates stored in meters = 0.001x of mm)
    meters_path = os.path.join(STL_DIR, "scale_test_meters_scan082.stl")
    with open(os.path.join(STL_DIR, "scan_082.stl"), "rb") as f:
        mesh_82 = parse_stl_bytes(f.read())
    write_binary_stl(
        meters_path,
        mesh_82.v1 * 0.001,
        mesh_82.v2 * 0.001,
        mesh_82.v3 * 0.001,
        header_text="Saigak-Orto Meters Scale Scan (requires x1000)"
    )
    demo_metadata.append({
        "scan_id": "scale_test_meters_scan082",
        "filename": "scale_test_meters_scan082.stl",
        "description": "Скан в метрах (требует настройки масштаба ×1000 мм/м — п. 5.1 и 8.1 ТЗ)",
        "category": "scale_calibration_demo",
    })

    # Save JSON metadata
    with open(os.path.join(DATA_DIR, "ground_truth_landmarks.json"), "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)

    with open(os.path.join(DATA_DIR, "extracted_features_105.json"), "w", encoding="utf-8") as f:
        json.dump(features_cache, f)

    split_doc = {
        "dataset_name": "Saigak-Orto Anonymized Infant 3D Head Scans & Deformity Benchmark",
        "total_primary_stl_scans": 105,
        "one_scan_per_child": True,
        "augmentations_policy": "Аугментации (3D-повороты, сдвиги, масштабирование, шум) выполняются строго внутри обучающей выборки (train). Тестовая выборка изолирована.",
        "train_split": {
            "count": len(train_ids),
            "scan_ids": train_ids,
        },
        "reserved_test_split": {
            "count": len(test_ids),
            "scan_ids": test_ids,
        },
        "clinical_and_defect_demos": demo_metadata,
    }
    with open(os.path.join(DATA_DIR, "dataset_split.json"), "w", encoding="utf-8") as f:
        json.dump(split_doc, f, ensure_ascii=False, indent=2)

    print("Data preparation complete! Saved ground_truth_landmarks.json and dataset_split.json.")


if __name__ == "__main__":
    main()
