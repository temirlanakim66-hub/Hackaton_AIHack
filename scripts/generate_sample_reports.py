"""
Generates the mandatory sample PDF reports and CSV/JSON exports required by Section 9.1 of the TZ:
1. report_case1_good_scan.pdf       - Качественный скан из тестовой выборки (scan_081.stl) с рассчитанными индексами
2. report_case2_severe_deformity.pdf - Случай выраженной деформации (тяжёлая плагиоцефалия, clinical_severe_plagio_P03.stl)
3. report_case2b_very_severe.pdf    - Случай очень тяжёлой деформации (CVAI >= 11%, clinical_very_severe_plagio_P04.stl)
4. report_case3_rejected_defect.pdf - Отказ от расчёта при непригодных/неполных входных данных (defect_incomplete_occipital.stl)
"""
import json
import os
import joblib

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.mesh_io import parse_stl_bytes, validate_mesh
from core.craniometry import compute_craniometry
from core.report_generator import (
    generate_pdf_report_bytes,
    export_measurement_json,
    export_measurement_csv_string,
)


STL_DIR = os.path.join(PROJECT_ROOT, "data", "stl_scans")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "landmark_model.joblib")
    model = joblib.load(model_path)

    cases = [
        {
            "scan_id": "scan_081",
            "filename": "scan_081.stl",
            "pdf_name": "report_case1_good_scan.pdf",
            "age_months": 5.5,
            "doctor_name": "Золотухин А.П. (клинический эксперт)",
            "manual_correction": False,
            "notes": "Тестовая выборка (scan_081): качественный 3D-скан, асимметрия свода черепа ниже порога шкалы (< 3,5%).",
        },
        {
            "scan_id": "DEMO-SEVERE-PLAGIO-P03",
            "filename": "clinical_severe_plagio_P03.stl",
            "pdf_name": "report_case2_severe_deformity.pdf",
            "age_months": 6.0,
            "doctor_name": "Золотухин А.П. / Бойко А.",
            "manual_correction": True,
            "notes": "[Демонстрационный случай] Выраженная позиционная плагиоцефалия (тяжёлая степень по шкале CVAI 8,75–11%). Показана консультация ортезиста.",
        },
        {
            "scan_id": "DEMO-VERY-SEVERE-P04",
            "filename": "clinical_very_severe_plagio_P04.stl",
            "pdf_name": "report_case2b_very_severe.pdf",
            "age_months": 7.5,
            "doctor_name": "Золотухин А.П. / Бойко А.",
            "manual_correction": True,
            "notes": "[Демонстрационный случай] Очень тяжёлая позиционная плагиоцефалия (CVAI >= 11,0%). Выраженное уплощение затылочного квадранта.",
        },
        {
            "scan_id": "DEMO-DEFECT-INCOMPLETE-01",
            "filename": "defect_incomplete_occipital.stl",
            "pdf_name": "report_case3_rejected_defect.pdf",
            "age_months": 4.0,
            "doctor_name": "Бойко А. (ведущий инженер-ортезист)",
            "manual_correction": False,
            "notes": "[Демонстрационный случай отказа] Неполный 3D-скан с отсутствующим затылочным сектором свода. Требуется повторное 3D-сканирование.",
        },
    ]

    exported_records = []
    for c in cases:
        stl_path = os.path.join(STL_DIR, c["filename"])
        with open(stl_path, "rb") as f:
            mesh = parse_stl_bytes(f.read(), scale_factor=1.0)

        quality = validate_mesh(mesh)
        if quality.is_usable:
            lms = model.predict_mesh_landmarks(mesh)
            cran = compute_craniometry(mesh, lms, level_ratio=0.30)
        else:
            lms = None
            cran = None

        pdf_bytes = generate_pdf_report_bytes(
            scan_id=c["scan_id"],
            mesh=mesh,
            quality=quality,
            craniometry=cran,
            landmarks=lms,
            patient_age_months=c["age_months"],
            doctor_name=c["doctor_name"],
            doctor_verified=True,
            manual_correction_applied=c["manual_correction"],
            clinical_notes=c["notes"],
        )

        out_pdf = os.path.join(REPORTS_DIR, c["pdf_name"])
        with open(out_pdf, "wb") as f_pdf:
            f_pdf.write(pdf_bytes)

        rec = export_measurement_json(
            scan_id=c["scan_id"],
            quality=quality,
            craniometry=cran,
            landmarks=lms,
            patient_age_months=c["age_months"],
            doctor_name=c["doctor_name"],
            doctor_verified=True,
            manual_correction_applied=c["manual_correction"],
        )
        exported_records.append(rec)
        print(f"Generated {c['pdf_name']} ({len(pdf_bytes) // 1024} KB) | Status: {quality.status}")

    with open(os.path.join(REPORTS_DIR, "sample_exports.json"), "w", encoding="utf-8") as f_json:
        json.dump(exported_records, f_json, ensure_ascii=False, indent=2)

    csv_str = export_measurement_csv_string(exported_records)
    with open(os.path.join(REPORTS_DIR, "sample_exports.csv"), "w", encoding="utf-8", newline="") as f_csv:
        f_csv.write(csv_str)

    print("Saved sample_exports.json and sample_exports.csv in reports/")


if __name__ == "__main__":
    main()
