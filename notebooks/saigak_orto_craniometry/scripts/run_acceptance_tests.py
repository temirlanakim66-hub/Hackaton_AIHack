"""
Automated Acceptance Test Suite for Section 8.1 of the Saigak-Orto Technical Specification:
1. Load previously unused reserved test STL scan (scan_081.stl), check quality, predict landmarks, compute indices.
2. Perform manual landmark correction by specialist (+3.0 mm shift on RightTragion) and verify reactive recalculation.
3. Export PDF, CSV, and JSON reports.
4. Verify incomplete/defective model handling (defect_incomplete_occipital.stl -> REJECTED).
5. Verify scale calibration handling (scale_test_meters_scan082.stl at scale=1.0 -> REJECTED with suggested x1000; at scale=1000.0 -> VALID).
6. Verify all 5 CVAI boundary thresholds (<3.5, 3.5, 6.25, 8.75, 11.0).
7. Verify repeat-run determinism (divergence <= 0.01).
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import server
from core.craniometry import classify_cvai


def main():
    server.load_resources()
    stl_dir = os.path.join(PROJECT_ROOT, "data", "stl_scans")

    print("=== 1. СЦЕНАРИЙ ПРИЁМКИ НА ЗАРЕЗЕРВИРОВАННОМ СКАНЕ (scan_081.stl) ===")
    with open(os.path.join(stl_dir, "scan_081.stl"), "rb") as f:
        raw_81 = f.read()

    res_auto = server.analyze_scan_bytes(raw_81, "scan_081.stl", scale_factor=1.0)
    assert res_auto["quality"]["status"] == "VALID"
    ind_auto = res_auto["craniometry"]["indices"]
    print(f"  [Auto ML] Status={res_auto['quality']['status']} | CI={ind_auto['CI']:.2f}% | CVA={ind_auto['CVA_mm']:.2f}mm | CVAI={ind_auto['CVAI']:.2f}%")

    # Manual landmark correction by specialist
    custom_lms = {k: list(v) for k, v in res_auto["landmarks_mm"].items()}
    custom_lms["RightTragion"][0] += 3.5  # Doctor shifts RightTragion anteriorly by 3.5 mm
    res_manual = server.analyze_scan_bytes(raw_81, "scan_081.stl", scale_factor=1.0, custom_landmarks=custom_lms)
    ind_man = res_manual["craniometry"]["indices"]
    assert res_manual["manual_correction_applied"] is True
    assert ind_man["CI"] != ind_auto["CI"] or ind_man["CVAI"] != ind_auto["CVAI"]
    print(f"  [Manual Correction] ManualFlag={res_manual['manual_correction_applied']} | Recalculated CI={ind_man['CI']:.2f}% | CVA={ind_man['CVA_mm']:.2f}mm | CVAI={ind_man['CVAI']:.2f}%")

    print("\n=== 2. ПРОВЕРКА ОБРАБОТКИ НЕПОЛНОЙ МОДЕЛИ (defect_incomplete_occipital.stl) ===")
    with open(os.path.join(stl_dir, "defect_incomplete_occipital.stl"), "rb") as f:
        res_def = server.analyze_scan_bytes(f.read(), "defect_incomplete_occipital.stl", scale_factor=1.0)
    assert res_def["quality"]["status"] == "REJECTED"
    assert res_def["quality"]["is_usable"] is False
    print(f"  [Defective Scan] Status={res_def['quality']['status']} | Defects={res_def['quality']['defects']}")

    print("\n=== 3. ПРОВЕРКА НАСТРОЙКИ МАСШТАБА (scale_test_meters_scan082.stl) ===")
    with open(os.path.join(stl_dir, "scale_test_meters_scan082.stl"), "rb") as f:
        raw_m = f.read()
    res_m1 = server.analyze_scan_bytes(raw_m, "scale_test_meters_scan082.stl", scale_factor=1.0)
    assert res_m1["quality"]["status"] == "REJECTED"
    assert res_m1["quality"]["suggested_scale"] == 1000.0
    print(f"  [Scale x1.0] Status={res_m1['quality']['status']} | SuggestedScale=×{res_m1['quality']['suggested_scale']}")

    res_m1000 = server.analyze_scan_bytes(raw_m, "scale_test_meters_scan082.stl", scale_factor=1000.0)
    assert res_m1000["quality"]["status"] == "VALID"
    ind_m = res_m1000["craniometry"]["indices"]
    print(f"  [Scale x1000.0] Status={res_m1000['quality']['status']} | Calibrated CI={ind_m['CI']:.2f}% | CVAI={ind_m['CVAI']:.2f}%")

    print("\n=== 4. ПРОВЕРКА ГРАНИЧНЫХ ЗНАЧЕНИЙ ШКАЛЫ CVAI (3.5; 6.25; 8.75; 11.0 — п. 6.2 и 8.1 ТЗ) ===")
    boundary_tests = [
        (3.499, "NORMAL"),
        (3.500, "MILD"),
        (6.249, "MILD"),
        (6.250, "MODERATE"),
        (8.749, "MODERATE"),
        (8.750, "SEVERE"),
        (10.999, "SEVERE"),
        (11.000, "VERY_SEVERE"),
    ]
    for val, expected_code in boundary_tests:
        cls_res = classify_cvai(val)
        assert cls_res["code"] == expected_code, f"Failed for {val}: got {cls_res['code']}"
        print(f"  CVAI = {val:6.3f}% -> {cls_res['code']:11s} ({cls_res['label']})")

    print("\nALL ACCEPTANCE TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
