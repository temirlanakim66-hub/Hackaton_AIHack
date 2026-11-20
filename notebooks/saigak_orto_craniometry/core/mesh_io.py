"""
3D Mesh I/O, Scale Calibration, and Surface Integrity/Completeness Validator.
Supports Binary STL, ASCII STL, and Wavefront OBJ formats.
"""
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class QualityReport:
    status: str  # "VALID", "WARNING", "REJECTED"
    is_usable: bool
    quality_score: float
    triangle_count: int
    unique_vertex_count: int
    extents_mm: List[float]
    bounding_box_min: List[float]
    bounding_box_max: List[float]
    applied_scale: float
    suggested_scale: float
    degenerate_face_ratio: float
    quadrant_coverage_pct: float
    vault_closure_pct: float
    defects: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "status": self.status,
            "is_usable": self.is_usable,
            "quality_score": round(self.quality_score, 1),
            "triangle_count": self.triangle_count,
            "unique_vertex_count": self.unique_vertex_count,
            "extents_mm": [round(x, 2) for x in self.extents_mm],
            "bounding_box_min": [round(x, 2) for x in self.bounding_box_min],
            "bounding_box_max": [round(x, 2) for x in self.bounding_box_max],
            "applied_scale": self.applied_scale,
            "suggested_scale": self.suggested_scale,
            "degenerate_face_ratio": round(self.degenerate_face_ratio, 5),
            "quadrant_coverage_pct": round(self.quadrant_coverage_pct, 1),
            "vault_closure_pct": round(self.vault_closure_pct, 1),
            "defects": self.defects,
            "warnings": self.warnings,
        }


@dataclass
class MeshData:
    v1: np.ndarray  # (N, 3) float64
    v2: np.ndarray  # (N, 3) float64
    v3: np.ndarray  # (N, 3) float64
    normals: np.ndarray  # (N, 3) float64
    file_size_bytes: int
    format_type: str
    scale_factor: float = 1.0


def compute_face_normals(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    e1 = v2 - v1
    e2 = v3 - v1
    cross = np.cross(e1, e2)
    norms = np.linalg.norm(cross, axis=1)
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    unit_normals = cross / safe_norms[:, None]
    areas = 0.5 * norms
    return unit_normals, areas


def parse_stl_bytes(raw_bytes: bytes, scale_factor: float = 1.0) -> MeshData:
    size = len(raw_bytes)
    if size < 84:
        raise ValueError("Файл STL повреждён или слишком мал (< 84 байт).")

    n_tri = struct.unpack("<I", raw_bytes[80:84])[0]
    expected_bin_size = 84 + n_tri * 50

    if expected_bin_size == size and n_tri > 0:
        dt = np.dtype([
            ("normal", "<f4", (3,)),
            ("v1", "<f4", (3,)),
            ("v2", "<f4", (3,)),
            ("v3", "<f4", (3,)),
            ("attr", "<u2"),
        ])
        tris = np.frombuffer(raw_bytes, dtype=dt, offset=84, count=n_tri)
        v1 = tris["v1"].astype(np.float64) * scale_factor
        v2 = tris["v2"].astype(np.float64) * scale_factor
        v3 = tris["v3"].astype(np.float64) * scale_factor
        normals, _ = compute_face_normals(v1, v2, v3)
        return MeshData(
            v1=v1, v2=v2, v3=v3,
            normals=normals,
            file_size_bytes=size,
            format_type="Binary STL",
            scale_factor=scale_factor,
        )

    # Fallback: ASCII STL
    text = raw_bytes.decode("utf-8", errors="ignore")
    verts = []
    for line in text.splitlines():
        ls = line.strip()
        if ls.startswith("vertex "):
            parts = ls.split()
            if len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if len(verts) < 3 or len(verts) % 3 != 0:
        raise ValueError("Не удалось распознать корректную триангуляцию в STL файле.")

    arr = np.array(verts, dtype=np.float64) * scale_factor
    v1 = arr[0::3]
    v2 = arr[1::3]
    v3 = arr[2::3]
    normals, _ = compute_face_normals(v1, v2, v3)
    return MeshData(
        v1=v1, v2=v2, v3=v3,
        normals=normals,
        file_size_bytes=size,
        format_type="ASCII STL",
        scale_factor=scale_factor,
    )


def parse_obj_bytes(raw_bytes: bytes, scale_factor: float = 1.0) -> MeshData:
    text = raw_bytes.decode("utf-8", errors="ignore")
    verts = []
    faces = []
    for line in text.splitlines():
        if line.startswith("v "):
            parts = line.split()
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            parts = line.split()[1:]
            idx = [int(p.split("/")[0]) - 1 for p in parts]
            if len(idx) >= 3:
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])

    v = np.array(verts, dtype=np.float64) * scale_factor
    f = np.array(faces, dtype=np.int32)
    v1, v2, v3 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    normals, _ = compute_face_normals(v1, v2, v3)
    return MeshData(
        v1=v1, v2=v2, v3=v3,
        normals=normals,
        file_size_bytes=len(raw_bytes),
        format_type="Wavefront OBJ",
        scale_factor=scale_factor,
    )


def write_binary_stl(path: str, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, header_text: str = "Saigak-Orto Binary STL") -> None:
    n_tri = len(v1)
    normals, _ = compute_face_normals(v1, v2, v3)
    header = header_text.encode("ascii", errors="ignore")[:80].ljust(80, b" ")
    dt = np.dtype([
        ("normal", "<f4", (3,)),
        ("v1", "<f4", (3,)),
        ("v2", "<f4", (3,)),
        ("v3", "<f4", (3,)),
        ("attr", "<u2"),
    ])
    data = np.zeros(n_tri, dtype=dt)
    data["normal"] = normals.astype(np.float32)
    data["v1"] = v1.astype(np.float32)
    data["v2"] = v2.astype(np.float32)
    data["v3"] = v3.astype(np.float32)
    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", n_tri))
        f.write(data.tobytes())


def validate_mesh(mesh: MeshData) -> QualityReport:
    defects: List[str] = []
    warnings: List[str] = []
    n_tri = len(mesh.v1)

    if n_tri < 400:
        defects.append(f"Недостаточная плотность сетки: всего {n_tri} треугольников (минимум 400).")

    all_verts = np.vstack([mesh.v1, mesh.v2, mesh.v3])
    if not np.all(np.isfinite(all_verts)):
        defects.append("Обнаружены некорректные координаты вершин (NaN или Inf).")
        all_verts = np.nan_to_num(all_verts, nan=0.0)

    vmin = all_verts.min(axis=0)
    vmax = all_verts.max(axis=0)
    extents = vmax - vmin

    # Estimate unique vertices via rounded quantization
    quant = np.round(all_verts, 3)
    unique_v_count = len(np.unique(quant, axis=0))

    _, areas = compute_face_normals(mesh.v1, mesh.v2, mesh.v3)
    degen_ratio = float(np.mean(areas <= 1e-9))
    if degen_ratio > 0.05:
        warnings.append(f"Высокая доля вырожденных треугольников ({degen_ratio * 100:.1f}%).")

    # Suggest scale if dimensions look like meters or centimeters
    raw_mean_ext = float(np.mean(extents)) / max(mesh.scale_factor, 1e-9)
    if 0.07 <= raw_mean_ext <= 0.25:
        suggested_scale = 1000.0
    elif 7.0 <= raw_mean_ext <= 25.0:
        suggested_scale = 10.0
    elif 2.8 <= raw_mean_ext <= 9.0:
        suggested_scale = 25.4
    else:
        suggested_scale = 1.0

    # Check scaled dimensions for infant head (0-18 months: typically 80-210 mm)
    min_dim = float(np.min(extents))
    max_dim = float(np.max(extents))
    if max_dim < 70.0 or min_dim < 60.0:
        defects.append(
            f"Габариты модели ({extents[0]:.1f} × {extents[1]:.1f} × {extents[2]:.1f} мм) меньше допустимых анатомических границ головы младенца. Проверьте коэффициент масштаба (рекомендуется ×{suggested_scale:g})."
        )
    elif max_dim > 250.0:
        defects.append(
            f"Габариты модели ({extents[0]:.1f} × {extents[1]:.1f} × {extents[2]:.1f} мм) превышают анатомическую норму (0–18 мес). Проверьте масштаб."
        )

    # Check angular / azimuthal coverage in the cranial vault band (Z_rel in [0.40, 0.80])
    z_rel = (all_verts[:, 2] - vmin[2]) / max(extents[2], 1e-6)
    vault_mask = (z_rel >= 0.40) & (z_rel <= 0.80)
    vault_pts = all_verts[vault_mask]

    quadrant_coverage_pct = 0.0
    vault_closure_pct = 0.0
    if len(vault_pts) >= 100:
        center_xy = 0.5 * (vmin[:2] + vmax[:2])
        rel_xy = vault_pts[:, :2] - center_xy
        angles = np.arctan2(rel_xy[:, 1], rel_xy[:, 0])  # [-pi, pi]
        # 36 angular sectors of 10 degrees each
        hist, _ = np.histogram(angles, bins=36, range=(-np.pi, np.pi))
        occupied_sectors = int(np.sum(hist >= max(3, len(vault_pts) // 360)))
        vault_closure_pct = 100.0 * occupied_sectors / 36.0

        # 4 anatomical quadrants (FL, FR, BL, BR)
        q_counts = [
            np.sum((rel_xy[:, 0] >= 0) & (rel_xy[:, 1] >= 0)),
            np.sum((rel_xy[:, 0] >= 0) & (rel_xy[:, 1] < 0)),
            np.sum((rel_xy[:, 0] < 0) & (rel_xy[:, 1] >= 0)),
            np.sum((rel_xy[:, 0] < 0) & (rel_xy[:, 1] < 0)),
        ]
        min_q_ratio = min(q_counts) / max(1, len(vault_pts) / 4.0)
        quadrant_coverage_pct = min(100.0, 100.0 * min_q_ratio)

        if vault_closure_pct < 85.0:
            defects.append(
                f"Нарушена замкнутость свода черепа: заполнено лишь {vault_closure_pct:.1f}% угловых секторов сечения (крупная лакуна / отсутствующий фрагмент скана)."
            )
        elif vault_closure_pct < 95.0:
            warnings.append(
                f"Частичное снижение плотности полигонов в отдельных секторах свода ({vault_closure_pct:.1f}%)."
            )

        if quadrant_coverage_pct < 35.0:
            defects.append(
                f"Неполная поверхность скана: критический дефицит точек в одном из анатомических квадрантов ({quadrant_coverage_pct:.1f}% от нормы)."
            )
    else:
        defects.append("В области свода черепа отсутствуют полигоны для построения измерительного сечения.")

    # Compute overall quality score
    score = 100.0
    score -= max(0.0, 100.0 - vault_closure_pct) * 1.5
    score -= max(0.0, 80.0 - quadrant_coverage_pct) * 0.5
    score -= degen_ratio * 100.0
    if defects:
        score = min(score, 35.0)
    score = max(0.0, min(100.0, score))

    if defects:
        status = "REJECTED"
        is_usable = False
    elif warnings:
        status = "WARNING"
        is_usable = True
    else:
        status = "VALID"
        is_usable = True

    return QualityReport(
        status=status,
        is_usable=is_usable,
        quality_score=score,
        triangle_count=n_tri,
        unique_vertex_count=unique_v_count,
        extents_mm=extents.tolist(),
        bounding_box_min=vmin.tolist(),
        bounding_box_max=vmax.tolist(),
        applied_scale=mesh.scale_factor,
        suggested_scale=suggested_scale,
        degenerate_face_ratio=degen_ratio,
        quadrant_coverage_pct=quadrant_coverage_pct,
        vault_closure_pct=vault_closure_pct,
        defects=defects,
        warnings=warnings,
    )
