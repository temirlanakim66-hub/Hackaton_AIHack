"""
3D Mesh I/O, Scale Calibration, and Surface Integrity/Completeness Validator.
Supports Binary STL, ASCII STL, and Wavefront OBJ formats.
Includes:
- Anatomical defect sector localization (for 3D red highlighting & technician instructions)
- Floating polygon / disconnected island artifact detection
- Automatic unit/scale mismatch diagnosis (mm vs cm vs m vs inches)
"""
import struct
from collections import deque
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
    defect_zones: List[Dict] = field(default_factory=list)
    technician_instructions: List[str] = field(default_factory=list)
    per_quadrant_coverage: Dict[str, float] = field(default_factory=dict)
    floating_triangles_count: int = 0
    floating_triangles_pct: float = 0.0
    disconnected_components_count: int = 1
    unit_diagnosis: str = "Миллиметры (мм) — масштаб соответствует норме"

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
            "defect_zones": self.defect_zones,
            "technician_instructions": self.technician_instructions,
            "per_quadrant_coverage": self.per_quadrant_coverage,
            "floating_triangles_count": self.floating_triangles_count,
            "floating_triangles_pct": round(self.floating_triangles_pct, 3),
            "disconnected_components_count": self.disconnected_components_count,
            "unit_diagnosis": self.unit_diagnosis,
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


def _describe_anatomical_sector(angle_mid_deg: float) -> Tuple[str, str]:
    """
    Map angular sector in STL coordinates (+X anterior/face, -X posterior/occiput,
    +Y left ear, -Y right ear) to Russian clinical anatomy and technician rescan instruction.
    """
    a = ((angle_mid_deg + 180.0) % 360.0) - 180.0
    if -180.0 <= a < -90.0:
        return (
            "правой затылочной области",
            "Не хватает данных в правой затылочной области — требуется повторное сканирование задне-правой поверхности головы (убедитесь, что рука или опора не перекрывают затылок).",
        )
    if 90.0 <= a <= 180.0:
        return (
            "левой затылочной области",
            "Не хватает данных в левой затылочной области — требуется повторное сканирование задне-левой поверхности головы.",
        )
    if -90.0 <= a < 0.0:
        return (
            "правой лобно-височной области",
            "Не хватает данных в правой лобно-височной области — выполните дополнительный проход сканером со стороны правого виска и лба.",
        )
    return (
        "левой лобно-височной области",
        "Не хватает данных в левой лобно-височной области — выполните дополнительный проход сканером со стороны левого виска и лба.",
    )


def _detect_floating_islands(centroids: np.ndarray, extents: np.ndarray) -> Tuple[int, int, float]:
    """
    Fast voxel-based spatial connected-component check to detect floating noise polygons
    disconnected from the main cranial surface (which could bias bounding box or centering).
    Returns: (num_components, floating_tri_count, floating_tri_pct)
    """
    n_tri = len(centroids)
    if n_tri < 20:
        return 1, 0, 0.0

    # Voxel size ~ 6% of max dimension (typically ~8-9 mm on an infant head)
    voxel_size = max(float(np.max(extents)) * 0.06, 1e-4)
    vmin = centroids.min(axis=0)
    ijk = np.floor((centroids - vmin) / voxel_size).astype(np.int32)

    # Map each occupied voxel to triangle count
    uniq_vox, inverse, counts = np.unique(ijk, axis=0, return_inverse=True, return_counts=True)
    vox_to_idx = {tuple(v): idx for idx, v in enumerate(uniq_vox.tolist())}
    n_vox = len(uniq_vox)

    visited = np.zeros(n_vox, dtype=bool)
    comp_tri_counts: List[int] = []

    # 26-connected neighbor offsets
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    for start in range(n_vox):
        if visited[start]:
            continue
        q = deque([start])
        visited[start] = True
        tri_sum = 0
        while q:
            cur = q.popleft()
            tri_sum += int(counts[cur])
            cx, cy, cz = uniq_vox[cur]
            for dx, dy, dz in offsets:
                nb = vox_to_idx.get((int(cx + dx), int(cy + dy), int(cz + dz)))
                if nb is not None and not visited[nb]:
                    visited[nb] = True
                    q.append(nb)
        comp_tri_counts.append(tri_sum)

    if len(comp_tri_counts) <= 1:
        return 1, 0, 0.0

    largest = max(comp_tri_counts)
    floating_count = int(n_tri - largest)
    floating_pct = 100.0 * floating_count / max(1, n_tri)
    return len(comp_tri_counts), floating_count, floating_pct


def validate_mesh(mesh: MeshData) -> QualityReport:
    defects: List[str] = []
    warnings: List[str] = []
    defect_zones: List[Dict] = []
    technician_instructions: List[str] = []
    n_tri = len(mesh.v1)

    if n_tri < 400:
        defects.append(f"Недостаточная плотность сетки: всего {n_tri} треугольников (минимум 400).")
        technician_instructions.append("Повторите 3D-сканирование с более высоким разрешением сетки.")

    all_verts = np.vstack([mesh.v1, mesh.v2, mesh.v3])
    if not np.all(np.isfinite(all_verts)):
        defects.append("Обнаружены некорректные координаты вершин (NaN или Inf).")
        technician_instructions.append("Повторно экспортируйте STL из программы 3D-сканера — исходный файл повреждён.")
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

    # Detect floating / disconnected noise polygons
    centroids = (mesh.v1 + mesh.v2 + mesh.v3) / 3.0
    n_components, floating_count, floating_pct = _detect_floating_islands(centroids, extents)
    if floating_count > 0:
        # Check whether floating triangles distort the bounding box significantly
        if floating_pct >= 0.5 or floating_count >= 50:
            warnings.append(
                f"Обнаружены «летающие» полигоны / шум вне основного контура: {floating_count} тр. ({floating_pct:.2f}%, отдельных фрагментов: {n_components})."
            )
            technician_instructions.append(
                "Обнаружены отдельные «летающие» фрагменты сетки (фон/рука). Рекомендуется обрезать лишний фон в ПО сканера для максимально точного центрирования."
            )

    # Suggest scale if dimensions look like meters, centimeters, or inches
    raw_mean_ext = float(np.mean(extents)) / max(mesh.scale_factor, 1e-9)
    unit_diagnosis = "Миллиметры (мм) — масштаб соответствует анатомической норме"
    if 0.07 <= raw_mean_ext <= 0.25:
        suggested_scale = 1000.0
        if abs(mesh.scale_factor - 1000.0) > 1e-3:
            unit_diagnosis = "Единицы файла похожи на метры (м) вместо миллиметров — требуется масштаб ×1000"
    elif 7.0 <= raw_mean_ext <= 25.0:
        suggested_scale = 10.0
        if abs(mesh.scale_factor - 10.0) > 1e-3:
            unit_diagnosis = "Единицы файла похожи на сантиметры (см) вместо миллиметров — требуется масштаб ×10"
    elif 2.8 <= raw_mean_ext <= 9.0:
        suggested_scale = 25.4
        if abs(mesh.scale_factor - 25.4) > 1e-3:
            unit_diagnosis = "Единицы файла похожи на дюймы (in) вместо миллиметров — требуется масштаб ×25.4"
    else:
        suggested_scale = 1.0

    # Check scaled dimensions for infant head (0-18 months: typically 80-210 mm)
    min_dim = float(np.min(extents))
    max_dim = float(np.max(extents))
    if max_dim < 70.0 or min_dim < 60.0:
        defects.append(
            f"Габариты модели ({extents[0]:.1f} × {extents[1]:.1f} × {extents[2]:.1f} мм) меньше допустимых анатомических границ головы младенца. Проверьте коэффициент масштаба (рекомендуется ×{suggested_scale:g})."
        )
        technician_instructions.append(
            f"Перепутаны единицы измерения STL ({unit_diagnosis}). Нажмите «Применить масштаб ×{suggested_scale:g}» в левой панели."
        )
    elif max_dim > 250.0:
        defects.append(
            f"Габариты модели ({extents[0]:.1f} × {extents[1]:.1f} × {extents[2]:.1f} мм) превышают анатомическую норму (0–18 мес). Проверьте масштаб или наличие «летающих» полигонов."
        )
        technician_instructions.append(
            "Проверьте масштаб STL-файла и убедитесь, что в сцену не попали лишние объекты (стол, плечи, шум сканера)."
        )

    # Check angular / azimuthal coverage in the cranial vault band (Z_rel in [0.40, 0.80])
    z_rel = (all_verts[:, 2] - vmin[2]) / max(extents[2], 1e-6)
    vault_mask = (z_rel >= 0.40) & (z_rel <= 0.80)
    vault_pts = all_verts[vault_mask]

    quadrant_coverage_pct = 0.0
    vault_closure_pct = 0.0
    per_quadrant_coverage = {
        "front_left": 0.0,
        "front_right": 0.0,
        "back_left": 0.0,
        "back_right": 0.0,
    }
    if len(vault_pts) >= 100:
        center_xy = 0.5 * (vmin[:2] + vmax[:2])
        rel_xy = vault_pts[:, :2] - center_xy
        angles = np.arctan2(rel_xy[:, 1], rel_xy[:, 0])  # [-pi, pi], 0=+X(front), +90=+Y(left), -90=-Y(right), 180=-X(back)
        # 36 angular sectors of 10 degrees each
        hist, bin_edges = np.histogram(angles, bins=36, range=(-np.pi, np.pi))
        min_sector_pts = max(3, len(vault_pts) // 360)
        occupied_mask = hist >= min_sector_pts
        occupied_sectors = int(np.sum(occupied_mask))
        vault_closure_pct = 100.0 * occupied_sectors / 36.0

        # Identify contiguous missing sectors (>= 20 degrees) for 3D red highlighting
        empty_indices = [i for i in range(36) if not occupied_mask[i]]
        if empty_indices:
            groups: List[List[int]] = []
            cur_grp = [empty_indices[0]]
            for idx in empty_indices[1:]:
                if idx == cur_grp[-1] + 1:
                    cur_grp.append(idx)
                else:
                    groups.append(cur_grp)
                    cur_grp = [idx]
            groups.append(cur_grp)
            # Merge wrap-around at -180/+180
            if len(groups) > 1 and groups[0][0] == 0 and groups[-1][-1] == 35:
                groups[0] = groups[-1] + groups[0]
                groups.pop()

            mean_r = float(np.median(np.linalg.norm(rel_xy, axis=1)))
            z_band_lo = float(vmin[2] + 0.42 * extents[2])
            z_band_hi = float(vmin[2] + 0.80 * extents[2])
            z_band_mid = 0.5 * (z_band_lo + z_band_hi)

            for grp in groups:
                if len(grp) < 2:
                    continue
                deg_start = -180.0 + grp[0] * 10.0
                deg_span = len(grp) * 10.0
                deg_end = deg_start + deg_span
                deg_mid = deg_start + 0.5 * deg_span
                region_name, instr = _describe_anatomical_sector(deg_mid)

                # Build 3D patch boundary points for direct red overlay in 3D viewer
                arc_angles = np.deg2rad(np.linspace(deg_start, deg_end, max(8, len(grp) * 2)))
                arc_low = [
                    [round(float(center_xy[0] + mean_r * np.cos(a)), 2),
                     round(float(center_xy[1] + mean_r * np.sin(a)), 2),
                     round(z_band_lo, 2)]
                    for a in arc_angles
                ]
                arc_high = [
                    [round(float(center_xy[0] + mean_r * np.cos(a)), 2),
                     round(float(center_xy[1] + mean_r * np.sin(a)), 2),
                     round(z_band_hi, 2)]
                    for a in arc_angles
                ]
                rad_mid = np.deg2rad(deg_mid)
                label_3d = [
                    round(float(center_xy[0] + mean_r * 1.05 * np.cos(rad_mid)), 2),
                    round(float(center_xy[1] + mean_r * 1.05 * np.sin(rad_mid)), 2),
                    round(z_band_mid, 2),
                ]

                defect_zones.append({
                    "region_name": region_name,
                    "angle_start_deg": round(deg_start, 1),
                    "angle_end_deg": round(deg_end, 1),
                    "missing_span_deg": round(deg_span, 1),
                    "center_xy": [round(float(center_xy[0]), 2), round(float(center_xy[1]), 2)],
                    "radius_mm": round(mean_r, 2),
                    "z_min_mm": round(z_band_lo, 2),
                    "z_max_mm": round(z_band_hi, 2),
                    "arc_low_3d": arc_low,
                    "arc_high_3d": arc_high,
                    "label_anchor_3d": label_3d,
                    "technician_instruction": instr,
                })
                if instr not in technician_instructions:
                    technician_instructions.append(instr)

        # 4 anatomical quadrants (FL: +X,+Y; FR: +X,-Y; BL: -X,+Y; BR: -X,-Y)
        expected_q = max(1.0, len(vault_pts) / 4.0)
        q_fl = int(np.sum((rel_xy[:, 0] >= 0) & (rel_xy[:, 1] >= 0)))
        q_fr = int(np.sum((rel_xy[:, 0] >= 0) & (rel_xy[:, 1] < 0)))
        q_bl = int(np.sum((rel_xy[:, 0] < 0) & (rel_xy[:, 1] >= 0)))
        q_br = int(np.sum((rel_xy[:, 0] < 0) & (rel_xy[:, 1] < 0)))
        per_quadrant_coverage = {
            "front_left": round(min(100.0, 100.0 * q_fl / expected_q), 1),
            "front_right": round(min(100.0, 100.0 * q_fr / expected_q), 1),
            "back_left": round(min(100.0, 100.0 * q_bl / expected_q), 1),
            "back_right": round(min(100.0, 100.0 * q_br / expected_q), 1),
        }
        min_q_ratio = min(q_fl, q_fr, q_bl, q_br) / expected_q
        quadrant_coverage_pct = min(100.0, 100.0 * min_q_ratio)

        if vault_closure_pct < 85.0:
            zone_desc = f" ({', '.join(z['region_name'] for z in defect_zones)})" if defect_zones else ""
            defects.append(
                f"Нарушена замкнутость свода черепа{zone_desc}: заполнено лишь {vault_closure_pct:.1f}% угловых секторов сечения (крупная лакуна / отсутствующий фрагмент скана)."
            )
        elif vault_closure_pct < 95.0:
            warnings.append(
                f"Частичное снижение плотности полигонов в отдельных секторах свода ({vault_closure_pct:.1f}%)."
            )

        if quadrant_coverage_pct < 35.0:
            worst_q_key = min(per_quadrant_coverage, key=per_quadrant_coverage.get)
            q_labels_ru = {
                "front_left": "левом лобном квадранте",
                "front_right": "правом лобном квадранте",
                "back_left": "левом затылочном квадранте",
                "back_right": "правом затылочном квадранте",
            }
            defects.append(
                f"Неполная поверхность скана: критический дефицит точек в {q_labels_ru[worst_q_key]} ({quadrant_coverage_pct:.1f}% от нормы)."
            )
    else:
        defects.append("В области свода черепа отсутствуют полигоны для построения измерительного сечения.")
        technician_instructions.append("Повторите 3D-сканирование: свод головы не попал в область захвата.")

    # Compute overall quality score
    score = 100.0
    score -= max(0.0, 100.0 - vault_closure_pct) * 1.5
    score -= max(0.0, 80.0 - quadrant_coverage_pct) * 0.5
    score -= degen_ratio * 100.0
    score -= min(10.0, floating_pct * 2.0)
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
        defect_zones=defect_zones,
        technician_instructions=technician_instructions,
        per_quadrant_coverage=per_quadrant_coverage,
        floating_triangles_count=floating_count,
        floating_triangles_pct=floating_pct,
        disconnected_components_count=n_components,
        unit_diagnosis=unit_diagnosis,
    )
