from pathlib import Path

import numpy as np
import SimpleITK as sitk

from ct_eus_vessel.mesh import save_mask_mesh_ply


def _read_ply_vertices(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    vertex_count = 0
    header_end = 0
    for index, line in enumerate(lines):
        if line.startswith("element vertex "):
            vertex_count = int(line.split()[-1])
        if line == "end_header":
            header_end = index + 1
            break
    vertices = []
    for line in lines[header_end : header_end + vertex_count]:
        vertices.append([float(value) for value in line.split()[:3]])
    return np.asarray(vertices, dtype=float)


def test_save_mask_mesh_ply_writes_slicer_ras_physical_coordinates(tmp_path: Path) -> None:
    mask = np.zeros((3, 4, 5), dtype=bool)
    mask[1, 1:3, 2:4] = True
    reference = sitk.Image([5, 4, 3], sitk.sitkUInt8)
    reference.SetSpacing((2.0, 3.0, 4.0))
    reference.SetOrigin((100.0, 200.0, 300.0))
    reference.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    out = tmp_path / "mesh.ply"

    assert save_mask_mesh_ply(mask, reference=reference, output_path=out, coordinate_system="slicer_ras")

    vertices = _read_ply_vertices(out)
    assert vertices[:, 0].max() < -100.0
    assert vertices[:, 1].max() < -200.0
    assert vertices[:, 2].min() > 300.0


def test_save_mask_mesh_ply_keeps_boundary_vertices_inside_reference_bounds(tmp_path: Path) -> None:
    mask = np.zeros((2, 2, 2), dtype=bool)
    mask[1, 0, 0] = True
    reference = sitk.Image([2, 2, 2], sitk.sitkUInt8)
    reference.SetSpacing((1.0, 1.0, 1.0))
    reference.SetOrigin((0.0, 0.0, 0.0))
    reference.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    out = tmp_path / "mesh.ply"

    assert save_mask_mesh_ply(mask, reference=reference, output_path=out, coordinate_system="lps")

    vertices = _read_ply_vertices(out)
    assert vertices[:, 0].min() >= 0.0
    assert vertices[:, 1].min() >= 0.0
    assert vertices[:, 2].max() <= 1.0
