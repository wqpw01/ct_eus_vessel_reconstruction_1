from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from skimage import measure


def _continuous_index_xyz_to_lps(points_xyz: np.ndarray, reference: sitk.Image) -> np.ndarray:
    spacing = np.asarray(reference.GetSpacing(), dtype=float)
    origin = np.asarray(reference.GetOrigin(), dtype=float)
    direction = np.asarray(reference.GetDirection(), dtype=float).reshape(3, 3)
    return origin + (direction @ (points_xyz * spacing).T).T


def _lps_to_ras(points_lps: np.ndarray) -> np.ndarray:
    points_ras = points_lps.copy()
    points_ras[:, 0] *= -1.0
    points_ras[:, 1] *= -1.0
    return points_ras


def save_mask_mesh_ply(
    mask_zyx: np.ndarray,
    *,
    reference: sitk.Image,
    output_path: Path,
    coordinate_system: str = "slicer_ras",
) -> bool:
    if mask_zyx.sum() == 0:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    padded = np.pad(mask_zyx.astype(np.uint8), 1, mode="constant")
    verts, faces, _normals, _values = measure.marching_cubes(
        padded,
        level=0.5,
    )
    # marching_cubes returns continuous z,y,x coordinates on the padded array.
    # Subtract one voxel of padding, then convert xyz continuous index to
    # patient physical coordinates. Slicer model files are expected in RAS.
    verts_index_xyz = np.column_stack([verts[:, 2], verts[:, 1], verts[:, 0]]) - 1.0
    max_index_xyz = np.asarray(reference.GetSize(), dtype=float) - 1.0
    verts_index_xyz = np.clip(verts_index_xyz, 0.0, max_index_xyz)
    verts_lps = _continuous_index_xyz_to_lps(verts_index_xyz, reference)
    if coordinate_system == "slicer_ras":
        verts_out = _lps_to_ras(verts_lps)
    elif coordinate_system == "lps":
        verts_out = verts_lps
    else:
        raise ValueError(f"Unsupported mesh coordinate system: {coordinate_system}")
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(verts_out)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for vertex in verts_out:
            handle.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        for face in faces:
            handle.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    return True
