from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _normalize_u8(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32, copy=False)
    p1, p99 = np.percentile(arr, [1, 99])
    if p99 <= p1:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)


def save_overlay_png(
    *,
    ct_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    output_path: Path,
    z_index: int | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if z_index is None:
        counts = mask_zyx.reshape(mask_zyx.shape[0], -1).sum(axis=1)
        z_index = int(np.argmax(counts)) if counts.max() > 0 else ct_zyx.shape[0] // 2
    base = _normalize_u8(ct_zyx[z_index])
    rgb = np.stack([base, base, base], axis=-1)
    overlay = mask_zyx[z_index] > 0
    rgb[overlay, 0] = 255
    rgb[overlay, 1] = (rgb[overlay, 1] * 0.35).astype(np.uint8)
    rgb[overlay, 2] = (rgb[overlay, 2] * 0.35).astype(np.uint8)
    Image.fromarray(rgb).save(output_path)


def save_mip_png(mask_zyx: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mip = (mask_zyx.max(axis=0) > 0).astype(np.uint8) * 255
    Image.fromarray(mip).save(output_path)
