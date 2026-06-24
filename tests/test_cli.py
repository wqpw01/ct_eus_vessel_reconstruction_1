import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    return env


def test_cli_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ct_eus_vessel.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=_env(),
    )

    assert result.returncode == 0
    assert "ct-eus-vessel" in result.stdout


def test_cli_index_writes_json_for_empty_directory(tmp_path: Path) -> None:
    output = tmp_path / "series.json"
    result = subprocess.run(
        [sys.executable, "-m", "ct_eus_vessel.cli", "index", "--input", str(tmp_path), "--json", str(output)],
        check=False,
        capture_output=True,
        text=True,
        env=_env(),
    )

    assert result.returncode == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["series"] == []
    assert payload["candidate_series"] == []


def _write_test_image(array_zyx: np.ndarray, path: Path) -> None:
    image = sitk.GetImageFromArray(array_zyx)
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))


def test_cli_compare_writes_metrics_json(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    output = tmp_path / "comparison.json"
    ref_mask = np.array([[[1, 0], [0, 2]]], dtype=np.uint8)
    cand_mask = np.array([[[1, 0], [0, 0]]], dtype=np.uint8)

    _write_test_image(np.zeros_like(ref_mask, dtype=np.int16), reference / "reference_ct.nrrd")
    _write_test_image(ref_mask, reference / "vessel_fused_multilabel.nrrd")
    _write_test_image(np.zeros_like(cand_mask, dtype=np.int16), candidate / "reference_ct.nrrd")
    _write_test_image(cand_mask, candidate / "vessel_fused_multilabel.nrrd")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ct_eus_vessel.cli",
            "compare",
            "--reference-output",
            str(reference),
            "--candidate-output",
            str(candidate),
            "--json",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_env(),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["overall"]["reference_voxels"] == 2
    assert payload["overall"]["candidate_voxels"] == 1
    assert payload["overall"]["recall"] == 0.5
