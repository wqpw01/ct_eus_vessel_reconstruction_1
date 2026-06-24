from __future__ import annotations

import argparse
import json
from pathlib import Path

from ct_eus_vessel.config import load_config
from ct_eus_vessel.dicom_index import index_dicom_series
from ct_eus_vessel.evaluation import compare_output_masks
from ct_eus_vessel.pipeline import run_pipeline
from ct_eus_vessel.serialization import to_jsonable
from ct_eus_vessel.series import filter_candidate_series, sort_series_for_phase_analysis


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_index(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    series = index_dicom_series(args.input)
    selection = config["series_selection"]
    candidates = sort_series_for_phase_analysis(
        filter_candidate_series(
            series,
            min_slices=selection["min_slices"],
            max_slice_thickness_mm=selection["max_slice_thickness_mm"],
            soft_kernel_keywords=selection["soft_kernel_keywords"],
            excluded_protocol_keywords=selection["excluded_protocol_keywords"],
        )
    )
    payload = {"series": series, "candidate_series": candidates}
    if args.json:
        _write_json(args.json, payload)
    else:
        print(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    summary = run_pipeline(
        input_path=args.input,
        output_dir=args.output,
        label_path=args.label,
        config_path=args.config,
        skip_frangi=args.skip_frangi,
        skip_mesh=args.skip_mesh,
        vesselness_mode=args.vesselness_mode,
        max_series=args.max_series,
        force_totalseg=args.force_totalseg,
        totalseg_device=args.totalseg_device,
    )
    compact = {
        "output_dir": summary["output_dir"],
        "label": summary["label"],
        "guidance_source": summary["guidance_source"],
        "phase_selection_source": summary["phase_selection_source"],
        "phase_mapping": summary["phase_mapping"],
        "hu_windows": summary["hu_windows"],
        "voxel_counts": summary["voxel_counts"],
        "bbox": summary["bbox"],
        "totalseg_cache_path": summary["totalseg_cache_path"],
        "metrics_report": Path(summary["output_dir"]) / "metrics_report.json",
    }
    print(json.dumps(to_jsonable(compact), ensure_ascii=False, indent=2))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    payload = compare_output_masks(
        reference_dir=args.reference_output,
        candidate_dir=args.candidate_output,
    )
    if args.json:
        _write_json(args.json, payload)
    else:
        print(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ct-eus-vessel")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser("index", help="Index DICOM series and phase-analysis candidates.")
    index.add_argument("--input", required=True, type=Path)
    index.add_argument("--config", type=Path)
    index.add_argument("--json", type=Path)
    index.set_defaults(func=cmd_index)

    run = subparsers.add_parser("run", help="Run multi-phase vessel extraction.")
    run.add_argument("--input", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--label", type=Path, help="Weak multilabel reference NIfTI.")
    run.add_argument("--config", type=Path)
    run.add_argument("--skip-frangi", action="store_true", help="Use HU-derived vesselness for fast smoke tests.")
    run.add_argument("--vesselness-mode", choices=["hu", "slice-frangi", "frangi3d"], help="Override vesselness backend.")
    run.add_argument("--skip-mesh", action="store_true", help="Skip PLY mesh generation for faster iteration.")
    run.add_argument("--max-series", type=int, help="Limit candidate series for debugging.")
    run.add_argument("--force-totalseg", action="store_true", help="Re-run TotalSegmentator even if cached output exists.")
    run.add_argument("--totalseg-device", help="Override TotalSegmentator device, e.g. gpu, gpu:0, cpu, or mps.")
    run.set_defaults(func=cmd_run)

    compare = subparsers.add_parser("compare", help="Compare a candidate output directory against a reference output directory.")
    compare.add_argument("--reference-output", required=True, type=Path)
    compare.add_argument("--candidate-output", required=True, type=Path)
    compare.add_argument("--json", type=Path)
    compare.set_defaults(func=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
