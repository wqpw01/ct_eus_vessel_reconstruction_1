from ct_eus_vessel.series import SeriesInfo, filter_candidate_series, sort_series_for_phase_analysis


def test_filter_candidate_series_keeps_only_thin_soft_diagnostic_ct() -> None:
    series = [
        SeriesInfo(series_uid="a", n_slices=227, protocol_name="Arterial Phase", series_description="1.0 x 1.0_A", convolution_kernel="B_SOFT_B", slice_thickness_mm=1.0, acquisition_time="100"),
        SeriesInfo(series_uid="b", n_slices=2, protocol_name="Scout", series_description="", convolution_kernel="", slice_thickness_mm=None, acquisition_time="090"),
        SeriesInfo(series_uid="c", n_slices=58, protocol_name="Chest Helical", series_description="5.0 x 5.0_lung", convolution_kernel="B_SHARP_C", slice_thickness_mm=5.0, acquisition_time="110"),
        SeriesInfo(series_uid="d", n_slices=257, protocol_name="Portal Phase", series_description="1.0 x 1.0_A", convolution_kernel="B_SOFT_B", slice_thickness_mm=1.0, acquisition_time="120"),
    ]

    candidates = filter_candidate_series(series)

    assert [item.series_uid for item in candidates] == ["a", "d"]


def test_sort_series_for_phase_analysis_uses_acquisition_time_then_uid() -> None:
    series = [
        SeriesInfo(series_uid="b", n_slices=200, acquisition_time="101.0"),
        SeriesInfo(series_uid="a", n_slices=200, acquisition_time="101.0"),
        SeriesInfo(series_uid="c", n_slices=200, acquisition_time="099.0"),
    ]

    ordered = sort_series_for_phase_analysis(series)

    assert [item.series_uid for item in ordered] == ["c", "a", "b"]
