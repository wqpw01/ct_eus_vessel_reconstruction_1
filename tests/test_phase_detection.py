from ct_eus_vessel.phase import PhaseScores, choose_phase_series, choose_phase_series_from_metadata
from ct_eus_vessel.series import SeriesInfo


def test_choose_phase_series_uses_vascular_roi_brightness_not_series_text() -> None:
    scores = [
        PhaseScores(series_uid="late", aorta=136, celiac_artery=88, portal_vein=129, ivc=126, liver_vein=134),
        PhaseScores(series_uid="arterial", aorta=423, celiac_artery=284, portal_vein=66, ivc=71, liver_vein=58),
        PhaseScores(series_uid="portal", aorta=164, celiac_artery=127, portal_vein=192, ivc=148, liver_vein=146),
        PhaseScores(series_uid="hepatic_venous", aorta=157, celiac_artery=126, portal_vein=164, ivc=139, liver_vein=174),
    ]

    mapping = choose_phase_series(scores)

    assert mapping.arterial_uid == "arterial"
    assert mapping.portal_uid == "portal"
    assert mapping.venous_uid == "hepatic_venous"
    assert mapping.support_uids == ["late"]


def test_choose_phase_series_from_metadata_matches_ct0021_order() -> None:
    candidates = [
        SeriesInfo(
            series_uid="1.2.3.172466",
            n_slices=227,
            protocol_name="Arterial Phase",
            series_description="1.0 x 1.0_A",
            acquisition_time="163714.396",
        ),
        SeriesInfo(
            series_uid="1.2.3.182466",
            n_slices=227,
            protocol_name="Portal Phase",
            series_description="1.0 x 1.0_A",
            acquisition_time="163738.396",
        ),
        SeriesInfo(
            series_uid="1.2.3.14300",
            n_slices=257,
            protocol_name="Portal Phase",
            series_description="1.0 x 1.0_A",
            acquisition_time="163803.145",
        ),
        SeriesInfo(
            series_uid="1.2.3.12740",
            n_slices=257,
            protocol_name="Portal Phase",
            series_description="1.0 x 1.0_A",
            acquisition_time="163821.336",
        ),
    ]

    mapping = choose_phase_series_from_metadata(candidates)

    assert mapping.arterial_uid.endswith("172466")
    assert mapping.portal_uid.endswith("182466")
    assert mapping.venous_uid.endswith("14300")
    assert mapping.support_uids == ["1.2.3.12740"]
