from __future__ import annotations

from dataclasses import dataclass

from ct_eus_vessel.series import SeriesInfo


@dataclass(frozen=True)
class PhaseScores:
    series_uid: str
    aorta: float | None = None
    celiac_artery: float | None = None
    portal_vein: float | None = None
    ivc: float | None = None
    liver_vein: float | None = None


@dataclass(frozen=True)
class PhaseMapping:
    arterial_uid: str
    portal_uid: str
    venous_uid: str
    support_uids: list[str]


def _score(values: list[float | None]) -> float:
    valid = [value for value in values if value is not None]
    if not valid:
        return float("-inf")
    return sum(valid) / len(valid)


def _pick_best(scores: list[PhaseScores], used: set[str], attrs: list[str]) -> str:
    remaining = [item for item in scores if item.series_uid not in used]
    if not remaining:
        raise ValueError("Not enough series to assign all requested vascular phases")
    return max(remaining, key=lambda item: (_score([getattr(item, attr) for attr in attrs]), item.series_uid)).series_uid


def choose_phase_series(scores: list[PhaseScores]) -> PhaseMapping:
    if len(scores) < 3:
        raise ValueError("At least three contrast series are required for arterial, portal, and venous phase mapping")
    used: set[str] = set()
    arterial_uid = _pick_best(scores, used, ["aorta", "celiac_artery"])
    used.add(arterial_uid)
    portal_uid = _pick_best(scores, used, ["portal_vein"])
    used.add(portal_uid)
    venous_uid = _pick_best(scores, used, ["liver_vein", "ivc"])
    used.add(venous_uid)
    support_uids = [item.series_uid for item in scores if item.series_uid not in used]
    return PhaseMapping(
        arterial_uid=arterial_uid,
        portal_uid=portal_uid,
        venous_uid=venous_uid,
        support_uids=support_uids,
    )


def _metadata_text(series: SeriesInfo) -> str:
    return " ".join(
        [
            series.protocol_name,
            series.series_description,
            series.convolution_kernel,
        ]
    ).casefold()


def _find_by_keywords(
    candidates: list[SeriesInfo],
    *,
    used: set[str],
    keywords: tuple[str, ...],
    start_after: int = -1,
) -> SeriesInfo | None:
    for index, item in enumerate(candidates):
        if index <= start_after or item.series_uid in used:
            continue
        text = _metadata_text(item)
        if any(keyword in text for keyword in keywords):
            return item
    return None


def _first_unused(candidates: list[SeriesInfo], *, used: set[str], start_after: int = -1) -> SeriesInfo | None:
    for index, item in enumerate(candidates):
        if index > start_after and item.series_uid not in used:
            return item
    for item in candidates:
        if item.series_uid not in used:
            return item
    return None


def choose_phase_series_from_metadata(candidates: list[SeriesInfo]) -> PhaseMapping:
    if len(candidates) < 3:
        raise ValueError("At least three contrast series are required for arterial, portal, and venous phase mapping")

    used: set[str] = set()
    arterial = _find_by_keywords(candidates, used=used, keywords=("arterial", "artery", "cta"))
    if arterial is None:
        arterial = candidates[0]
    used.add(arterial.series_uid)
    arterial_index = candidates.index(arterial)

    portal = _find_by_keywords(candidates, used=used, keywords=("portal", "pv"), start_after=arterial_index)
    if portal is None:
        portal = _first_unused(candidates, used=used, start_after=arterial_index)
    if portal is None:
        raise ValueError("Not enough series to assign portal phase")
    used.add(portal.series_uid)
    portal_index = candidates.index(portal)

    venous = _find_by_keywords(
        candidates,
        used=used,
        keywords=("venous", "delay", "delayed", "late", "equilibrium"),
        start_after=portal_index,
    )
    if venous is None:
        venous = _first_unused(candidates, used=used, start_after=portal_index)
    if venous is None:
        raise ValueError("Not enough series to assign venous phase")
    used.add(venous.series_uid)

    return PhaseMapping(
        arterial_uid=arterial.series_uid,
        portal_uid=portal.series_uid,
        venous_uid=venous.series_uid,
        support_uids=[item.series_uid for item in candidates if item.series_uid not in used],
    )
