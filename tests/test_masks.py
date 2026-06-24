import numpy as np
import SimpleITK as sitk

from ct_eus_vessel.masks import anchor_multilabel_from_weak_label, body_region_mask, bone_like_exclusion, masks_from_weak_label


def test_weak_label_background_is_not_hard_exclusion() -> None:
    label = sitk.GetImageFromArray(np.array([[[0, 6], [8, 11]]], dtype=np.uint16))
    reference = sitk.Image(label)

    hard, soft = masks_from_weak_label(label, reference, organ_label_ids={6, 11}, vessel_label_ids={8})

    assert hard is None
    assert not soft[0, 0, 0]
    assert soft[0, 0, 1]
    assert not soft[0, 1, 0]
    assert soft[0, 1, 1]


def test_bone_like_exclusion_dilates_high_hu_but_preserves_anchor() -> None:
    image = np.zeros((1, 5, 5), dtype=np.float32)
    image[0, 2, 2] = 900
    anchors = np.zeros_like(image, dtype=bool)
    anchors[0, 2, 2] = True

    exclusion = bone_like_exclusion(image, hu_threshold=700, dilation_voxels=1, preserve_mask=anchors)

    assert not exclusion[0, 2, 2]
    assert exclusion[0, 2, 1]
    assert exclusion[0, 1, 2]


def test_anchor_multilabel_from_weak_label_maps_vessel_classes() -> None:
    label = sitk.GetImageFromArray(np.array([[[8, 25, 10], [9, 15, 6]]], dtype=np.uint16))
    reference = sitk.Image(label)

    mapped = anchor_multilabel_from_weak_label(
        label,
        reference,
        arterial_ids={8, 25},
        portal_ids={10},
        venous_ids={9, 15},
    )

    assert mapped[0, 0, 0] == 1
    assert mapped[0, 0, 1] == 1
    assert mapped[0, 0, 2] == 2
    assert mapped[0, 1, 0] == 3
    assert mapped[0, 1, 1] == 3
    assert mapped[0, 1, 2] == 0


def test_body_region_mask_keeps_body_and_removes_separate_ct_table() -> None:
    image = np.full((4, 14, 14), -1000, dtype=np.float32)
    image[:, 2:9, 3:10] = 40
    image[:, 5:7, 5:7] = 180
    image[:, 12, 1:13] = 300

    body = body_region_mask(
        image,
        spacing_xyz=(1.0, 1.0, 1.0),
        min_hu=-600,
        closing_mm=0,
        dilation_mm=0,
    )

    assert body[:, 2:9, 3:10].all()
    assert not body[:, 12, 1:13].any()
