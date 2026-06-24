from pathlib import Path

import pydicom
from pydicom.dataset import FileDataset
from pydicom.uid import ExplicitVRLittleEndian

from ct_eus_vessel.dicom_index import index_dicom_series


def _write_header(path: Path, *, series_uid: str, z: float, instance: int, protocol: str = "Arterial Phase") -> None:
    meta = pydicom.Dataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    meta.MediaStorageSOPInstanceUID = f"{series_uid}.{instance}"
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.Modality = "CT"
    ds.SeriesInstanceUID = series_uid
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.ProtocolName = protocol
    ds.SeriesDescription = "1.0 x 1.0_A"
    ds.ConvolutionKernel = "B_SOFT_B"
    ds.SliceThickness = "1.0"
    ds.AcquisitionTime = "123456.000"
    ds.SeriesTime = "123400"
    ds.BodyPartExamined = "ABDOMEN"
    ds.FrameOfReferenceUID = "1.2.3.99"
    ds.PixelSpacing = ["0.5", "0.5"]
    ds.ImagePositionPatient = ["1.0", "2.0", str(z)]
    ds.ImageOrientationPatient = ["1", "0", "0", "0", "1", "0"]
    ds.InstanceNumber = instance
    ds.save_as(path)


def test_index_dicom_series_groups_headers_and_extracts_geometry(tmp_path: Path) -> None:
    _write_header(tmp_path / "a1.dcm", series_uid="1.2.3.1", z=10.0, instance=1)
    _write_header(tmp_path / "a2.dcm", series_uid="1.2.3.1", z=11.0, instance=2)
    _write_header(tmp_path / "b1.dcm", series_uid="1.2.3.2", z=20.0, instance=1, protocol="Portal Phase")

    indexed = index_dicom_series(tmp_path)

    assert [item.series_uid for item in indexed] == ["1.2.3.1", "1.2.3.2"]
    first = indexed[0]
    assert first.n_slices == 2
    assert first.protocol_name == "Arterial Phase"
    assert first.convolution_kernel == "B_SOFT_B"
    assert first.slice_thickness_mm == 1.0
    assert first.spacing_xy == (0.5, 0.5)
    assert first.z_range == (10.0, 11.0)
    assert first.direction == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert [path.name for path in first.files] == ["a1.dcm", "a2.dcm"]
