"""lulc_convolutions with a fake worker pool, which records the convolutions the task schedules."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from seals import seals_generate_base_data

BINARY = Path('lulc') / 'esa' / 'seals7' / 'binaries' / '2020' / 'binary_esa_seals7_2020_forest.tif'


class FakePool:
    """Records the scheduled convolutions instead of running the FFTs."""
    scheduled: list = []

    def __init__(self, num_workers: int) -> None:
        FakePool.scheduled = []

    def starmap_async(self, function, iterable: list) -> SimpleNamespace:
        FakePool.scheduled = list(iterable)
        return SimpleNamespace(get=lambda: [])

    def close(self) -> None:
        pass

    def join(self) -> None:
        pass


def project(tmp_path: Path, **attributes) -> SimpleNamespace:
    """A global run whose binary is in base_data and whose convolution is missing."""
    fine, base = tmp_path / 'fine_processed_inputs', tmp_path / 'base_data'
    binary = base / BINARY
    binary.parent.mkdir(parents=True)
    binary.write_text('x')

    def get_path(ref_path: str, **kwargs) -> str:
        # Returns the base_data path if the file exists there, else the path to generate, as p.get_path does.
        return str(base / ref_path) if (base / ref_path).exists() else str(fine / ref_path)

    return SimpleNamespace(
        run_this=True, aoi='global', all_class_labels=['forest'], gaussian_sigmas_to_test=[1],
        years_to_convolve_override=None, key_base_year=2020, lulc_src_label='esa',
        lulc_simplification_label='seals7', fine_processed_inputs_dir=str(fine), base_data_dir=str(base),
        aoi_binary_paths={2020: {'forest': str(binary)}}, get_path=get_path, **attributes)


def test_convolution_reads_the_found_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(seals_generate_base_data.multiprocessing, 'Pool', FakePool)

    seals_generate_base_data.lulc_convolutions(project(tmp_path))

    assert FakePool.scheduled[0][0] == str(tmp_path / 'base_data' / BINARY)
