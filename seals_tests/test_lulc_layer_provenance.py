"""Provenance-checked reuse and promotion of the simplified LULC layers in base_data."""

import json
from pathlib import Path

from seals import seals_utils

SCHEME = Path('lulc') / 'esa' / 'seals7'
CONVOLUTION = SCHEME / 'convolutions' / '2020' / 'convolution_esa_seals7_2020_forest_gaussian_1.tif'
BINARY = SCHEME / 'binaries' / '2020' / 'binary_esa_seals7_2020_forest.tif'


def record(mapping: dict) -> dict:
    return seals_utils.lulc_layer_provenance(mapping)


def touch(path: Path, content: str = 'x') -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_record_ignores_key_order_and_type() -> None:
    assert record({10: 2, 0: 0}) == record({'0': 0, '10': 2})


def test_status_is_unknown_without_record(tmp_path: Path) -> None:
    assert seals_utils.lulc_layers_status(str(tmp_path), record({10: 2})) == 'unknown'


def test_status_match_and_mismatch(tmp_path: Path) -> None:
    (tmp_path / 'provenance.json').write_text(json.dumps(record({10: 2, 30: 3})))

    assert seals_utils.lulc_layers_status(str(tmp_path), record({30: 3, 10: 2})) == 'match'
    # Mosaic cropland counted as natural rather than as cropland.
    assert seals_utils.lulc_layers_status(str(tmp_path), record({10: 2, 30: 5})) == 'mismatch'


def test_promotion_copies_and_records_a_fresh_chain(tmp_path: Path) -> None:
    project, base = tmp_path / 'project' / SCHEME, tmp_path / 'base_data' / SCHEME
    touch(tmp_path / 'project' / CONVOLUTION)

    copied = seals_utils.promote_lulc_layers_to_base_data(str(project), str(base), record({10: 2}))

    assert copied == [str(tmp_path / 'base_data' / CONVOLUTION)]
    assert seals_utils.lulc_layers_status(str(base), record({10: 2})) == 'match'


def test_promotion_never_overwrites(tmp_path: Path) -> None:
    project, base = tmp_path / 'project' / SCHEME, tmp_path / 'base_data' / SCHEME
    touch(tmp_path / 'project' / CONVOLUTION, 'new')
    existing = touch(tmp_path / 'base_data' / CONVOLUTION, 'old')

    assert seals_utils.promote_lulc_layers_to_base_data(str(project), str(base)) == []
    assert existing.read_text() == 'old'


def test_no_record_for_layers_next_to_unrecorded_ones(tmp_path: Path) -> None:
    """Convolutions derived from base_data binaries without a record get no record either."""
    project, base = tmp_path / 'project' / SCHEME, tmp_path / 'base_data' / SCHEME
    touch(tmp_path / 'project' / CONVOLUTION)
    touch(tmp_path / 'base_data' / BINARY)

    seals_utils.promote_lulc_layers_to_base_data(str(project), str(base), record({10: 2}))

    assert (tmp_path / 'base_data' / CONVOLUTION).exists()
    assert seals_utils.lulc_layers_status(str(base), record({10: 2})) == 'unknown'

