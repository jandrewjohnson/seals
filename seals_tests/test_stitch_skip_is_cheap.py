"""A pass over stitched maps that already exist must not walk the allocation zones or re-read every
map for the non-changing-classes check: on 21 Sep 2026 that cost 25-35 minutes per pass on the
NGFS rerun. The check's verdict is recorded beside the map and honoured while map and base are
the very same files."""
import inspect
import os
import time
from seals import seals_main, seals_utils


def _raster_stub(path, size):
    with open(path, 'wb') as f:
        f.write(b'\x00' * size)
    return str(path)


def test_recorded_check_is_honoured_and_invalidated_by_a_rebuilt_map(tmp_path):
    projected = _raster_stub(tmp_path / 'lulc_2050.tif', 2048)
    base = _raster_stub(tmp_path / 'lulc_2023.tif', 2048)
    assert not seals_utils.stitched_map_already_checked(projected, base)
    seals_utils.record_stitched_map_checked(projected, base)
    assert seals_utils.stitched_map_already_checked(projected, base)
    later = time.time() + 10
    os.utime(projected, (later, later))
    assert not seals_utils.stitched_map_already_checked(projected, base)


def test_record_names_both_files(tmp_path):
    projected = _raster_stub(tmp_path / 'lulc_2050.tif', 100)
    base = _raster_stub(tmp_path / 'lulc_2023.tif', 200)
    seals_utils.record_stitched_map_checked(projected, base)
    record = seals_utils._stitched_check_record(projected, base)
    assert record['map']['size'] == 100 and record['base']['size'] == 200
    assert os.path.exists(projected + '.non_changing_check.json')


def test_stitch_task_lists_tiles_only_when_the_map_is_missing():
    src = inspect.getsource(seals_main.stitched_lulc_simplified_scenarios)
    guard = src.index('if not hb.path_exists(p.lulc_projected_stitched_path):')
    listing = src.index('list_filtered_paths_recursively(')
    assert guard < listing, 'the allocation zones are walked before the map is known to be missing'
    check = src.index('stitched_map_already_checked(')
    assertion = src.index('assert_non_changing_classes_unchanged(')
    recorded = src.index('record_stitched_map_checked(')
    assert check < assertion < recorded
