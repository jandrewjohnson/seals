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


def test_each_map_is_recorded_before_the_next_year_begins():
    """An interrupted stitch must not leave a finished map without its provenance record.

    Writing the records only after the whole scenario-year loop left every finished map unrecorded
    for the rest of the run. All three bounded runs on 14 Oct 2026 failed that way -- not from a
    node failure but because the pipeline itself raised partway through the loop -- and each one
    refused a map an earlier attempt had left unrecorded, on a different year.

    The record is therefore written twice and both are required. Once inside the loop, so a map is
    never on disk without one; and once after the POG conversion, which rewrites the maps in place,
    so the stored hash describes the bytes finally there. record_stitched leaves an identical
    record alone, so the second pass only rewrites what the conversion changed.

    The conversion stays 16-wide. Serialising it to collapse the two writes into one would cost
    hours on 27 global maps; a crash inside the conversion instead leaves a stale record, which
    fails loudly as 'provenance differs' and is cleared by deleting that map.
    """
    src = inspect.getsource(seals_main.stitched_lulc_simplified_scenarios)
    in_loop = src.index('record_stitched(p.lulc_projected_stitched_path')
    cleanup = src.index('for file_path in vrt_paths_to_remove:')
    convert = src.index('make_paths_pogs_in_parallel(')
    refresh = src.index('for path, tiles, baseline in strict_stitched:')
    assert in_loop < cleanup, 'the record is written after the year loop rather than inside it'
    assert cleanup < convert < refresh, 'the records must be refreshed after the in-place conversion'
    assert 'max_workers=16' in src, 'the conversion must stay parallel'
