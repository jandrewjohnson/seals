"""Block lists must work whichever path produced them: fresh ints or resumed strings.

Job 3472099 resumed the pilot, reused all 772 accepted maps with byte-identical hashes, and then
failed in read_inputs:

    col,row = p.fine_blocks_list[:2]
    TypeError: can't multiply sequence by non-int of type 'float'

On a FRESH run seals_main builds the lists with get_subglobal_block_list_from_resolution_and_bb,
which yields ints. On a RESUME it reads the precached CSV via
hb.file_to_python_object(path, '2d_list') and passes no output_value_data_type, so cast_type never
runs and every element is a STRING. Measured on this pilot's own cached file, the first row is
['0', '0', '360', '360', '0', '0'] -- all str -- and '0' * 0.0027777778 raises that exact error,
because a string is a sequence.

So this was a resume-only defect: 772 tiles allocated cleanly on the fresh path first.
"""
import pytest

from seals.allocation_tasks import _block_offsets


# The pilot's real cached first row, as strings.
CACHED_ROW = ['0', '0', '360', '360', '0', '0']
FRESH_ROW = [0, 0, 360, 360, 0, 0]


def test_the_resumed_string_row_becomes_the_fresh_int_row():
    assert _block_offsets(CACHED_ROW) == FRESH_ROW
    assert all(isinstance(v, int) for v in _block_offsets(CACHED_ROW))


def test_the_fresh_int_row_is_unchanged():
    assert _block_offsets(FRESH_ROW) == FRESH_ROW


def test_the_geotransform_offset_now_computes_on_both_paths():
    """The exact arithmetic that raised, over both representations."""
    gt = [16.0, 0.0027777778, 0.0, -22.0, 0.0, -0.0027777778]
    results = []
    for row in (CACHED_ROW, FRESH_ROW):
        col, r = _block_offsets(row)[:2]
        results.append((gt[0] + col * gt[1] + r * gt[2], gt[3] + col * gt[4] + r * gt[5]))
    assert results[0] == results[1], 'a resumed run must place the tile where a fresh run does'


def test_a_nonzero_offset_shifts_the_origin_identically_on_both_paths():
    gt = [16.0, 0.0027777778, 0.0, -22.0, 0.0, -0.0027777778]
    pairs = []
    for row in (['720', '1080', '360', '360'], [720, 1080, 360, 360]):
        col, r = _block_offsets(row)[:2]
        pairs.append((gt[0] + col * gt[1], gt[3] + r * gt[5]))
    assert pairs[0] == pairs[1]
    assert pairs[0][0] == pytest.approx(16.0 + 720 * 0.0027777778)


def test_the_raw_string_row_really_does_raise_without_normalisation():
    """Pins the defect, so this guard cannot quietly stop testing anything."""
    with pytest.raises(TypeError, match="can't multiply sequence by non-int"):
        CACHED_ROW[0] * 0.0027777778


def test_floats_that_are_whole_numbers_are_accepted():
    assert _block_offsets(['0.0', '360.0', 360.0]) == [0, 360, 360]


def test_a_fractional_element_is_refused_rather_than_truncated():
    """Pixel indices are whole. A fractional one means the list is not what this task thinks."""
    with pytest.raises(ValueError, match='not a whole number'):
        _block_offsets(['0', '12.5'])
