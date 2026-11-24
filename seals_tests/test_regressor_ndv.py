"""A static regressor's declared no-data cells must not enter calibration as values or decide ranks in allocation.

seals_utils.load_static_regressor reads the no-data value each file declares. Calibration normalizes the valid cells
among themselves; additive no-data cells then take the block's valid median and multiplicative ones take 0, which leaves
the layer's factor at 1. A block with no no-data cell is read exactly as before. The base-data tests use the real
travel-time raster on a coastal tile (145_98) and an interior tile without no-data (130_105).
"""

import os
import tempfile

import hazelbean as hb
import numpy as np
import pytest
from osgeo import gdal

from seals import seals_cython_functions, seals_utils
from test_regional_demand_feasibility import write_raster

INT_NDV = -2147483647

def block():
    """A 4 x 4 travel-time block whose top-left cell is no-data."""
    values = np.arange(10, 170, 10, dtype=np.int32).reshape(4, 4)
    values[0, 0] = INT_NDV
    return values

def written(array, ndv, data_type):
    path = os.path.join(tempfile.mkdtemp(), 'regressor.tif')
    write_raster(path, array, 0.5, ndv, data_type)
    return path

def test_mask_follows_the_declared_value():
    values = np.array([[1.0, -9999.0], [np.nan, 4.0]])
    assert seals_utils.regressor_ndv_mask(values, None).sum() == 0
    assert seals_utils.regressor_ndv_mask(values, -9999.0).tolist() == [[False, True], [False, False]]
    assert seals_utils.regressor_ndv_mask(values, float('nan')).tolist() == [[False, False], [True, False]]

@pytest.mark.parametrize('ndv, data_type, dtype', [(INT_NDV, gdal.GDT_Int32, np.int32), (-9999.0, gdal.GDT_Float32, np.float32)])
def test_allocation_gives_no_data_the_valid_median(ndv, data_type, dtype):
    values = block().astype(dtype)
    values[0, 0] = ndv
    loaded = seals_utils.load_static_regressor(written(values, ndv, data_type), [0, 0, 4, 4], 'additive', False)
    assert loaded[0, 0] == np.median(values.ravel()[1:])
    assert np.array_equal(loaded.ravel()[1:], values.ravel()[1:])

def test_calibration_normalizes_the_valid_cells_among_themselves():
    values = block()
    loaded = seals_utils.load_static_regressor(written(values, INT_NDV, gdal.GDT_Int32), [0, 0, 4, 4], 'additive', True)
    valid = np.ones(values.shape, bool)
    valid[0, 0] = False
    assert loaded[valid].min() == 0.0 and loaded[valid].max() == 1.0
    assert loaded[0, 0] == np.median(loaded[valid])
    # As it was: the no-data minimum squeezes every valid cell into a range near 1e-8.
    before = np.nan_to_num(hb.normalize_array(values), nan=0.0)
    assert before[valid].max() - before[valid].min() < 1e-7

def test_multiplicative_no_data_leaves_the_factor_at_one():
    values = np.ones((4, 4), dtype=np.uint8)
    values[0, 0] = 255
    loaded = seals_utils.load_static_regressor(written(values, 255, gdal.GDT_Byte), [0, 0, 4, 4], 'multiplicative', False)
    assert loaded[0, 0] == 0.0 and (loaded.ravel()[1:] == 1.0).all()

def test_a_block_of_only_no_data_adds_nothing():
    values = np.full((4, 4), INT_NDV, dtype=np.int32)
    path = written(values, INT_NDV, gdal.GDT_Int32)
    for normalize in (False, True):
        assert (seals_utils.load_static_regressor(path, [0, 0, 4, 4], 'additive', normalize) == 0.0).all()

def test_a_block_without_no_data_reads_exactly_as_before():
    values = np.arange(16, dtype=np.int32).reshape(4, 4) + 5
    path = written(values, INT_NDV, gdal.GDT_Int32)
    raw = hb.load_geotiff_chunk_by_cr_size(path, [0, 0, 4, 4])
    assert seals_utils.load_static_regressor(path, [0, 0, 4, 4], 'additive', False).tobytes() == raw.tobytes()
    before = np.nan_to_num(hb.normalize_array(raw), nan=0.0)
    assert seals_utils.load_static_regressor(path, [0, 0, 4, 4], 'additive', True).tobytes() == before.tobytes()

def allocate_one_urban_cell(layer):
    """One coarse cell, all othernat, gains one urban cell ranked by a travel-time layer with a negative coefficient."""
    lulc = np.full((4, 4), 5, dtype=np.int64)
    change = np.zeros((5, 1, 1))
    change[0, 0, 0] = 1
    coefficients = np.zeros((5, 1))
    coefficients[0, 0] = -1.0                  # urban prefers short travel times
    out = seals_cython_functions.seals_allocation_gridded_input(
        change, lulc, layer[np.newaxis].astype(np.float64), coefficients, np.array([2], dtype=np.int64), np.ones((4, 4), dtype=np.int64),
        np.array([1, 2, 3, 4, 5], dtype=np.int64), ['urban', 'cropland', 'grassland', 'forest', 'othernat'], np.ones((4, 4)),
        tempfile.mkdtemp(), 0.0, np.int64(0), '', '')
    return np.argwhere(out[0] == 1).tolist()

def test_no_data_cell_no_longer_takes_the_first_rank():
    values = block()
    assert allocate_one_urban_cell(values) == [[0, 0]]       # as it was: -1 x -2147483647 ranks the no-data cell first
    path = written(values, INT_NDV, gdal.GDT_Int32)
    assert allocate_one_urban_cell(seals_utils.load_static_regressor(path, [0, 0, 4, 4], 'additive', False)) == [[0, 1]]

BASE_DATA_DIR = os.path.join(os.path.expanduser('~'), 'Files', 'base_data')   # as in conftest.py
TRAVEL_TIME_REF_PATH = os.path.join('seals', 'static_regressors', 'travel_time_to_market_mins.tif')

def tile(col, row):
    path = os.path.join(BASE_DATA_DIR, TRAVEL_TIME_REF_PATH)
    if not os.path.exists(path):
        pytest.skip('travel-time raster not in base_data')
    return path, [col * 360, row * 360, 360, 360]

@pytest.mark.requires_base_data
def test_coastal_tile_keeps_no_data_out():
    path, cr = tile(145, 98)
    raw = hb.load_geotiff_chunk_by_cr_size(path, cr)
    invalid = raw == hb.get_ndv_from_path(path)
    assert invalid.any() and not invalid.all()
    allocation = seals_utils.load_static_regressor(path, cr, 'additive', False)
    calibration = seals_utils.load_static_regressor(path, cr, 'additive', True)
    assert allocation.min() >= raw[~invalid].min() and allocation.max() <= raw[~invalid].max()
    assert calibration[~invalid].max() - calibration[~invalid].min() == 1.0
    assert np.unique(calibration[invalid]).size == 1

@pytest.mark.requires_base_data
def test_interior_tile_is_byte_identical():
    path, cr = tile(130, 105)
    raw = hb.load_geotiff_chunk_by_cr_size(path, cr)
    assert seals_utils.load_static_regressor(path, cr, 'additive', False).tobytes() == raw.tobytes()
    before = np.nan_to_num(hb.normalize_array(raw), nan=0.0)
    assert seals_utils.load_static_regressor(path, cr, 'additive', True).tobytes() == before.tobytes()
