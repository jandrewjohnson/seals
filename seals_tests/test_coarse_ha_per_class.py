"""Aggregating a fine LULC map to hectares per class per coarse cell, which feeds the observed coarse change that calibration filters its zones on."""
import numpy as np
import pytest
from osgeo import gdal

from seals import seals_utils

gdal.UseExceptions()


def write(path, array):
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(path), array.shape[1], array.shape[0], 1, gdal.GDT_Byte)
    ds.SetGeoTransform([0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
    ds.GetRasterBand(1).WriteArray(array)
    ds = None
    return str(path)


def test_area_is_class_fraction_times_coarse_hectares(tmp_path):
    """A 4x4 fine map over a 2x2 coarse grid: each coarse cell holds four fine cells."""
    fine = np.array([[1, 1, 2, 2],
                     [1, 2, 2, 2],
                     [3, 3, 1, 1],
                     [3, 3, 1, 1]], dtype=np.uint8)
    ha_per_cell_coarse = np.array([[100.0, 200.0], [300.0, 400.0]])

    ha = seals_utils.coarse_ha_per_class_from_fine_lulc(write(tmp_path / "fine.tif", fine), [1, 2, 3], ha_per_cell_coarse)

    np.testing.assert_allclose(ha[1], [[75.0, 0.0], [0.0, 400.0]])
    np.testing.assert_allclose(ha[2], [[25.0, 200.0], [0.0, 0.0]])
    np.testing.assert_allclose(ha[3], [[0.0, 0.0], [300.0, 0.0]])


def test_reading_in_chunks_gives_the_same_answer(tmp_path):
    rng = np.random.default_rng(0)
    fine = rng.integers(1, 4, size=(30, 12)).astype(np.uint8)
    ha_per_cell_coarse = rng.uniform(50, 150, size=(10, 4))
    path = write(tmp_path / "fine.tif", fine)

    whole = seals_utils.coarse_ha_per_class_from_fine_lulc(path, [1, 2, 3], ha_per_cell_coarse, coarse_rows_per_chunk=10)
    chunked = seals_utils.coarse_ha_per_class_from_fine_lulc(path, [1, 2, 3], ha_per_cell_coarse, coarse_rows_per_chunk=3)

    for i in [1, 2, 3]:
        np.testing.assert_allclose(chunked[i], whole[i])
    np.testing.assert_allclose(sum(whole.values()), ha_per_cell_coarse)


def test_a_fine_grid_that_does_not_nest_is_refused(tmp_path):
    fine = np.ones((5, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="not a whole multiple"):
        seals_utils.coarse_ha_per_class_from_fine_lulc(write(tmp_path / "fine.tif", fine), [1], np.ones((2, 2)))
