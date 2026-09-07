"""Interpolating a coarse state at a year the coarse model does not step at.

The arithmetic matters more than it looks: the coarse map reaches allocation as an anomaly added to
the regional total, so its amplitude is not renormalised away, and getting the weight wrong biases
how sharply change concentrates inside each region.
"""
import os

import numpy as np
import pytest

import hazelbean as hb
from seals import seals_utils


def write_raster(path, array):
    from osgeo import gdal, osr
    hb.create_directories(os.path.dirname(path))
    array = np.asarray(array, dtype='float64')
    dataset = gdal.GetDriverByName('GTiff').Create(
        path, array.shape[1], array.shape[0], 1, gdal.GDT_Float64)
    dataset.SetGeoTransform((0.0, 1.0, 0.0, 0.0, 0.0, -1.0))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(-9999.0)
    band.WriteArray(array)
    dataset = None


@pytest.fixture
def coarse_tree(tmp_path):
    """A coarse directory with states at 2020 and 2025 and nothing between."""
    source_dir = str(tmp_path / 'prop')
    earlier = np.array([[0.0, 10.0], [20.0, 30.0]])
    later = np.array([[0.0, 20.0], [40.0, 80.0]])
    write_raster(os.path.join(source_dir, '2020', 'cropland_prop_2020.tif'), earlier)
    write_raster(os.path.join(source_dir, '2025', 'cropland_prop_2025.tif'), later)
    return source_dir, earlier, later


def test_years_available_reads_the_directory_names(coarse_tree):
    source_dir, _, _ = coarse_tree
    assert seals_utils.coarse_years_available(source_dir) == [2020, 2025]


@pytest.mark.parametrize('target, expected', [
    (2023, (2020, 2025)),      # the case this exists for
    (2020, None),              # already a step, leave it alone
    (2025, None),
    (2015, None),              # cannot be bracketed
    (2030, None),
])
def test_bracketing(coarse_tree, target, expected):
    source_dir, _, _ = coarse_tree
    assert seals_utils.bracketing_coarse_years(source_dir, target) == expected


def test_interpolated_values_use_the_right_weight(coarse_tree):
    """2023 sits 3/5 of the way from 2020 to 2025, so the weight is 0.6."""
    source_dir, earlier, later = coarse_tree
    written = seals_utils.interpolate_coarse_state_at_year(
        source_dir, 2023, (2020, 2025), 'cropland_prop_{year}.tif')

    assert len(written) == 1
    result = hb.as_array(written[0])
    expected = earlier + 0.6 * (later - earlier)      # [[0, 16], [32, 60]]
    assert np.allclose(result, expected)
    # and it must lie between the brackets everywhere, which is the property that matters
    assert np.all(result >= np.minimum(earlier, later) - 1e-9)
    assert np.all(result <= np.maximum(earlier, later) + 1e-9)


def test_rerun_does_not_rewrite(coarse_tree):
    source_dir, _, _ = coarse_tree
    first = seals_utils.interpolate_coarse_state_at_year(
        source_dir, 2023, (2020, 2025), 'cropland_prop_{year}.tif')
    second = seals_utils.interpolate_coarse_state_at_year(
        source_dir, 2023, (2020, 2025), 'cropland_prop_{year}.tif')
    assert len(first) == 1 and second == []


def test_target_outside_the_brackets_raises(coarse_tree):
    source_dir, _, _ = coarse_tree
    with pytest.raises(NameError):
        seals_utils.interpolate_coarse_state_at_year(
            source_dir, 2030, (2020, 2025), 'cropland_prop_{year}.tif')


def test_missing_bracket_file_raises(coarse_tree, tmp_path):
    source_dir, _, _ = coarse_tree
    with pytest.raises(NameError):
        seals_utils.interpolate_coarse_state_at_year(
            source_dir, 2023, (2020, 2025), 'pastureland_prop_{year}.tif')
