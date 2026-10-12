"""A candidate tile must be written at the TILE's extent, not the reference raster's.

Job 3467067 reached allocation and failed in allocation_audit with
"output shape (4680,6120) versus input (360,360)". The write passed the AOI-wide base raster as
save_array_as_geotiff's match argument together with the tile's geotransform. That function takes
n_cols/n_rows from the match raster whenever the caller leaves them None -- its `array.shape`
fallback is positioned AFTER the match block, so it never runs -- so the file came out at the
reference's 4680x6120 while carrying the tile's geotransform. Wrong on both counts: wrong extent,
and an origin that claims the tile while spanning the whole AOI.

The reference raster is there to supply projection and data type. It must never supply extent.
"""
import numpy as np
import pytest
from osgeo import gdal

import hazelbean as hb


REFERENCE = (4680, 6120)   # rows, cols -- the AOI-wide fine raster this run uses
TILE = (360, 360)


def _reference_raster(path, rows=REFERENCE[0] // 13, cols=REFERENCE[1] // 17):
    """A larger reference raster. Shrunk by the tile factor so the test stays fast; the point is
    only that it is LARGER than the tile and has its own geotransform."""
    ds = gdal.GetDriverByName('GTiff').Create(str(path), cols, rows, 1, gdal.GDT_Byte)
    ds.SetGeoTransform([-180.0, 360.0 / cols, 0.0, 90.0, 0.0, -180.0 / rows])
    ds.GetRasterBand(1).WriteArray(np.zeros((rows, cols), dtype=np.uint8))
    ds = None
    return str(path)


def _tile_geotransform(reference_path, col_offset, row_offset):
    """The tile's geotransform, offset from the reference exactly as read_inputs computes it."""
    gt = list(hb.get_raster_info_hb(reference_path)['geotransform'])
    gt[0] += col_offset * gt[1] + row_offset * gt[2]
    gt[3] += col_offset * gt[4] + row_offset * gt[5]
    return gt


def test_a_small_tile_is_written_at_its_own_size_not_the_references(tmp_path):
    reference = _reference_raster(tmp_path / 'reference.tif')
    reference_size = tuple(hb.get_raster_info_hb(reference)['raster_size'])
    values = np.full((40, 40), 3, dtype=np.uint8)
    assert (values.shape[1], values.shape[0]) != reference_size, 'the tile must be smaller'

    out = str(tmp_path / 'candidate.tif')
    gt = _tile_geotransform(reference, 80, 120)
    hb.save_array_as_geotiff(values, out, reference, geotransform_override=gt,
                             n_cols=values.shape[1], n_rows=values.shape[0],
                             ndv=255, data_type=1, compress=True)

    info = hb.get_raster_info_hb(out)
    assert tuple(info['raster_size']) == (values.shape[1], values.shape[0]), (
        'written %s, allocated %s' % (tuple(info['raster_size']), (values.shape[1], values.shape[0])))
    assert tuple(info['raster_size']) != reference_size, 'it must not inherit the reference extent'


def test_the_tile_geotransform_survives_the_write(tmp_path):
    reference = _reference_raster(tmp_path / 'reference.tif')
    values = np.full((40, 40), 2, dtype=np.uint8)
    gt = _tile_geotransform(reference, 80, 120)
    out = str(tmp_path / 'candidate.tif')
    hb.save_array_as_geotiff(values, out, reference, geotransform_override=gt,
                             n_cols=values.shape[1], n_rows=values.shape[0],
                             ndv=255, data_type=1, compress=True)
    written = np.asarray(hb.get_raster_info_hb(out)['geotransform'], dtype=float)
    assert np.allclose(written, np.asarray(gt, dtype=float), rtol=0, atol=1e-12)
    # And it is NOT the reference's own origin, which is what a dropped override would give.
    assert not np.allclose(written, np.asarray(
        hb.get_raster_info_hb(reference)['geotransform'], dtype=float))


def test_the_values_read_back_unchanged(tmp_path):
    # Dimensions alone are not enough: the pixels must be the allocated ones, in place.
    reference = _reference_raster(tmp_path / 'reference.tif')
    rng = np.random.default_rng(20261012)
    values = rng.integers(1, 6, size=(40, 40)).astype(np.uint8)
    out = str(tmp_path / 'candidate.tif')
    hb.save_array_as_geotiff(values, out, reference,
                             geotransform_override=_tile_geotransform(reference, 80, 120),
                             n_cols=values.shape[1], n_rows=values.shape[0],
                             ndv=255, data_type=1, compress=True)
    assert np.array_equal(hb.as_array(out), values)


def test_omitting_the_dimensions_reproduces_the_failure(tmp_path):
    """The defect itself, pinned. Without explicit n_cols/n_rows the reference extent wins."""
    reference = _reference_raster(tmp_path / 'reference.tif')
    reference_size = tuple(hb.get_raster_info_hb(reference)['raster_size'])
    values = np.full((40, 40), 3, dtype=np.uint8)
    out = str(tmp_path / 'wrong.tif')
    hb.save_array_as_geotiff(values, out, reference,
                             geotransform_override=_tile_geotransform(reference, 80, 120),
                             ndv=255, data_type=1, compress=True)
    written = tuple(hb.get_raster_info_hb(out)['raster_size'])
    assert written == reference_size, (
        'this is the bug being guarded against: the file takes the REFERENCE extent %s rather '
        'than the array extent %s' % (reference_size, (values.shape[1], values.shape[0])))
    assert written != (values.shape[1], values.shape[0])
