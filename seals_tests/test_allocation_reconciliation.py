"""The allocation reconciliation diagnostic: does it report what the allocator was asked for, what it
did, and the cases where a contraction target exceeds the stock available to give?"""
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from seals import seals_utils

CLASSES = {'cropland': 2, 'grassland': 3, 'othernat': 5}


def _write(path, array, transform, nodata=None):
    with rasterio.open(path, 'w', driver='GTiff', height=array.shape[0], width=array.shape[1],
                       count=1, dtype=array.dtype, transform=transform, crs='EPSG:4326',
                       nodata=nodata) as dst:
        dst.write(array, 1)
    return str(path)


@pytest.fixture
def grids(tmp_path):
    """One coarse cell over a 4x4 fine block: 8 cells othernat, 8 cropland, one hectare each."""
    fine = from_origin(0, 4, 1, 1)
    coarse = from_origin(0, 4, 4, 4)
    before = np.full((4, 4), CLASSES['cropland'], dtype=np.int32)
    before[:2, :] = CLASSES['othernat']
    after = before.copy()
    after[0, :] = CLASSES['grassland']          # grassland takes 4 ha, all of it from othernat
    ha = np.ones((4, 4), dtype=np.float64)
    return dict(tmp=tmp_path, fine=fine, coarse=coarse,
                before=_write(tmp_path / 'before.tif', before, fine),
                after=_write(tmp_path / 'after.tif', after, fine),
                ha=_write(tmp_path / 'ha.tif', ha, fine))


def _demand(grids, **by_class):
    return {k: _write(grids['tmp'] / ('%s_demand.tif' % k), np.array([[v]], dtype=np.float64),
                      grids['coarse'], nodata=-9999.0) for k, v in by_class.items()}


def test_it_reports_demanded_realised_and_available(grids):
    paths = _demand(grids, grassland=4.0, othernat=-4.0, cropland=0.0)
    out = seals_utils.reconcile_allocation(paths, grids['before'], grids['after'], grids['ha'], CLASSES)
    by = out.set_index('class_label')
    assert by.loc['grassland', 'demanded_ha'] == 4.0
    assert by.loc['grassland', 'realised_ha'] == 4.0
    assert by.loc['othernat', 'realised_ha'] == -4.0
    assert by.loc['othernat', 'available_ha'] == 8.0
    assert by.loc['cropland', 'realised_ha'] == 0.0


def test_a_contraction_larger_than_the_stock_is_flagged(grids):
    paths = _demand(grids, grassland=20.0, othernat=-20.0)
    out = seals_utils.reconcile_allocation(paths, grids['before'], grids['after'], grids['ha'], CLASSES)
    by = out.set_index('class_label')
    assert by.loc['othernat', 'infeasible'], 'othernat demand of -20 ha against 8 ha must be flagged'
    assert not by.loc['grassland', 'infeasible'], 'an expansion is never infeasible on this test'
    # the diagnostic reports the gap without changing it
    assert by.loc['othernat', 'realised_ha'] == -4.0


def test_a_feasible_contraction_is_not_flagged(grids):
    paths = _demand(grids, othernat=-8.0)
    out = seals_utils.reconcile_allocation(paths, grids['before'], grids['after'], grids['ha'], CLASSES)
    assert not out.set_index('class_label').loc['othernat', 'infeasible']


def test_mismatched_grids_raise(grids, tmp_path):
    odd = _write(tmp_path / 'odd.tif', np.ones((3, 3), dtype=np.int32), grids['fine'])
    with pytest.raises(ValueError, match='share one grid'):
        seals_utils.reconcile_allocation(_demand(grids, othernat=-1.0), grids['before'], odd,
                                         grids['ha'], CLASSES)


def test_nodata_in_the_demand_is_not_read_as_a_target(grids):
    path = _write(grids['tmp'] / 'nd_demand.tif', np.array([[-9999.0]]), grids['coarse'], nodata=-9999.0)
    out = seals_utils.reconcile_allocation({'othernat': path}, grids['before'], grids['after'],
                                           grids['ha'], CLASSES)
    assert out.set_index('class_label').loc['othernat', 'demanded_ha'] == 0.0
    assert not out.set_index('class_label').loc['othernat', 'infeasible']


# ---------------------------------------------------------------------------
# The diagnostic certifies the pilot, so it must FAIL rather than absorb a bad input
# ---------------------------------------------------------------------------

def test_a_NONFINITE_demand_cell_FAILS_instead_of_counting_as_zero(grids):
    """Zeroing a NaN demand reports an unmet demand as satisfied."""
    path = _write(grids['tmp'] / 'nan_demand.tif', np.array([[np.nan]], dtype=np.float64),
                  grids['coarse'])
    with pytest.raises(ValueError, match='non-finite'):
        seals_utils.reconcile_allocation({'grassland': path}, grids['before'], grids['after'],
                                         grids['ha'], CLASSES)


def test_a_NONFINITE_cell_DECLARED_as_nodata_is_accepted_as_no_demand(grids):
    """A raster that declares a non-finite nodata is saying those cells are unmeasured."""
    path = _write(grids['tmp'] / 'nan_ndv.tif', np.array([[np.nan]], dtype=np.float64),
                  grids['coarse'], nodata=np.nan)
    out = seals_utils.reconcile_allocation({'grassland': path}, grids['before'], grids['after'],
                                           grids['ha'], CLASSES)
    assert (out.demanded_ha == 0).all()


def test_a_DEMAND_RASTER_THAT_DOES_NOT_COVER_THE_FINE_GRID_FAILS(grids):
    """Clamping an out-of-range fine cell into an edge cell attributes land to the wrong cell."""
    offset = from_origin(100, 104, 4, 4)          # nowhere near the fine grid
    path = _write(grids['tmp'] / 'offset.tif', np.array([[4.0]]), offset)
    with pytest.raises(ValueError, match='does not cover the fine grid'):
        seals_utils.reconcile_allocation({'grassland': path}, grids['before'], grids['after'],
                                         grids['ha'], CLASSES)


def test_a_CLASS_THAT_APPEARS_WITH_NO_DEMAND_AND_NO_STOCK_IS_REPORTED(grids):
    """The allocator gaining a class nobody asked for is exactly what must not be omitted."""
    path = _write(grids['tmp'] / 'zero_demand.tif', np.array([[0.0]]), grids['coarse'])
    out = seals_utils.reconcile_allocation({'grassland': path}, grids['before'], grids['after'],
                                           grids['ha'], CLASSES)
    row = out[out.class_label == 'grassland']
    assert len(row) == 1, 'a cell that gained grassland with no demand and no stock must appear'
    assert float(row.demanded_ha.iloc[0]) == 0.0
    assert float(row.realised_ha.iloc[0]) == 4.0, 'the unrequested gain is reported, not dropped'
