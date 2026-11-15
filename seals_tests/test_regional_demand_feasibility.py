"""The pre-allocation stock check, run through ProjectFlow with a stand-in for allocation_zones.

Two regions side by side, each one coarse cell wide and two tall, 4 x 4 fine pixels per coarse cell. Region 1 holds
forest, othernat, cropland and water; region 2 holds othernat and grassland, so region 2's othernat sits across the
border from region 1."""
import os
import numpy as np
import pandas as pd
import pytest
import hazelbean as hb
from osgeo import gdal, osr

from seals import seals_initialize_project, seals_main

LABELS = ['urban', 'cropland', 'grassland', 'forest', 'othernat']
IDS = [1, 2, 3, 4, 5]
ALL_LABELS = LABELS + ['water', 'other']
WATER, NDV = 6, 255
REACHED = []
TILES = {}

def write_raster(path, array, cell_size, ndv, data_type):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(path, array.shape[1], array.shape[0], 1, data_type)
    ds.SetGeoTransform([0.0, cell_size, 0.0, 2.0, 0.0, -cell_size])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(ndv)
    band.WriteArray(array)
    ds = None

def base_map():
    lulc = np.zeros((8, 8), dtype=np.int32)
    lulc[:, :4] = 5                     # region 1: othernat by default
    lulc[0:3, 0:4] = 4                  # 12 forest
    lulc[3:5, 0:4] = 2                  # 8 cropland
    lulc[7, 0:4] = WATER                # 4 water, never stock
    lulc[:, 4:] = 5                     # region 2: 20 othernat
    lulc[0:3, 4:8] = 3                  # 12 grassland
    return lulc                         # region 1 othernat: rows 5-6, 8 pixels

def build(root, changes, years=(2018,), ha=None, protection=None, lulc=None):
    """changes: {year: {region_id: {label: hectares}}}, spread over the region's two coarse cells."""
    lulc = base_map() if lulc is None else lulc
    ha = np.ones((8, 8)) if ha is None else ha
    paths = {'lulc': os.path.join(root, 'lulc_2017.tif'), 'ha': os.path.join(root, 'ha_fine.tif'),
             'regional_change_dir': os.path.join(root, 'regional_change'), 'fine_processed_inputs_dir': os.path.join(root, 'fpi')}
    write_raster(paths['lulc'], lulc, 0.25, NDV, gdal.GDT_Int32)
    write_raster(paths['ha'], ha, 0.25, -9999.0, gdal.GDT_Float64)
    regions = np.array([[1, 2], [1, 2]], dtype=np.int32)
    write_raster(os.path.join(paths['regional_change_dir'], 'region_ids.tif'), regions, 1.0, 0, gdal.GDT_Int32)
    binaries = os.path.join(paths['fine_processed_inputs_dir'], 'lulc', 'esa', 'seals7', 'binaries', '2017')
    for k, label in enumerate(ALL_LABELS):
        write_raster(os.path.join(binaries, 'binary_esa_seals7_2017_' + label + '.tif'), (lulc == k + 1).astype(np.float64), 0.25, -9999.0,
                     gdal.GDT_Float64)
    rows = [dict(spatial_regressor_name='suitability', data_location=paths['lulc'], type='additive', **{l: 1.0 for l in LABELS})]
    rows += [dict(spatial_regressor_name=label + '_constraint', data_location='x.tif', type='multiplicative', **{l: 1.0 for l in LABELS})
             for label in ALL_LABELS]
    if protection is not None:
        paths['protection'] = os.path.join(root, 'protection.tif')
        write_raster(paths['protection'], protection.astype(np.float64), 0.25, -9999.0, gdal.GDT_Float64)
        rows.append(dict(spatial_regressor_name='protection', data_location=paths['protection'], type='multiplicative', **{l: 0.0 for l in LABELS}))
    paths['coefficients'] = os.path.join(root, 'coefficients.csv')
    pd.DataFrame(rows).to_csv(paths['coefficients'], index=False)
    paths['change_dirs'] = {}
    previous = 2017
    for year in years:
        directory = os.path.join(root, 'change', str(year))
        paths['change_dirs'][year] = directory
        for k, label in enumerate(LABELS):
            coarse = np.zeros((2, 2))
            for region_id, by_label in changes.get(year, {}).items():
                coarse[regions == region_id] = by_label.get(label, 0.0) / 2
            write_raster(os.path.join(directory, label + '_%d_%d_ha_diff_ssp2_rcp45_luh2-message_bau.tif' % (year, previous)), coarse, 1.0, -9999.0,
                         gdal.GDT_Float64)
        previous = year
    paths['years'] = list(years)
    return paths

def scenario_years(p):
    """Stand-in for the allocations iterator: one iteration per year of one scenario."""
    if p.run_this:
        years = p.feasibility_test_years
        p.iterator_replacements = {
            'year': years, 'previous_year': [2017] + years[:-1],
            'projected_coarse_change_dir': [p.feasibility_test_change_dirs[y] for y in years],
            'cur_dir_parent_dir': [os.path.join(p.cur_dir, 'ssp2', 'rcp45', 'luh2-message', 'bau', str(y)) for y in years]}
    return p

def allocation_stand_in(p):
    """Stand-in for allocation_zones: records that allocation was reached and writes the year's tile like allocation does."""
    if p.run_this:
        REACHED.append(int(p.year))
        if int(p.year) in TILES:
            tile_dir = os.path.join(p.cur_dir_parent_dir, 'allocation_zones', '0_0', 'allocation')
            after, cumulative = TILES[int(p.year)]
            write_raster(os.path.join(tile_dir, 'lulc_esa_seals7_ssp2_rcp45_luh2-message_bau_%d.tif' % int(p.year)), after.astype(np.int32), 0.25,
                         NDV, gdal.GDT_Int32)
            write_raster(os.path.join(tile_dir, 'cumulative_change_happened.tif'), cumulative.astype(np.int32), 0.25, -9999, gdal.GDT_Int32)
        else:
            os.makedirs(os.path.join(p.cur_dir_parent_dir, 'allocation_zones'), exist_ok=True)
    return p

def run(project_dir, paths):
    del REACHED[:]
    p = hb.ProjectFlow(project_dir=project_dir)
    p.key_base_year = 2017
    p.lulc_simplified_paths = {2017: paths['lulc']}
    p.aoi_ha_per_cell_fine_path = paths['ha']
    p.regional_change_dir = paths['regional_change_dir']
    p.fine_processed_inputs_dir = paths['fine_processed_inputs_dir']
    p.calibration_parameters_source = paths['coefficients']
    p.changing_class_indices, p.changing_class_labels, p.all_class_labels = IDS, LABELS, ALL_LABELS
    p.lulc_ndv = NDV
    p.lulc_src_label, p.lulc_simplification_label = 'esa', 'seals7'
    p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label = 'ssp2', 'rcp45', 'luh2-message', 'bau'
    p.regional_projections_input_path = 'regional_demand.csv'
    p.years = paths['years']
    p.feasibility_test_years = paths['years']
    p.feasibility_test_change_dirs = paths['change_dirs']
    iterator = p.add_iterator(scenario_years)
    check = p.add_task(seals_main.regional_demand_feasibility, parent=iterator, creates_dir=False, skip_existing=0)
    p.add_task(allocation_stand_in, parent=check)
    p.execute()
    return p

def table(project_dir, year=2018):
    path = os.path.join(project_dir, 'intermediate', 'scenario_years', 'ssp2', 'rcp45', 'luh2-message', 'bau', str(year),
                        'regional_demand_feasibility.csv')
    return pd.read_csv(path, encoding='utf-8-sig').set_index(['region_id', 'class_label'])

# --- Registration ---------------------------------------------------------------------------------------------------

def test_standard_tree_checks_each_year_before_its_zones_are_allocated(tmp_path):
    p = hb.ProjectFlow(project_dir=str(tmp_path))
    p.run_in_parallel = 0
    seals_initialize_project.build_standard_task_tree(p)
    check = p.regional_demand_feasibility_task
    assert check.parent is p.allocations_task and check.creates_dir is False
    assert p.allocation_zones_task.parent is check and p.allocation_task.parent is p.allocation_zones_task
    assert [c.name for c in p.allocations_task.children] == ['regional_demand_feasibility']

# --- Pass and refuse ------------------------------------------------------------------------------------------------

def test_feasible_demand_reaches_allocation(tmp_path):
    paths = build(str(tmp_path / 'in'), {2018: {1: {'cropland': 3.0, 'forest': -3.0}}})
    run(str(tmp_path / 'project'), paths)
    assert REACHED == [2018]
    t = table(str(tmp_path / 'project'))
    assert t.loc[(1, 'forest'), 'available_ha'] == 12.0 and t['shortfall_ha'].max() == 0
    # The check makes no directory of its own, so the task under it keeps the path it had under the iterator.
    year_dir = os.path.join(str(tmp_path / 'project'), 'intermediate', 'scenario_years', 'ssp2', 'rcp45', 'luh2-message', 'bau', '2018')
    assert os.path.isdir(os.path.join(year_dir, 'allocation_stand_in')) and not os.path.isdir(os.path.join(year_dir, 'regional_demand_feasibility'))

def test_loss_beyond_stock_fails_before_allocation_naming_region_class_and_hectares(tmp_path):
    paths = build(str(tmp_path / 'in'), {2018: {1: {'cropland': 13.0, 'forest': -13.0}}})
    with pytest.raises(NameError, match=r'region 1 forest loss requested 13 ha, available 12 ha'):
        run(str(tmp_path / 'project'), paths)
    assert REACHED == []
    assert table(str(tmp_path / 'project')).loc[(1, 'forest'), 'shortfall_ha'] == 1.0

def test_exact_capacity_passes(tmp_path):
    paths = build(str(tmp_path / 'in'), {2018: {1: {'cropland': 12.0, 'forest': -12.0}}})
    run(str(tmp_path / 'project'), paths)
    assert REACHED == [2018]

def test_stock_across_the_border_cannot_pay(tmp_path):
    """Region 1 has 8 ha of othernat; region 2's 20 ha next door do not count."""
    paths = build(str(tmp_path / 'in'), {2018: {1: {'cropland': 9.0, 'othernat': -9.0}}})
    with pytest.raises(NameError, match=r'region 1 othernat loss requested 9 ha, available 8 ha'):
        run(str(tmp_path / 'project'), paths)
    assert REACHED == []

def test_unequal_pixel_areas_are_counted_in_hectares(tmp_path):
    ha = np.repeat((1.0 + 0.25 * (np.arange(8) % 4))[:, None], 8, axis=1)   # forest rows 0-2: 4 * (1 + 1.25 + 1.5) = 15 ha
    ok = build(str(tmp_path / 'ok'), {2018: {1: {'cropland': 15.0, 'forest': -15.0}}}, ha=ha)
    run(str(tmp_path / 'ok_project'), ok)
    assert REACHED == [2018]
    short = build(str(tmp_path / 'short'), {2018: {1: {'cropland': 15.5, 'forest': -15.5}}}, ha=ha)
    with pytest.raises(NameError, match=r'region 1 forest loss requested 15.5 ha, available 15 ha'):
        run(str(tmp_path / 'short_project'), short)

def test_enough_physical_stock_but_too_little_eligible_stock_fails(tmp_path):
    """A protection layer covers half the forest: 12 ha are there, 6 ha may be taken."""
    protection = np.zeros((8, 8))
    protection[0:3, 0:2] = 1
    paths = build(str(tmp_path / 'in'), {2018: {1: {'cropland': 8.0, 'forest': -8.0}}}, protection=protection)
    with pytest.raises(NameError, match=r'region 1 forest loss requested 8 ha, available 6 ha'):
        run(str(tmp_path / 'project'), paths)
    t = table(str(tmp_path / 'project'))
    assert t.loc[(1, 'forest'), 'physical_stock_ha'] == 12.0 and REACHED == []

def test_land_changed_in_an_earlier_year_is_locked_and_the_next_year_starts_from_its_map(tmp_path):
    """2018 turns 4 othernat pixels into forest. In 2019 forest holds 16 ha but the 4 new ones are locked."""
    lulc = base_map()
    after = lulc.copy()
    after[5, 0:4] = 4
    cumulative = (after != lulc).astype(np.int32)
    TILES.clear()
    TILES[2018] = (after, cumulative)
    paths = build(str(tmp_path / 'in'), {2018: {1: {'forest': 4.0, 'othernat': -4.0}}, 2019: {1: {'cropland': 13.0, 'forest': -13.0}}},
                  years=(2018, 2019))
    try:
        with pytest.raises(NameError, match=r'2019: .*region 1 forest loss requested 13 ha, available 12 ha'):
            run(str(tmp_path / 'project'), paths)
    finally:
        TILES.clear()
    assert REACHED == [2018]
    t = table(str(tmp_path / 'project'), 2019)
    assert t.loc[(1, 'forest'), 'physical_stock_ha'] == 16.0 and t.loc[(1, 'forest'), 'eligible_stock_ha'] == 12.0

def test_a_positive_residual_othernat_is_a_gain_not_a_negative_stock(tmp_path):
    paths = build(str(tmp_path / 'in'), {2018: {2: {'grassland': -5.0, 'othernat': 5.0}}})
    run(str(tmp_path / 'project'), paths)
    t = table(str(tmp_path / 'project'))
    assert t.loc[(2, 'othernat'), 'check'] == 'gain' and t.loc[(2, 'othernat'), 'available_ha'] == 12.0
    assert REACHED == [2018]

def test_gains_beyond_the_land_they_may_take_fail(tmp_path):
    """Region 1 cropland may take forest and othernat only, 20 ha; water and its own land do not count."""
    paths = build(str(tmp_path / 'in'), {2018: {1: {'cropland': 21.0}}})
    with pytest.raises(NameError, match=r'region 1 cropland gain requested 21 ha, available 20 ha'):
        run(str(tmp_path / 'project'), paths)
    assert REACHED == []

# --- Missing and changed inputs -------------------------------------------------------------------------------------

def test_a_missing_input_is_reported_not_assumed(tmp_path):
    paths = build(str(tmp_path / 'in'), {2018: {1: {'cropland': 3.0, 'forest': -3.0}}})
    os.remove(os.path.join(paths['change_dirs'][2018], 'forest_2018_2017_ha_diff_ssp2_rcp45_luh2-message_bau.tif'))
    with pytest.raises(NameError, match=r'cannot check 2018 without: .*forest_2018_2017_ha_diff'):
        run(str(tmp_path / 'project'), paths)
    assert REACHED == []

@pytest.mark.parametrize('change', ['demand', 'protection'])
def test_unchanged_inputs_reuse_the_table_and_a_changed_input_recomputes_it(tmp_path, change):
    protection = np.zeros((8, 8))
    paths = build(str(tmp_path / 'in'), {2018: {1: {'cropland': 8.0, 'forest': -8.0}}}, protection=protection)
    project = str(tmp_path / 'project')
    run(project, paths)
    path = os.path.join(project, 'intermediate', 'scenario_years', 'ssp2', 'rcp45', 'luh2-message', 'bau', '2018', 'regional_demand_feasibility.csv')
    stamp = os.stat(path).st_mtime_ns
    run(project, paths)
    assert os.stat(path).st_mtime_ns == stamp and REACHED == [2018]
    if change == 'demand':
        build(str(tmp_path / 'in'), {2018: {1: {'cropland': 13.0, 'forest': -13.0}}}, protection=protection)
        message = r'requested 13 ha, available 12 ha'
    else:
        protection[0:3, 0:2] = 1
        build(str(tmp_path / 'in'), {2018: {1: {'cropland': 8.0, 'forest': -8.0}}}, protection=protection)
        message = r'requested 8 ha, available 6 ha'
    with pytest.raises(NameError, match=message):
        run(project, paths)
    assert REACHED == []
