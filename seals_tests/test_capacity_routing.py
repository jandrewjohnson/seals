"""Capacity routing alone: the same targets, preferences, coefficients and kernel, with and without adjacent_coarse_budgets.

Both runs go through the stock check, the real Cython kernel on the coefficients allocation_zones prepares (the
kernel_allocation stand-in for allocation_zones, see test_donor_restrictions_projectflow) and the transition audit. The
routed run adds adjacent_coarse_budgets before the kernel. No donor rule, mask or coefficient differs between the runs.

Region 1 asks cropland +8 and OtherNat -8, split evenly over its two coarse cells, but its top cell holds no OtherNat:
the coarse budget asks that cell to lose land it does not have. Region 2 asks nothing."""
import os
import time

import numpy as np
import pandas as pd
import pytest
import hazelbean as hb
from osgeo import gdal

from seals import seals_cython_functions, seals_main, seals_utils
import test_donor_restrictions_projectflow as donor
import test_regional_demand_feasibility as feasibility
from test_regional_demand_feasibility import ALL_LABELS, IDS, LABELS, NDV, build, scenario_years, write_raster

TARGETS = {2018: {1: {'cropland': 8.0, 'othernat': -8.0}}}
YEAR_DIR = ('project', 'intermediate', 'scenario_years', 'ssp2', 'rcp45', 'luh2-message', 'bau', '2018')

def region_1(bottom_forest):
    lulc = feasibility.base_map()
    lulc[0:2, 0:4] = 4                 # top cell: 8 forest, 8 cropland, no OtherNat
    lulc[2:4, 0:4] = 2
    lulc[4:7, 0:4] = 5                 # bottom cell: 12 OtherNat, 4 cropland
    lulc[7, 0:4] = 2
    if bottom_forest:
        lulc[4, 0:4] = 4               # bottom cell: 4 forest, ranked first, and 8 OtherNat
    return lulc

def kernel_allocation_accepted(p):
    """Stand-in for allocation_zones like donor.kernel_allocation, but starting from the accepted map: the base map
    with earlier years' tiles over it, and pixels changed before locked, as allocation does from year to year."""
    if p.run_this:
        lulc_path = p.lulc_simplified_paths[p.key_base_year]
        base = hb.as_array(lulc_path).astype(np.int64)
        years = [int(y) for y in p.years]
        scenario_dir = os.path.dirname(p.cur_dir_parent_dir)
        tile_name = 'lulc_esa_seals7_ssp2_rcp45_luh2-message_bau_{}.tif'
        first = int(p.previous_year) == int(p.key_base_year)
        tiles = seals_utils.earlier_allocation_tiles(scenario_dir, tile_name, [y for y in years if y < int(p.year)], None if first else p.previous_year)
        lulc, locked = base.copy(), np.zeros(base.shape, dtype=bool)
        seals_utils.paste_allocation_tiles(lulc, locked, 0, seals_utils.allocation_tile_windows(tiles, hb.get_raster_info_hb(lulc_path)['geotransform']))
        df = seals_utils.prepare_allocation_coefficients(p, pd.read_csv(p.calibration_parameters_source), p.calibration_parameters_source)
        columns = dict(zip(seals_utils.coefficient_class_labels(df), seals_utils.coefficient_class_columns(df)))
        layers = np.array([hb.as_array(path).astype(np.float64) for path in df['data_location']])
        coefficients = df[[columns[label] for label in p.changing_class_labels]].to_numpy(np.float64).T
        types = np.array([{'multiplicative': 1, 'additive': 2}[t] for t in df['type']], dtype=np.int64)
        change = np.array([hb.as_array(path) for path in seals_utils.projected_coarse_change_paths(p)], dtype=np.float64)
        valid = ((base != p.lulc_ndv) & ~locked).astype(np.int64)
        after, _, change_happened = seals_cython_functions.seals_allocation_gridded_input(
            change, lulc, layers, coefficients, types, valid, np.array(IDS, dtype=np.int64), list(LABELS),
            hb.as_array(p.aoi_ha_per_cell_fine_path).astype(np.float64), p.cur_dir, 0.0, np.int64(0), '', '')
        tile_dir = os.path.join(p.cur_dir_parent_dir, 'allocation_zones', '0_0', 'allocation')
        write_raster(os.path.join(tile_dir, tile_name.format(int(p.year))), after.astype(np.int32), 0.25, NDV, gdal.GDT_Int32)
        write_raster(os.path.join(tile_dir, 'cumulative_change_happened.tif'), (locked | (change_happened == 1)).astype(np.int32), 0.25, -9999,
                     gdal.GDT_Int32)
    return p

def run(root, routed, bottom_forest=False, change=None, lulc=None, ha=None, layer=None, years=(2018,), accepted=False):
    lulc = region_1(bottom_forest) if lulc is None else lulc
    paths = build(os.path.join(root, 'in'), change or TARGETS, lulc=lulc, ha=ha, years=years)
    preference_path = os.path.join(root, 'in', 'preference.tif')
    write_raster(preference_path, donor.preference(lulc), 0.25, -9999.0, gdal.GDT_Float64)
    df = pd.read_csv(paths['coefficients'])
    df.loc[df['spatial_regressor_name'] == 'suitability', 'data_location'] = preference_path
    if layer:
        name, array, coefficients = layer
        layer_path = os.path.join(root, 'in', name + '.tif')
        write_raster(layer_path, array.astype(np.float64), 0.25, -9999.0, gdal.GDT_Float64)
        df = pd.concat([df, pd.DataFrame([dict(spatial_regressor_name=name, data_location=layer_path, type='multiplicative', **coefficients)])])
    df.to_csv(paths['coefficients'], index=False)
    p = hb.ProjectFlow(project_dir=os.path.join(root, 'project'))
    p.key_base_year = 2017
    p.lulc_simplified_paths = {2017: paths['lulc']}
    p.aoi_ha_per_cell_fine_path = paths['ha']
    p.regional_change_dir = paths['regional_change_dir']
    p.fine_processed_inputs_dir = paths['fine_processed_inputs_dir']
    p.calibration_parameters_source = paths['coefficients']
    p.changing_class_indices, p.changing_class_labels, p.all_class_labels = IDS, LABELS, ALL_LABELS
    p.all_class_indices = list(range(1, len(ALL_LABELS) + 1))
    p.lulc_ndv = NDV
    p.lulc_src_label, p.lulc_simplification_label = 'esa', 'seals7'
    p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label = 'ssp2', 'rcp45', 'luh2-message', 'bau'
    p.regional_projections_input_path = 'regional_demand.csv'
    p.years = p.feasibility_test_years = paths['years']
    p.feasibility_test_change_dirs = paths['change_dirs']
    iterator = p.add_iterator(scenario_years)
    check = p.add_iterator(seals_main.regional_demand_feasibility, parent=iterator, creates_dir=False, skip_existing=0)
    if routed:
        p.add_task(seals_main.adjacent_coarse_budgets, parent=check, skip_existing=0)
    p.add_task(kernel_allocation_accepted if accepted else donor.kernel_allocation, parent=check)
    p.add_task(seals_main.regional_transitions, parent=check, skip_existing=0)
    start = time.perf_counter()
    p.execute()
    elapsed = time.perf_counter() - start
    d = os.path.join(root, *YEAR_DIR, 'regional_transitions')
    transitions = pd.read_csv(os.path.join(d, 'regional_transitions.csv'), encoding='utf-8-sig')
    summary = pd.read_csv(os.path.join(d, 'regional_transitions_summary.csv'), encoding='utf-8-sig').set_index(['region_id', 'class_label'])
    return transitions, summary, elapsed

def flow(transitions, source, destination):
    return donor.flow(transitions, 1, source, destination)

def realized_error(summary):
    """Region 1's realized minus GTAP target per changing class (the audit's intended net is the unrouted target here)."""
    return {label: float(summary.loc[(1, label), 'realized_net_ha']) - TARGETS[2018][1].get(label, 0.0) for label in LABELS}

def test_standard_allocation_takes_forest_where_the_cell_has_no_othernat(tmp_path):
    transitions, summary, _ = run(str(tmp_path), routed=False)
    assert flow(transitions, 'forest', 'cropland') == 4.0 and flow(transitions, 'othernat', 'cropland') == 4.0
    assert realized_error(summary) == {'urban': 0.0, 'cropland': 0.0, 'grassland': 0.0, 'forest': -4.0, 'othernat': 4.0}

def test_routing_moves_the_unpayable_budget_to_the_cell_that_holds_othernat(tmp_path):
    transitions, summary, _ = run(str(tmp_path), routed=True)
    budgets = pd.read_csv(os.path.join(str(tmp_path), *YEAR_DIR, 'adjacent_coarse_budgets', 'adjacent_coarse_budgets.csv'),
                          encoding='utf-8-sig').set_index(['region_id', 'class_label'])
    assert budgets.loc[(1, 'cropland'), 'coarse_input_ha'] == 8.0 and budgets.loc[(1, 'cropland'), 'budget_ha'] == 8.0
    assert budgets[['unplaced_gain_ha', 'unplaced_loss_ha']].to_numpy().max() == 0
    assert budgets.loc[(1, 'othernat'), 'cells_changed'] == 2 and budgets.loc[(1, 'cropland'), 'cells_changed'] == 2
    assert flow(transitions, 'othernat', 'cropland') == 8.0 and flow(transitions, 'forest', 'cropland') == 0.0
    assert all(v == 0.0 for v in realized_error(summary).values())

def test_routing_does_not_choose_the_donor_where_the_target_cell_also_holds_forest(tmp_path):
    """Capacity only: once both classes are in the receiving cell, the kernel still takes the better-ranked forest."""
    _, standard, _ = run(str(tmp_path / 'standard'), routed=False, bottom_forest=True)
    transitions, routed, _ = run(str(tmp_path / 'routed'), routed=True, bottom_forest=True)
    assert realized_error(standard)['forest'] == -8.0 and realized_error(standard)['othernat'] == 8.0
    assert flow(transitions, 'forest', 'cropland') == 4.0 and flow(transitions, 'othernat', 'cropland') == 4.0
    assert realized_error(routed)['forest'] == -4.0 and realized_error(routed)['othernat'] == 4.0

@pytest.mark.parametrize('routed', [False, True])
def test_land_is_conserved_and_no_stock_goes_negative(tmp_path, routed):
    _, summary, _ = run(str(tmp_path), routed=routed)
    for region_id in (1, 2):
        rows = summary.loc[region_id]
        assert rows['before_ha'].sum() == rows['after_ha'].sum()
    assert summary['after_ha'].min() >= 0

def test_an_unplaceable_budget_is_refused_not_shrunk(tmp_path):
    """Region 1 asks more cropland than it loses: the routed cells cannot pay the extra and nothing is allocated."""
    with pytest.raises(NameError, match='region 1 cropland gain 2 ha, loss 0 ha unplaced'):
        run(str(tmp_path), routed=True, change={2018: {1: {'cropland': 10.0, 'othernat': -8.0}}})

def test_an_unchanged_rerun_reuses_the_routed_budgets_and_a_changed_demand_recomputes_them(tmp_path):
    root = str(tmp_path)
    _, _, first = run(root, routed=True)
    table = os.path.join(root, *YEAR_DIR, 'adjacent_coarse_budgets', 'adjacent_coarse_budgets.csv')
    stamp = os.stat(table).st_mtime_ns
    _, _, repeat = run(root, routed=True)
    assert os.stat(table).st_mtime_ns == stamp
    with open(os.path.join(root, 'timing.txt'), 'w') as f:
        f.write('first %.4f s, cached repeat %.4f s\n' % (first, repeat))
    _, summary, _ = run(root, routed=True, change={2018: {1: {'cropland': 6.0, 'othernat': -6.0}}})
    assert os.stat(table).st_mtime_ns != stamp
    assert summary.loc[(1, 'cropland'), 'realized_net_ha'] == 6.0

def protected_top_othernat():
    """Top cell: 8 forest, 4 OtherNat under a protection mask, 4 cropland. Bottom cell: 12 OtherNat, 4 cropland."""
    lulc = region_1(False)
    lulc[2, 0:4] = 5
    mask = np.zeros(lulc.shape)
    mask[2, 0:4] = 1
    return lulc, ('protection', mask, {l: 0.0 for l in LABELS})

def test_protected_othernat_in_a_cell_does_not_count_as_capacity(tmp_path):
    """Open stock would keep the top cell's OtherNat budget there; the mask makes it unlosable, so routing moves it."""
    lulc, layer = protected_top_othernat()
    _, standard, _ = run(str(tmp_path / 'standard'), routed=False, lulc=lulc, layer=layer)
    transitions, routed, _ = run(str(tmp_path / 'routed'), routed=True, lulc=lulc, layer=layer)
    assert realized_error(standard)['forest'] == -4.0
    assert flow(transitions, 'othernat', 'cropland') == 8.0 and flow(transitions, 'forest', 'cropland') == 0.0
    top_othernat = hb.as_array(os.path.join(str(tmp_path / 'routed'), *YEAR_DIR, 'adjacent_coarse_budgets',
                                            'othernat_2018_2017_ha_diff_ssp2_rcp45_luh2-message_bau.tif'))[0, 0]
    assert top_othernat == 0.0

def test_a_gain_routed_to_a_cell_its_class_may_not_take_is_refused(tmp_path):
    """Cropland +8 and pasture +2 from OtherNat -10. The region as a whole has the land (the stock check passes: cropland
    may take the top cell's forest, pasture the bottom cell's OtherNat), but routing sends both gains to the bottom cell,
    where a mask lets pasture take OtherNat and not cropland. The routed cell cannot hold cropland's gain: refused."""
    lulc = region_1(False)
    mask = np.zeros(lulc.shape)
    mask[4:7, 0:4] = 1
    change = {2018: {1: {'cropland': 8.0, 'grassland': 2.0, 'othernat': -10.0}}}
    with pytest.raises(NameError, match=r'exceed the land the gaining classes may take.*region 1 cropland 8 ha above its land'):
        run(str(tmp_path), routed=True, change=change, layer=('crop_off_othernat', mask, {l: (0.0 if l == 'cropland' else 1.0) for l in LABELS}))

def test_unequal_pixel_areas_route_hectares_and_spare_forest(tmp_path):
    ha = np.repeat((1.0 + 0.25 * (np.arange(8) % 4))[:, None], 8, axis=1)
    transitions, _, _ = run(str(tmp_path), routed=True, ha=ha)
    budgets = pd.read_csv(os.path.join(str(tmp_path), *YEAR_DIR, 'adjacent_coarse_budgets', 'adjacent_coarse_budgets.csv'),
                          encoding='utf-8-sig').set_index(['region_id', 'class_label'])
    assert budgets.loc[(1, 'cropland'), 'budget_ha'] == pytest.approx(8.0)
    assert flow(transitions, 'forest', 'cropland') == 0.0
    assert 8.0 <= flow(transitions, 'othernat', 'cropland') < 8.0 + ha.max()      # whole pixels, at most one over

def test_two_years_route_on_the_accepted_map(tmp_path):
    """Each year asks cropland +4 from OtherNat. 2019 starts from 2018's map with 2018's changes locked."""
    change = {2018: {1: {'cropland': 4.0, 'othernat': -4.0}}, 2019: {1: {'cropland': 4.0, 'othernat': -4.0}}}
    run(str(tmp_path), routed=True, change=change, years=(2018, 2019), accepted=True)
    for year, cropland_before in ((2018, 12.0), (2019, 16.0)):
        d = os.path.join(str(tmp_path), 'project', 'intermediate', 'scenario_years', 'ssp2', 'rcp45', 'luh2-message', 'bau', str(year))
        transitions = pd.read_csv(os.path.join(d, 'regional_transitions', 'regional_transitions.csv'), encoding='utf-8-sig')
        summary = pd.read_csv(os.path.join(d, 'regional_transitions', 'regional_transitions_summary.csv'),
                              encoding='utf-8-sig').set_index(['region_id', 'class_label'])
        assert summary.loc[(1, 'cropland'), 'before_ha'] == cropland_before
        assert donor.flow(transitions, 1, 'othernat', 'cropland') == 4.0 and donor.flow(transitions, 1, 'forest', 'cropland') == 0.0
        assert summary.loc[(1, 'othernat'), 'intended_net_ha'] == -4.0
