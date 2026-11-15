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

from seals import seals_main
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

def run(root, routed, bottom_forest=False, change=None):
    lulc = region_1(bottom_forest)
    paths = build(os.path.join(root, 'in'), change or TARGETS, lulc=lulc)
    preference_path = os.path.join(root, 'in', 'preference.tif')
    write_raster(preference_path, donor.preference(lulc), 0.25, -9999.0, gdal.GDT_Float64)
    df = pd.read_csv(paths['coefficients'])
    df.loc[df['spatial_regressor_name'] == 'suitability', 'data_location'] = preference_path
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
    p.add_task(donor.kernel_allocation, parent=check)
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
