"""The transition audit, run through ProjectFlow after a stand-in for allocation_zones, on the stock-check fixture.

Region 1 is the left half of the map (forest, cropland, othernat, water), region 2 the right half (grassland, othernat).
The stand-in writes each year's tile, as allocation does; the audit reads only maps and reports."""
import os
import numpy as np
import pandas as pd
import pytest
import hazelbean as hb

from seals import seals_initialize_project, seals_main, seals_utils
import test_regional_demand_feasibility as feasibility
from test_regional_demand_feasibility import ALL_LABELS, IDS, LABELS, NDV, TILES, base_map, build, scenario_years

def run(project_dir, paths, tiles, rules_path=None):
    TILES.clear()
    TILES.update(tiles)
    del feasibility.REACHED[:]
    p = hb.ProjectFlow(project_dir=project_dir)
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
    p.regional_transitions_allowed_path = rules_path
    p.years = paths['years']
    p.feasibility_test_years = paths['years']
    p.feasibility_test_change_dirs = paths['change_dirs']
    iterator = p.add_iterator(scenario_years)
    check = p.add_iterator(seals_main.regional_demand_feasibility, parent=iterator, creates_dir=False, skip_existing=0)
    p.add_task(feasibility.allocation_stand_in, parent=check)
    p.add_task(seals_main.regional_transitions, parent=check, skip_existing=0)
    try:
        p.execute()
    finally:
        TILES.clear()
    return p

def year_dir(project_dir, year):
    return os.path.join(project_dir, 'intermediate', 'scenario_years', 'ssp2', 'rcp45', 'luh2-message', 'bau', str(year), 'regional_transitions')

def tables(project_dir, year=2018):
    d = year_dir(project_dir, year)
    transitions = pd.read_csv(os.path.join(d, 'regional_transitions.csv'), encoding='utf-8-sig')
    summary = pd.read_csv(os.path.join(d, 'regional_transitions_summary.csv'), encoding='utf-8-sig').set_index(['region_id', 'class_label'])
    return transitions, summary

def moved(transitions, region_id, source, destination):
    rows = transitions[(transitions.region_id == region_id) & (transitions.from_class_label == source) & (transitions.to_class_label == destination)]
    return float(rows['ha'].sum())

def crop_expansion():
    """2018: region 1 cropland takes 3 forest pixels and 2 othernat pixels."""
    lulc = base_map()
    after = lulc.copy()
    after[0, 0:3] = 2
    after[5, 0:2] = 2
    return lulc, after

# --- Unit: the transition matrix ------------------------------------------------------------------------------------

def test_matrix_rows_are_before_stocks_columns_after_stocks_and_their_difference_is_the_net():
    before = np.array([[1, 1, 2, 2]])
    after = np.array([[1, 2, 2, 3]])
    ha = np.array([[1.0, 2.0, 3.0, 4.0]])
    region = np.array([[1, 1, 1, 0]])
    matrix = seals_utils.regional_transition_hectares(before, after, ha, region, np.ones_like(before, bool), [1, 2, 3], 1)
    assert matrix[1].tolist() == [[1.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]]   # the region-0 pixel is left out
    assert (matrix[1].sum(axis=0) - matrix[1].sum(axis=1)).tolist() == [-2.0, 2.0, 0.0]
    with pytest.raises(NameError, match='no label: \\[9\\]'):
        seals_utils.regional_transition_hectares(before, np.array([[1, 9, 2, 3]]), ha, region, np.ones_like(before, bool), [1, 2, 3], 1)

def test_a_matrix_that_loses_land_is_refused():
    matrix = np.zeros((2, 2, 2))
    matrix[1] = [[3.0, 1.0], [0.0, 2.0]]
    with pytest.raises(NameError, match='does not reproduce the map stocks'):
        seals_utils.summarize_regional_transitions(matrix, np.array([[0, 0], [4.0, 3.0]]), np.array([[0, 0], [3.0, 3.0]]), {}, ['a', 'b'])

# --- Through ProjectFlow --------------------------------------------------------------------------------------------

def test_standard_tree_audits_each_year_after_its_zones_are_allocated(tmp_path):
    p = hb.ProjectFlow(project_dir=str(tmp_path))
    p.run_in_parallel = 0
    seals_initialize_project.build_standard_task_tree(p)
    assert [c.name for c in p.regional_demand_feasibility_task.children] == ['allocation_zones', 'regional_transitions']

def test_no_change_keeps_every_hectare_on_the_diagonal(tmp_path):
    lulc = base_map()
    paths = build(str(tmp_path / 'in'), {2018: {}})
    run(str(tmp_path / 'project'), paths, {2018: (lulc, np.zeros_like(lulc))})
    transitions, summary = tables(str(tmp_path / 'project'))
    assert transitions['unchanged'].all()
    assert summary['gross_gain_ha'].abs().max() == 0 and summary['gross_loss_ha'].abs().max() == 0
    assert summary.loc[(1, 'forest'), 'before_ha'] == 12.0 and summary.loc[(1, 'water'), 'unchanged_ha'] == 4.0

def test_forest_to_cropland_and_othernat_to_cropland_are_reported_apart(tmp_path):
    lulc, after = crop_expansion()
    paths = build(str(tmp_path / 'in'), {2018: {1: {'cropland': 5.0, 'forest': -5.0}}})
    run(str(tmp_path / 'project'), paths, {2018: (after, (after != lulc).astype(int))})
    transitions, summary = tables(str(tmp_path / 'project'))
    assert moved(transitions, 1, 'forest', 'cropland') == 3.0 and moved(transitions, 1, 'othernat', 'cropland') == 2.0
    assert summary.loc[(1, 'cropland'), 'gross_gain_ha'] == 5.0 and summary.loc[(1, 'cropland'), 'realized_net_ha'] == 5.0
    # GTAP asked forest to pay all 5 ha: the audit shows 2 ha came from othernat instead.
    assert summary.loc[(1, 'forest'), 'realized_minus_intended_ha'] == 2.0
    assert summary.loc[(1, 'othernat'), 'realized_minus_intended_ha'] == -2.0
    assert np.isnan(summary.loc[(1, 'water'), 'intended_net_ha'])                 # no target for a non-changing class
    assert summary.loc[2, 'gross_gain_ha'].sum() == 0                             # nothing crosses into region 2

def test_unequal_pixel_areas_are_counted_in_hectares(tmp_path):
    lulc, after = crop_expansion()
    ha = np.repeat((1.0 + 0.25 * (np.arange(8) % 4))[:, None], 8, axis=1)       # row 0 is 1 ha, row 5 is 1.25 ha
    paths = build(str(tmp_path / 'in'), {2018: {1: {'cropland': 5.5, 'forest': -5.5}}}, ha=ha)
    run(str(tmp_path / 'project'), paths, {2018: (after, (after != lulc).astype(int))})
    transitions, summary = tables(str(tmp_path / 'project'))
    assert moved(transitions, 1, 'forest', 'cropland') == 3.0 and moved(transitions, 1, 'othernat', 'cropland') == 2.5
    assert summary.loc[(1, 'cropland'), 'after_ha'] - summary.loc[(1, 'cropland'), 'before_ha'] == pytest.approx(5.5)

def test_nodata_is_kept_out_of_the_matrix_and_reported_when_it_changes(tmp_path):
    lulc = base_map()
    lulc[7, 7] = NDV                                                              # region 2, outside the footprint from the start
    after = lulc.copy()
    after[0, 7] = NDV                                                             # a grassland pixel loses its data
    paths = build(str(tmp_path / 'in'), {2018: {}}, lulc=lulc)
    run(str(tmp_path / 'project'), paths, {2018: (after, np.zeros_like(lulc))})
    transitions, summary = tables(str(tmp_path / 'project'))
    assert summary.loc[(2, 'grassland'), 'nodata_changed_ha'] == 1.0
    assert summary.loc[(2, 'grassland'), 'before_ha'] == 11.0 and summary.loc[(2, 'grassland'), 'after_ha'] == 11.0
    assert summary.loc[(2, 'othernat'), 'before_ha'] == 19.0

def test_a_class_id_with_no_label_is_logged_and_does_not_stop_the_run(tmp_path):
    lulc = base_map()
    after = lulc.copy()
    after[0, 0] = 99
    paths = build(str(tmp_path / 'in'), {2018: {}})
    run(str(tmp_path / 'project'), paths, {2018: (after, (after != lulc).astype(int))})   # does not raise
    assert not os.path.exists(os.path.join(year_dir(str(tmp_path / 'project'), 2018), 'regional_transitions.csv'))

def test_float_error_that_grows_with_the_stock_is_not_a_lost_hectare():
    matrix = np.zeros((2, 1, 1))
    matrix[1, 0, 0] = 1e9
    stock = np.array([[0.0], [1e9 + 1e-3]])
    seals_utils.summarize_regional_transitions(matrix, stock, stock, {}, ['a'])

def test_explicit_donor_rules_flag_disallowed_hectares_and_none_are_assumed(tmp_path):
    lulc, after = crop_expansion()
    rules = [{'from_class_label': a, 'to_class_label': b, 'allowed': int(not (a == 'othernat' and b == 'cropland'))}
             for a in ALL_LABELS for b in ALL_LABELS if a != b]
    rules_path = str(tmp_path / 'allowed.csv')
    pd.DataFrame(rules).to_csv(rules_path, index=False)
    paths = build(str(tmp_path / 'in'), {2018: {1: {'cropland': 5.0, 'forest': -5.0}}})
    run(str(tmp_path / 'with_rules'), paths, {2018: (after, (after != lulc).astype(int))}, rules_path=rules_path)
    _, summary = tables(str(tmp_path / 'with_rules'))
    assert summary.loc[(1, 'cropland'), 'disallowed_gain_ha'] == 2.0
    run(str(tmp_path / 'no_rules'), paths, {2018: (after, (after != lulc).astype(int))})
    _, summary = tables(str(tmp_path / 'no_rules'))
    assert summary['disallowed_gain_ha'].isna().all()
    pd.DataFrame(rules[1:]).to_csv(rules_path, index=False)
    with pytest.raises(NameError, match='do not cover every pair'):
        run(str(tmp_path / 'partial_rules'), paths, {2018: (after, (after != lulc).astype(int))}, rules_path=rules_path)

def test_the_next_year_starts_from_the_accepted_map_not_the_base_map(tmp_path):
    """2019 writes no tile, as for a zone with no change: its starting map is 2018's, so nothing moves in 2019."""
    lulc, after = crop_expansion()
    paths = build(str(tmp_path / 'in'), {2018: {1: {'cropland': 5.0, 'forest': -5.0}}, 2019: {}}, years=(2018, 2019))
    run(str(tmp_path / 'project'), paths, {2018: (after, (after != lulc).astype(int))})
    transitions, summary = tables(str(tmp_path / 'project'), 2019)
    assert transitions['unchanged'].all()
    assert summary.loc[(1, 'cropland'), 'before_ha'] == 13.0 and summary.loc[(1, 'forest'), 'before_ha'] == 9.0

def test_the_audit_changes_no_map_and_reuses_its_tables_only_while_inputs_are_unchanged(tmp_path):
    lulc, after = crop_expansion()
    paths = build(str(tmp_path / 'in'), {2018: {1: {'cropland': 5.0, 'forest': -5.0}}})
    project = str(tmp_path / 'project')
    run(project, paths, {2018: (after, (after != lulc).astype(int))})
    tile = os.path.join(project, 'intermediate', 'scenario_years', 'ssp2', 'rcp45', 'luh2-message', 'bau', '2018', 'allocation_zones', '0_0',
                        'allocation', 'lulc_esa_seals7_ssp2_rcp45_luh2-message_bau_2018.tif')
    assert np.array_equal(hb.as_array(tile), after) and np.array_equal(hb.as_array(paths['lulc']), lulc)
    summary_path = os.path.join(year_dir(project, 2018), 'regional_transitions_summary.csv')
    stamp = os.stat(summary_path).st_mtime_ns
    run(project, paths, {2018: (after, (after != lulc).astype(int))})          # the tile is rewritten with the same bytes
    assert os.stat(summary_path).st_mtime_ns == stamp
    after2 = after.copy()
    after2[1, 0] = 2                                                              # one more forest pixel to cropland
    paths = build(str(tmp_path / 'in'), {2018: {1: {'cropland': 6.0, 'forest': -6.0}}})
    run(project, paths, {2018: (after2, (after2 != lulc).astype(int))})
    transitions, _ = tables(project)
    assert moved(transitions, 1, 'forest', 'cropland') == 4.0
