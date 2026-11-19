"""Can existing eligibility inputs stop unintended forest or pasture loss when OtherNat is the intended donor?

Two fixtures, both recording what the original Cython allocator does under each restriction; they change nothing in it.

Kernel alone: three coarse cells of one tile, with an additive preference standing in for LuccME, then the transition
audit's matrix: coarse budget -> realized source and destination hectares per cell. Each cell is its own region here,
so the audit's per-region table is the per-cell table.

Through ProjectFlow: GTAP target -> coarse budgets -> the original Cython allocator -> the transition audit's
source-to-destination hectares.
Real: the stock check, the transition audit, the coefficient preparation allocation_zones uses
(seals_utils.prepare_allocation_coefficients, which rebuilds the presence constraints) and the Cython kernel
seals_allocation_gridded_input with allow_contracting 0. Stand-in: the task that calls the kernel replaces
allocation_zones/allocation, which need a pyramid-compatible grid and calibration blocks a synthetic fixture cannot give;
it loads the same coarse rasters, layers and coefficients and writes the tile allocation would write.

Region 1 (left half): each coarse cell holds 4 forest, 4 pasture (grassland), 5 OtherNat and 3 cropland pixels of 1 ha.
GTAP asks cropland +10 and OtherNat -10, forest and pasture 0: OtherNat is the intended donor. Region 2 (right half)
asks cropland +4 and pasture -4: pasture is the intended donor there, and each of its coarse cells holds pasture.
A preference layer, standing in for LuccME, ranks forest above pasture above OtherNat for every receiver.
Every restriction below is a synthetic fixture setting."""
import os
import tempfile

import hazelbean as hb
import numpy as np
import pandas as pd
import pytest
from osgeo import gdal

from seals import seals_cython_functions, seals_main, seals_utils
import test_regional_demand_feasibility as feasibility
from test_regional_demand_feasibility import ALL_LABELS, LABELS, NDV, build, scenario_years, write_raster

IDS = [1, 2, 3, 4, 5]                        # urban cropland grassland forest othernat
LAB = ['urban', 'cropland', 'grassland', 'forest', 'othernat', 'water']
WATER = 6

def lulc_cells():
    a = np.array([[4, 4, 4, 4], [3, 3, 3, 3], [5, 5, 5, 5], [5, 5, 2, 2]])   # cell A: 4 forest, 4 grass, 6 othernat, 2 crop
    b = np.array([[3, 3, 3, 3], [3, 3, 4, 4], [5, 5, 5, 5], [5, 5, 2, 2]])   # cell B: 6 grass, 2 forest, 6 othernat, 2 crop
    c = np.array([[4, 4, 4, 4], [5, 5, 5, 5], [5, 5, 5, 5], [2, 2, 6, 6]])   # cell C: 4 forest, 8 othernat, 2 crop, 2 water
    return np.hstack([a, b, c])

def suitability(lulc):                        # stand-in for a LuccME preference: forest > grassland > othernat
    return np.select([lulc == 4, lulc == 3, lulc == 5], [3.0, 2.0, 1.0], 0.5)

# Coarse budgets, urban cropland grassland forest othernat. A: OtherNat pays two expanding classes. B: pasture pays
# cropland. C: forest and OtherNat both lose, in a 1:2 ratio.
BUDGETS = {'A': [0, 2, 2, 0, -4], 'B': [0, 2, -2, 0, 0], 'C': [0, 3, 0, -1, -2]}

def coefficients(forbid=()):
    """Rows receive; columns are the urban..othernat and water binaries. A gainer never takes its own class, urban or water."""
    c = np.ones((5, 6))
    for k in range(5):
        c[k, k] = 0
    c[:, 0] = 0
    c[:, 5] = 0
    for gainer, donor in forbid:
        c[LAB.index(gainer), LAB.index(donor)] = 0
    return c

def allocate(lulc, budgets, coef, protection=None):
    layers = [suitability(lulc)] + [(lulc == i).astype(float) for i in IDS + [WATER]]
    full = np.hstack([np.ones((5, 1)), coef])
    types = [2] + [1] * 6
    if protection is not None:
        layers.append(protection.astype(float))
        full = np.hstack([full, np.zeros((5, 1))])
        types.append(1)
    change = np.array([budgets[k] for k in 'ABC'], dtype=float).T.reshape(5, 1, 3)
    out = seals_cython_functions.seals_allocation_gridded_input(
        change, lulc.astype(np.int64), np.array(layers), full, np.array(types, dtype=np.int64), np.ones_like(lulc, dtype=np.int64),
        np.array(IDS, dtype=np.int64), LAB[:5], np.ones(lulc.shape), tempfile.mkdtemp(), 0.0, np.int64(0), '', '')
    return out[0]

def target_sign_protection(lulc, budgets):
    """Protect, in each coarse cell, the pixels of every class whose budget there is not negative."""
    prot = np.zeros(lulc.shape)
    for j, cell in enumerate('ABC'):
        block = lulc[:, 4 * j:4 * j + 4]
        keep = [IDS[k] for k in range(5) if budgets[cell][k] >= 0]
        prot[:, 4 * j:4 * j + 4] = np.isin(block, keep)
    return prot

CASES = {'1 as is': dict(),
         '2 cropland, grassland may not take forest': dict(forbid=[('cropland', 'forest'), ('grassland', 'forest')]),
         '3 also cropland may not take grassland': dict(forbid=[('cropland', 'forest'), ('grassland', 'forest'), ('cropland', 'grassland')]),
         '4 target-sign protection layer': dict(protect=True)}

def realized(case):
    lulc = lulc_cells()
    protection = target_sign_protection(lulc, BUDGETS) if case.get('protect') else None
    after = allocate(lulc, BUDGETS, coefficients(case.get('forbid', ())), protection)
    region = np.kron(np.array([[1, 2, 3]]), np.ones((4, 4), dtype=np.int64))
    matrix = seals_utils.regional_transition_hectares(lulc, after, np.ones(lulc.shape), region, np.ones(lulc.shape, bool), IDS + [WATER], 3)
    return lulc, after, matrix

def cell_error(matrix):
    """Absolute net-minus-budget hectares per cell, summed over classes."""
    return [float(np.abs(matrix[j + 1].sum(axis=0)[:5] - matrix[j + 1].sum(axis=1)[:5] - BUDGETS[cell]).sum()) for j, cell in enumerate('ABC')]

def paid(matrix, cell, donor, gainer):
    return float(matrix['ABC'.index(cell) + 1][LAB.index(donor), LAB.index(gainer)])

# Measured, per cell A, B, C. Kept as evidence of what each restriction achieves, not accepted as a target.
MEASURED_CELL_ERROR = {'1 as is': [8.0, 4.0, 4.0],
                       '2 cropland, grassland may not take forest': [4.0, 0.0, 2.0],
                       '3 also cropland may not take grassland': [0.0, 4.0, 2.0],
                       '4 target-sign protection layer': [0.0, 0.0, 4.0]}

@pytest.mark.parametrize('name', sorted(CASES))
def test_measured_cell_errors_under_each_restriction(name):
    _, _, matrix = realized(CASES[name])
    assert cell_error(matrix) == MEASURED_CELL_ERROR[name]

def test_as_is_every_expanding_class_takes_forest_first():
    _, _, matrix = realized(CASES['1 as is'])
    assert paid(matrix, 'A', 'forest', 'cropland') == 2.0 and paid(matrix, 'A', 'forest', 'grassland') == 2.0
    assert paid(matrix, 'A', 'othernat', 'cropland') == 0.0                    # the intended donor pays nothing

def test_no_global_coefficient_rule_serves_both_an_othernat_cell_and_a_pasture_cell():
    """Forbidding forest moves cropland onto expanding pasture in A; forbidding pasture too breaks B, where pasture is
    the intended donor. Coefficients apply in every cell, so A and B cannot both be right."""
    _, _, two = realized(CASES['2 cropland, grassland may not take forest'])
    assert paid(two, 'A', 'grassland', 'cropland') == 2.0 and cell_error(two)[1] == 0.0
    _, _, three = realized(CASES['3 also cropland may not take grassland'])
    assert cell_error(three)[0] == 0.0 and paid(three, 'B', 'othernat', 'cropland') == 2.0

def test_a_layer_protecting_non_losing_classes_per_cell_spares_forest_and_pasture():
    """A multiplicative layer, built per cell from the signs of the budgets, protects every class not asked to lose.
    It is an option to discuss, not an existing input: SEALS has no per-year layer producer."""
    _, _, matrix = realized(CASES['4 target-sign protection layer'])
    assert cell_error(matrix)[:2] == [0.0, 0.0]
    assert paid(matrix, 'A', 'othernat', 'cropland') == 2.0 and paid(matrix, 'A', 'othernat', 'grassland') == 2.0

@pytest.mark.xfail(strict=True, reason='Open: with two losing classes the split is set by rank, so forest pays 3 ha on a -1 budget '
                                       'and OtherNat 0 on -2. Only an allocator that reads negative budgets can meet it.')
def test_two_losing_classes_pay_their_own_budgets():
    _, _, matrix = realized(CASES['4 target-sign protection layer'])
    assert cell_error(matrix)[2] == 0.0

# --- Through ProjectFlow: two regions with different intended donors ------------------------------------------------

TARGETS = {2018: {1: {'cropland': 10.0, 'othernat': -10.0}, 2: {'cropland': 4.0, 'grassland': -4.0}}}

def donor_map():
    lulc = feasibility.base_map()
    cell = np.array([[4, 4, 4, 4], [3, 3, 3, 3], [5, 5, 5, 5], [5, 2, 2, 2]])
    lulc[0:4, 0:4] = cell
    lulc[4:8, 0:4] = cell
    lulc[4, 4:8] = 3                                   # pasture in both of region 2's coarse cells, so each can pay its 2 ha
    return lulc

def kernel_allocation(p):
    """Stand-in for allocation_zones + allocation on one tile: the real kernel on the inputs allocation reads."""
    if p.run_this:
        lulc_path = p.lulc_simplified_paths[p.key_base_year]
        lulc = hb.as_array(lulc_path).astype(np.int64)
        df = seals_utils.prepare_allocation_coefficients(p, pd.read_csv(p.calibration_parameters_source), p.calibration_parameters_source)
        columns = dict(zip(seals_utils.coefficient_class_labels(df), seals_utils.coefficient_class_columns(df)))
        layers = np.array([hb.as_array(path).astype(np.float64) for path in df['data_location']])
        coef = df[[columns[label] for label in p.changing_class_labels]].to_numpy(np.float64).T
        types = np.array([{'multiplicative': 1, 'additive': 2}[t] for t in df['type']], dtype=np.int64)
        change = np.array([hb.as_array(path) for path in seals_utils.projected_coarse_change_paths(p)], dtype=np.float64)
        valid = (lulc != p.lulc_ndv).astype(np.int64)
        after, _, change_happened = seals_cython_functions.seals_allocation_gridded_input(
            change, lulc, layers, coef, types, valid, np.array(IDS, dtype=np.int64), list(LABELS),
            hb.as_array(p.aoi_ha_per_cell_fine_path).astype(np.float64), p.cur_dir, 0.0, np.int64(0), '', '')
        tile_dir = os.path.join(p.cur_dir_parent_dir, 'allocation_zones', '0_0', 'allocation')
        write_raster(os.path.join(tile_dir, 'lulc_esa_seals7_ssp2_rcp45_luh2-message_bau_%d.tif' % int(p.year)), after.astype(np.int32), 0.25,
                     NDV, gdal.GDT_Int32)
        write_raster(os.path.join(tile_dir, 'cumulative_change_happened.tif'), change_happened.astype(np.int32), 0.25, -9999, gdal.GDT_Int32)
    return p

def run(root, protected=None, layer=None):
    """protected: additional_protected_class_labels. layer: (name, array, per-class coefficients) added as a multiplicative row."""
    lulc = donor_map()
    paths = build(os.path.join(root, 'in'), TARGETS, lulc=lulc)
    preference_path = os.path.join(root, 'in', 'preference.tif')
    write_raster(preference_path, suitability(lulc), 0.25, -9999.0, gdal.GDT_Float64)
    df = pd.read_csv(paths['coefficients'])
    df.loc[df['spatial_regressor_name'] == 'suitability', 'data_location'] = preference_path
    if layer:
        name, array, layer_coefficients = layer
        layer_path = os.path.join(root, 'in', name + '.tif')
        write_raster(layer_path, array.astype(np.float64), 0.25, -9999.0, gdal.GDT_Float64)
        df = pd.concat([df, pd.DataFrame([dict(spatial_regressor_name=name, data_location=layer_path, type='multiplicative', **layer_coefficients)])])
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
    p.additional_protected_class_labels = protected
    p.lulc_ndv = NDV
    p.lulc_src_label, p.lulc_simplification_label = 'esa', 'seals7'
    p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label = 'ssp2', 'rcp45', 'luh2-message', 'bau'
    p.regional_projections_input_path = 'regional_demand.csv'
    p.years = p.feasibility_test_years = paths['years']
    p.feasibility_test_change_dirs = paths['change_dirs']
    iterator = p.add_iterator(scenario_years)
    check = p.add_iterator(seals_main.regional_demand_feasibility, parent=iterator, creates_dir=False, skip_existing=0)
    p.add_task(kernel_allocation, parent=check)
    p.add_task(seals_main.regional_transitions, parent=check, skip_existing=0)
    p.execute()
    d = os.path.join(root, 'project', 'intermediate', 'scenario_years', 'ssp2', 'rcp45', 'luh2-message', 'bau', '2018', 'regional_transitions')
    transitions = pd.read_csv(os.path.join(d, 'regional_transitions.csv'), encoding='utf-8-sig')
    summary = pd.read_csv(os.path.join(d, 'regional_transitions_summary.csv'), encoding='utf-8-sig').set_index(['region_id', 'class_label'])
    return transitions, summary

def flow(transitions, region_id, source, destination):
    rows = transitions[(transitions.region_id == region_id) & (transitions.from_class_label == source) & (transitions.to_class_label == destination)]
    return float(rows['ha'].sum())

def target_sign_layer(lulc):
    """Per coarse cell, the pixels of every class whose budget there is not negative (this fixture's budgets)."""
    keep = {1: [1, 2, 3, 4], 2: [1, 2, 5]}            # region 1 may lose only OtherNat, region 2 only pasture
    region = np.kron(np.array([[1, 2], [1, 2]]), np.ones((4, 4), dtype=np.int64))
    return np.where(region == 1, np.isin(lulc, keep[1]), np.isin(lulc, keep[2]))

# --- What each existing restriction achieves (measured; evidence, not targets) -------------------------------------

def test_as_is_cropland_takes_forest_and_pasture_and_othernat_pays_nothing(tmp_path):
    transitions, summary = run(str(tmp_path))
    assert summary.loc[(1, 'cropland'), 'intended_net_ha'] == 10.0 and summary.loc[(1, 'cropland'), 'realized_net_ha'] == 10.0
    assert flow(transitions, 1, 'forest', 'cropland') == 8.0 and flow(transitions, 1, 'grassland', 'cropland') == 2.0
    assert flow(transitions, 1, 'othernat', 'cropland') == 0.0
    assert summary.loc[(1, 'othernat'), 'realized_minus_intended_ha'] == 10.0     # the intended donor kept all 10 ha

def test_protecting_forest_and_pasture_is_refused_because_region_2_needs_pasture(tmp_path):
    """additional_protected_class_labels is per scenario, so it applies in both regions: region 2's intended donor is
    then off limits, and the stock check refuses before allocating instead of taking OtherNat there."""
    with pytest.raises(NameError, match=r'region 2 grassland loss requested 4 ha, available 0 ha'):
        run(str(tmp_path), protected='forest grassland')

def test_a_mask_stopping_only_cropland_from_forest_and_pasture_is_refused_for_region_2(tmp_path):
    """A scenario multiplicative row survives the presence-constraint rebuild and can name one receiver, but it covers
    every pixel it is drawn over: drawn over all forest and pasture it leaves region 2's cropland no pasture to take."""
    mask = np.isin(donor_map(), [3, 4])
    with pytest.raises(NameError, match=r'region 2 grassland loss requested 4 ha, available 0 ha'):
        run(str(tmp_path), layer=('crop_off_forest_pasture', mask, {l: (0.0 if l == 'cropland' else 1.0) for l in LABELS}))

def test_the_same_mask_drawn_over_region_1_only_meets_both_regions(tmp_path):
    """Synthetic: drawn only where OtherNat is the intended donor. A mask that follows the regions' donors is a donor
    policy written as a layer; nothing in SEALS produces it."""
    mask = np.isin(donor_map(), [3, 4]) & (np.kron(np.array([[1, 2], [1, 2]]), np.ones((4, 4), dtype=np.int64)) == 1)
    transitions, _ = run(str(tmp_path), layer=('crop_off_forest_pasture', mask, {l: (0.0 if l == 'cropland' else 1.0) for l in LABELS}))
    assert flow(transitions, 1, 'othernat', 'cropland') == 10.0 and flow(transitions, 2, 'grassland', 'cropland') == 4.0

def test_a_mask_drawn_from_the_budget_signs_meets_both_regions(tmp_path):
    """Synthetic: a layer that, per cell, protects the classes not asked to lose. SEALS has no producer for it."""
    transitions, summary = run(str(tmp_path), layer=('target_sign_protection', target_sign_layer(donor_map()), {l: 0.0 for l in LABELS}))
    assert flow(transitions, 1, 'othernat', 'cropland') == 10.0 and flow(transitions, 2, 'grassland', 'cropland') == 4.0
    changing = summary.loc[summary.index.get_level_values('class_label').isin(LABELS)]
    assert changing['realized_minus_intended_ha'].fillna(0).abs().max() == 0.0
