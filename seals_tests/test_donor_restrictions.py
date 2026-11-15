"""Can existing eligibility inputs stop unintended forest or pasture loss when OtherNat is the intended donor?

The original Cython allocator on three coarse cells of one tile, with an additive preference standing in for LuccME,
then the transition audit's matrix: coarse budget -> realized source and destination hectares per cell. Each cell is
its own region here, so the audit's per-region table is the per-cell table. These record what the allocator does under
each restriction; they change nothing in it."""
import tempfile

import numpy as np
import pytest

from seals import seals_cython_functions, seals_utils

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
