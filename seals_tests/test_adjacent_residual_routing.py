"""Residual routing: targets that fit stay, and only the excess moves outwards within its region until placed or stuck."""
import itertools

import numpy as np
import pytest

from seals import seals_utils

def route(targets, capacity, regions, weights=None, neighborhood='queen', zero_weight_cells='after_positive'):
    targets = np.asarray(targets, dtype=float)
    if targets.ndim == 2:
        targets = targets[None]
    weights = np.ones_like(targets) if weights is None else np.asarray(weights, dtype=float).reshape(targets.shape)
    return seals_utils.route_residual_to_adjacent_cells(targets, np.asarray(capacity, dtype=float), np.asarray(regions),
                                                        weights, neighborhood, zero_weight_cells)

def region_wide_reweighting(total, weights, capacity):
    """The earlier proposal: spread the whole regional total by weight, capped by capacity, re-spreading the rest."""
    weights, capacity = np.asarray(weights, float), np.asarray(capacity, float)
    target, left = np.zeros_like(weights), float(total)
    while left > 1e-12:
        open_cells = (target < capacity - 1e-12) & (weights > 0)
        if not open_cells.any():
            break
        new = np.minimum(target + np.where(open_cells, weights, 0) / weights[open_cells].sum() * left, capacity)
        left -= (new - target).sum()
        target = new
    return target

def test_targets_that_fit_are_returned_unchanged():
    targets = np.array([[[3.0, 0.0, 2.0]], [[1.0, 4.0, 0.0]]])
    result = route(targets, [[5.0, 4.0, 9.0]], [[1, 1, 1]], weights=np.ones_like(targets))
    assert np.array_equal(result['targets'], targets)
    assert result['unplaced'].sum() == 0 and result['rings'] == 0

def test_local_routing_differs_from_region_wide_reweighting():
    targets = [[10.0, 1.0, 1.0, 1.0, 1.0]]
    capacity = [[4.0, 7.0, 1.0, 1.0, 12.0]]
    weights = [[1.0, 0.1, 1.0, 1.0, 5.0]]
    result = route(targets, capacity, [[1, 1, 1, 1, 1]], weights)
    assert result['targets'][0].tolist() == [[4.0, 7.0, 1.0, 1.0, 1.0]]      # the 6 ha excess goes next door
    assert np.all(result['targets'][0][0, 1:] >= np.asarray(targets)[0, 1:])  # no fitting target is reduced
    reweighted = region_wide_reweighting(14.0, weights[0], capacity[0])
    assert reweighted[4] > 1.0                                              # region-wide sends hectares to the far cell
    assert not np.allclose(reweighted, result['targets'][0][0])

def test_residual_follows_a_chain_of_full_neighbours():
    result = route([[9.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 10.0]], [[1, 1, 1, 1]])
    assert result['targets'][0].tolist() == [[1.0, 0.0, 0.0, 8.0]]
    assert result['rings'] == 3

def test_residual_never_crosses_into_another_region():
    result = route([[9.0, 0.0, 0.0]], [[1.0, 20.0, 20.0]], [[1, 2, 1]])
    assert result['targets'][0].tolist() == [[1.0, 0.0, 0.0]]                # region 2 blocks the only path
    assert result['unplaced'].sum() == 8.0

def test_a_blocked_neighbour_of_the_same_region_passes_the_path_on():
    result = route([[9.0, 0.0, 0.0]], [[1.0, 0.0, 20.0]], [[1, 1, 1]])
    assert result['targets'][0].tolist() == [[1.0, 0.0, 8.0]]
    assert result['rings'] == 2

def test_cells_without_a_region_break_the_path():
    result = route([[9.0, 0.0, 0.0]], [[1.0, 20.0, 20.0]], [[1, 0, 1]])
    assert result['unplaced'].sum() == 8.0

def test_exhaustion_stops_and_reports_what_is_left():
    result = route([[20.0, 1.0, 0.0]], [[5.0, 3.0, 4.0]], [[1, 1, 1]])
    assert result['targets'].sum() == 12.0
    assert result['unplaced'][0].tolist() == [[9.0, 0.0, 0.0]]

def test_a_disconnected_part_of_the_region_is_not_used():
    regions = [[1, 1, 2, 1]]
    result = route([[9.0, 0.0, 0.0, 0.0]], [[1.0, 2.0, 0.0, 50.0]], regions)
    assert result['targets'][0].tolist() == [[1.0, 2.0, 0.0, 0.0]]
    assert result['unplaced'].sum() == 6.0

def test_rook_and_queen_reach_a_diagonal_cell_at_different_rings():
    targets = np.zeros((2, 2)); targets[0, 0] = 5.0
    capacity = np.array([[1.0, 0.0], [0.0, 10.0]])
    regions = np.ones((2, 2), dtype=int)
    queen = route(targets, capacity, regions, neighborhood='queen')
    rook = route(targets, capacity, regions, neighborhood='rook')
    assert queen['targets'][0][1, 1] == 4.0 and queen['rings'] == 1
    assert rook['targets'][0][1, 1] == 4.0 and rook['rings'] == 2

def test_competing_sources_and_classes_share_spare_capacity_and_ignore_order():
    targets = np.zeros((2, 1, 3))
    targets[0, 0, 0] = 6.0
    targets[1, 0, 2] = 2.0
    targets[1, 0, 0] = 2.0
    capacity = np.array([[2.0, 4.0, 0.0]])
    regions = np.ones((1, 3), dtype=int)
    result = route(targets, capacity, regions, weights=np.ones_like(targets))
    assert result['targets'][:, 0, 1].sum() == pytest.approx(4.0)          # the shared neighbour is filled once
    assert result['targets'].sum(axis=0).max() <= 4.0 + 1e-9
    for order in itertools.permutations(range(2)):
        permuted = route(targets[list(order)], capacity, regions, weights=np.ones_like(targets))
        assert np.allclose(permuted['targets'], result['targets'][list(order)])
    flipped = route(targets[:, :, ::-1], capacity[:, ::-1], regions, weights=np.ones_like(targets))
    assert np.allclose(flipped['targets'], result['targets'][:, :, ::-1])

def test_zero_weight_cells_follow_the_chosen_policy():
    targets, capacity, regions = [[6.0, 0.0, 0.0]], [[1.0, 10.0, 10.0]], [[1, 1, 1]]
    weights = [[1.0, 0.0, 1.0]]
    after_positive = route(targets, capacity, regions, weights, zero_weight_cells='after_positive')
    never = route(targets, capacity, regions, weights, zero_weight_cells='never')
    assert after_positive['targets'][0].tolist() == [[1.0, 5.0, 0.0]]       # ring 1 holds only a zero-weight cell
    assert never['targets'][0].tolist() == [[1.0, 0.0, 5.0]]                # skipped, so ring 2 takes it

def test_regional_hectares_are_conserved_and_small_residuals_are_reported():
    result = route([[1.0 + 3e-12, 0.0]], [[1.0, 0.0]], [[1, 1]])
    assert result['unplaced'].sum() == pytest.approx(3e-12, abs=1e-15)     # below the stopping tolerance, still reported
    assert result['targets'].sum() + result['unplaced'].sum() == pytest.approx(1.0 + 3e-12, abs=1e-15)

@pytest.mark.parametrize('bad', ['target', 'capacity', 'weight'])
@pytest.mark.parametrize('value', [np.nan, np.inf, -1.0])
def test_non_finite_or_negative_inputs_raise(bad, value):
    targets, capacity, weights = np.ones((1, 1, 1)), np.ones((1, 1)), np.ones((1, 1, 1))
    {'target': targets, 'capacity': capacity, 'weight': weights}[bad].flat[0] = value
    with pytest.raises(NameError):
        seals_utils.route_residual_to_adjacent_cells(targets, capacity, np.ones((1, 1), dtype=int), weights, 'queen', 'after_positive')

def test_unknown_options_and_targets_outside_regions_raise():
    with pytest.raises(NameError):
        route([[1.0]], [[1.0]], [[1]], neighborhood='hex')
    with pytest.raises(NameError):
        route([[1.0]], [[1.0]], [[1]], zero_weight_cells='sometimes')
    with pytest.raises(NameError):
        route([[1.0, 0.0]], [[1.0, 1.0]], [[0, 1]])

def test_non_finite_pixel_hectares_raise():
    with pytest.raises(NameError):
        seals_utils.sum_fine_classes_to_coarse(np.array([[2, 2]]), np.array([[np.nan, 1.0]]), [2], 1, 255)
