import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'zone_change_budgets', Path(__file__).parents[1] / 'seals/zone_change_budgets.py')
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)
CODES = [1, 2, 3, 4, 5]


def test_balances_signed_changes_without_mutating_requests():
    before = np.array([[5, 5, 5, 5], [4, 4, 2, 2]])
    request = np.array([0., 2., 1., -1., 999.])[:, None, None]
    saved = request.copy()
    target, audit = budget.strict_othernat_targets(
        request, before, np.ones_like(before), np.ones_like(before), CODES, 4)
    np.testing.assert_array_equal(request, saved)
    np.testing.assert_array_equal(target[:, 0, 0], [0, 2, 1, -1, -2])
    assert audit[-1]['available_ha'] == 4
    after = np.array([[2, 3, 5, 5], [2, 4, 2, 2]])
    records = budget.audit_strict_allocation(target, before, after, np.ones_like(before),
                                            np.ones_like(before), CODES)
    assert all(row['residual_ha'] == 0 for row in records)


@pytest.mark.parametrize('demand', [[0, 3, 0, 0, 0], [0, -3, 0, 0, 0], [0, 0, 0, 3, 0]])
def test_shortages_raise_instead_of_scaling_crop_or_forest(demand):
    before = np.array([[2, 2, 5, 5]])
    with pytest.raises(ValueError, match='GTAP demands were not reduced'):
        budget.strict_othernat_targets(np.array(demand)[:, None, None], before,
                                       np.ones_like(before), np.ones_like(before), CODES, 4)


def test_masked_reserve_is_unavailable():
    before = np.array([[5, 5]])
    with pytest.raises(ValueError, match='Insufficient land'):
        budget.strict_othernat_targets(np.array([0, 2, 0, 0, 0])[:, None, None],
                                       before, np.ones_like(before), np.array([[1, 0]]), CODES, 4)


def test_each_year_uses_remaining_stock():
    before = np.array([[5, 5, 5, 5]])
    request = np.array([0, 3, 0, 0, 0])[:, None, None]
    budget.strict_othernat_targets(request, before, np.ones_like(before), np.ones_like(before), CODES, 4)
    after = np.array([[2, 2, 2, 5]])
    with pytest.raises(ValueError, match='Insufficient land'):
        budget.strict_othernat_targets(request, after, np.ones_like(after), np.ones_like(after), CODES, 4)


def test_no_borrowing_from_neighboring_coarse_cell():
    before = np.array([[2, 2, 5, 5]])
    request = np.zeros((5, 1, 2))
    request[1, 0, 0] = 1
    with pytest.raises(ValueError, match=r'cell \(0, 0\)'):
        budget.strict_othernat_targets(request, before, np.ones_like(before), np.ones_like(before), CODES, 4)


def test_class_order_is_explicit_and_residual_can_be_positive():
    before = np.array([[2, 2, 5, 5]])
    target, _ = budget.strict_othernat_targets(
        np.array([999, -2, 0, 0, 0])[:, None, None], before,
        np.ones_like(before), np.ones_like(before), [5, 2, 1, 4, 3], 0)
    np.testing.assert_array_equal(target[:, 0, 0], [2, -2, 0, 0, 0])


@pytest.mark.parametrize('value', [np.nan, np.inf])
def test_nonfinite_demands_rejected(value):
    request = np.zeros((5, 1, 1)); request[1, 0, 0] = value
    before = np.array([[5]])
    with pytest.raises(ValueError, match='finite'):
        budget.strict_othernat_targets(request, before, np.ones_like(before), np.ones_like(before), CODES, 4)


def test_wrong_donor_rejected_even_within_pixel_rounding():
    before = np.array([[4, 5]])
    target = np.array([0, 1, 0, 0, -1])[:, None, None]
    with pytest.raises(ValueError, match='donor overdraw'):
        budget.audit_strict_allocation(target, before, np.array([[2, 5]]),
                                       np.ones_like(before), np.ones_like(before), CODES)


def test_unmet_target_rejected():
    before = np.array([[5, 5, 5]])
    target = np.array([0, 3, 0, 0, -3])[:, None, None]
    with pytest.raises(ValueError, match='Allocation failed'):
        budget.audit_strict_allocation(target, before, before, np.ones_like(before), np.ones_like(before), CODES)


def test_masked_pixel_change_rejected():
    before = np.array([[5, 5]])
    with pytest.raises(ValueError, match='masked'):
        budget.audit_strict_allocation(np.zeros((5, 1, 1)), before, np.array([[5, 2]]),
                                       np.ones_like(before), np.array([[1, 0]]), CODES)
