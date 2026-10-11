"""The regressor type mapping must be one implementation, and must reject an unfitted sigma.

allocation_tasks.read_inputs restated this mapping as
`1 if t=='multiplicative' else 2 if t=='additive' or str(t).startswith('gaussian_') else 0`,
which admits ANY gaussian_* name. A coefficient table carrying a sigma that was never fitted was
therefore coded as additive and allocated on, instead of failing. The canonical construction in
seals_main.allocation_zones admits only the sigmas in p.gaussian_sigmas_to_test.
"""
import numpy as np
import pytest

from seals import seals_utils


def _canonical(types, gaussian_sigmas_to_test):
    """The construction as seals_main.allocation_zones builds it, copied for comparison only."""
    codes = {'multiplicative': 1, 'additive': 2}
    codes.update({'gaussian_' + str(sigma): 2 for sigma in gaussian_sigmas_to_test})
    return np.asarray([codes[i] for i in types], np.int64)


def test_it_matches_the_canonical_construction():
    sigmas = [1, 3, 5]
    types = ['multiplicative', 'additive', 'gaussian_1', 'gaussian_3', 'gaussian_5', 'additive']
    assert np.array_equal(seals_utils.regressor_type_codes(types, sigmas), _canonical(types, sigmas))


def test_an_untested_sigma_is_an_error_not_an_additive_default():
    with pytest.raises(ValueError, match='gaussian_7'):
        seals_utils.regressor_type_codes(['multiplicative', 'gaussian_7'], [1, 3, 5])
    # The behaviour the restatement had, pinned so a regression is visible: it returned 2.
    assert seals_utils.regressor_type_codes(['gaussian_3'], [1, 3, 5]).tolist() == [2]


def test_the_canonical_construction_also_rejects_it():
    # Same input, same verdict: the shared helper is not stricter than the standard path, it is
    # the standard path. KeyError there, ValueError here, both refusals.
    with pytest.raises(KeyError):
        _canonical(['gaussian_7'], [1, 3, 5])


def test_an_unknown_name_is_rejected_and_the_message_names_what_is_allowed():
    with pytest.raises(ValueError, match="multiplicative"):
        seals_utils.regressor_type_codes(['logarithmic'], [1])


def test_no_sigmas_tested_means_no_gaussian_is_admitted():
    assert seals_utils.regressor_type_codes(['additive'], []).tolist() == [2]
    with pytest.raises(ValueError):
        seals_utils.regressor_type_codes(['gaussian_1'], [])
