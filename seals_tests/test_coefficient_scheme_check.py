import pandas as pd
import pytest

from seals.seals_utils import check_coefficients_match_class_scheme, coefficient_class_labels


SEALS7 = ['urban', 'cropland', 'grassland', 'forest', 'othernat']
SEALS8 = ['urban', 'cropland', 'pasture', 'natural_grassland', 'forest', 'othernat']


def table(class_labels, prefixed=True, extra_columns=()):
    cols = {'spatial_regressor_name': ['soil_bulk_density'], 'data_location': ['x.tif'],
            'type': ['additive']}
    for c in class_labels:
        cols[('class_' + c) if prefixed else c] = [1.0]
    for c in extra_columns:
        cols[c] = [0]
    return pd.DataFrame(cols)


def test_reads_the_prefixed_column_convention():
    assert coefficient_class_labels(table(SEALS7, prefixed=True)) == SEALS7


def test_reads_the_bare_column_convention():
    """default_global_coefficients.csv names its columns after the class directly."""
    assert coefficient_class_labels(table(SEALS7, prefixed=False)) == SEALS7


def test_ignores_bookkeeping_columns():
    d = table(SEALS7, prefixed=False, extra_columns=('calibration_block_index',))
    d.insert(0, 'Unnamed: 0', [0])
    assert coefficient_class_labels(d) == SEALS7


def test_matching_scheme_passes():
    check_coefficients_match_class_scheme(table(SEALS7), SEALS7)


def test_seals7_coefficients_under_a_seals8_correspondence_raise():
    with pytest.raises(ValueError, match='different class scheme'):
        check_coefficients_match_class_scheme(table(SEALS7), SEALS8)


def test_the_message_names_what_is_missing():
    with pytest.raises(ValueError) as e:
        check_coefficients_match_class_scheme(table(SEALS7), SEALS8)
    assert 'pasture' in str(e.value) and 'natural_grassland' in str(e.value)


def test_the_message_names_what_is_unexpected():
    """A seals7 correspondence with seals8 coefficients: grassland is gone, two took its place."""
    with pytest.raises(ValueError) as e:
        check_coefficients_match_class_scheme(table(SEALS8), SEALS7)
    assert 'does not define' in str(e.value)


def test_a_renamed_class_is_caught():
    """The manuscript global set calls it nonforestnatural where Brazil calls it othernat."""
    manuscript = ['urban', 'cropland', 'grassland', 'forest', 'nonforestnatural']
    with pytest.raises(ValueError, match='nonforestnatural'):
        check_coefficients_match_class_scheme(table(manuscript), SEALS7)


def test_reordering_alone_is_fine():
    """Column order is not part of the contract.

    The per-class columns are selected by name, built in the order the correspondence
    defines, so a file carrying the same classes in a different order is read correctly and
    must not be rejected.
    """
    swapped = ['cropland', 'urban', 'grassland', 'forest', 'othernat']
    check_coefficients_match_class_scheme(table(swapped), SEALS7)
