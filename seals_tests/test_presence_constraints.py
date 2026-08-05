import pandas as pd
import pytest

from seals.seals_utils import apply_presence_constraints, protected_class_labels


SEALS7_ALL = ['urban', 'cropland', 'grassland', 'forest', 'othernat', 'water', 'other']
SEALS7_CHANGING = ['urban', 'cropland', 'grassland', 'forest', 'othernat']


def coefficients(all_labels, changing_labels, n_blocks=2):
    """A minimal coefficient table: one constraint row per class, plus a fitted row."""
    rows = []
    for block in range(n_blocks):
        for label in all_labels:
            rows.append({'spatial_regressor_name': label + '_presence_constraint',
                         'type': 'multiplicative', 'calibration_block_index': block,
                         **{'class_' + c: 1.0 for c in changing_labels}})
        rows.append({'spatial_regressor_name': 'soil_bulk_density',
                     'type': 'additive', 'calibration_block_index': block,
                     **{'class_' + c: 3.5 for c in changing_labels}})
    return pd.DataFrame(rows)


def test_works_with_the_bare_column_convention():
    """Older files name the class columns directly, without a class_ prefix."""
    d = coefficients(SEALS7_ALL, SEALS7_CHANGING)
    d = d.rename(columns={c: c[len('class_'):] for c in d.columns if c.startswith('class_')})

    out = apply_presence_constraints(d, SEALS7_ALL, SEALS7_CHANGING)

    water = out[out['spatial_regressor_name'] == 'water_presence_constraint']
    assert (water[SEALS7_CHANGING] == 0).all().all()


def test_non_changing_classes_are_excluded_by_default():
    d = coefficients(SEALS7_ALL, SEALS7_CHANGING)
    out = apply_presence_constraints(d, SEALS7_ALL, SEALS7_CHANGING)

    cc = [c for c in out.columns if c.startswith('class_')]
    zeroed = out[(out[cc] == 0).all(axis=1)]['spatial_regressor_name'].unique()

    assert sorted(zeroed) == ['other_presence_constraint', 'water_presence_constraint']


def test_zero_count_matches_the_shipped_production_shape():
    """Three excluded classes x five changing x n blocks, the pattern the delivered file has."""
    d = coefficients(SEALS7_ALL, SEALS7_CHANGING, n_blocks=1840)
    out = apply_presence_constraints(d, SEALS7_ALL, SEALS7_CHANGING,
                                     additional_protected_class_labels=['urban'])

    cc = [c for c in out.columns if c.startswith('class_')]
    assert int((out[out['type'] == 'multiplicative'][cc] == 0).sum().sum()) == 3 * 5 * 1840


def test_fitted_coefficients_are_untouched():
    d = coefficients(SEALS7_ALL, SEALS7_CHANGING)
    out = apply_presence_constraints(d, SEALS7_ALL, SEALS7_CHANGING)

    cc = [c for c in out.columns if c.startswith('class_')]
    fitted = out['type'] != 'multiplicative'
    assert (out.loc[fitted, cc] == 3.5).all().all()


def test_a_no_expansion_area_added_as_a_non_changing_class_is_excluded():
    """Solar as a constraint: present in the land-cover list, absent from the changing list."""
    all_labels = SEALS7_ALL + ['solar']
    d = coefficients(all_labels, SEALS7_CHANGING)
    out = apply_presence_constraints(d, all_labels, SEALS7_CHANGING)

    cc = [c for c in out.columns if c.startswith('class_')]
    solar = out[out['spatial_regressor_name'] == 'solar_presence_constraint']
    assert (solar[cc] == 0).all().all()


def test_other_projects_are_unaffected_by_another_project_s_extra_class():
    """The class list is per project, so a scheme without solar never gains the row."""
    d = coefficients(SEALS7_ALL, SEALS7_CHANGING)
    out = apply_presence_constraints(d, SEALS7_ALL, SEALS7_CHANGING)

    assert 'solar_presence_constraint' not in set(out['spatial_regressor_name'])


def test_protecting_a_class_with_no_row_is_an_error():
    d = coefficients(SEALS7_ALL, SEALS7_CHANGING)

    with pytest.raises(ValueError, match='correspondence'):
        apply_presence_constraints(d, SEALS7_ALL, SEALS7_CHANGING,
                                   additional_protected_class_labels=['solar'])


def test_protected_set_is_derived_plus_named():
    """Non-changing comes from the correspondence; anything else has to be named."""
    assert protected_class_labels(SEALS7_ALL, SEALS7_CHANGING) == ['water', 'other']
    assert protected_class_labels(SEALS7_ALL, SEALS7_CHANGING, ['urban']) == ['water', 'other', 'urban']


def test_urban_is_not_non_changing():
    """Urban has a budget and expands, so protecting it is a scenario choice, not a derivation."""
    assert 'urban' not in protected_class_labels(SEALS7_ALL, SEALS7_CHANGING)


def test_the_older_constraint_row_naming_is_matched():
    """Older tables call the row '<class>_constraint' rather than '<class>_presence_constraint'."""
    d = coefficients(SEALS7_ALL, SEALS7_CHANGING)
    d['spatial_regressor_name'] = d['spatial_regressor_name'].str.replace(
        '_presence_constraint', '_constraint', regex=False)

    out = apply_presence_constraints(d, SEALS7_ALL, SEALS7_CHANGING)

    cc = [c for c in out.columns if c.startswith('class_')]
    water = out[out['spatial_regressor_name'] == 'water_constraint']
    assert (water[cc] == 0).all().all()
