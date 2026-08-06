import os

from seals.seals_utils import resolve_constraint_layers

import pandas as pd


def table():
    rows = [
        {'spatial_regressor_name': 'water_presence_constraint', 'type': 'multiplicative',
         'data_location': '/somewhere/else/binaries/2015/binary_mapbiomas_300m_seals7_2015_water.tif'},
        {'spatial_regressor_name': 'other_constraint', 'type': 'multiplicative',
         'data_location': '/somewhere/else/binaries/2015/binary_x.tif'},
        {'spatial_regressor_name': 'soil_bulk_density', 'type': 'additive',
         'data_location': '/base_data/soil/bulk_density.tif'},
    ]
    return pd.DataFrame(rows)


def resolved():
    return resolve_constraint_layers(table(), '/proj/fine_processed_inputs',
                                     'mapbiomas_300m', 'seals7', 2020)


def test_the_layer_year_becomes_the_base_year_being_allocated_from():
    out = resolved()
    water = out.loc[0, 'data_location']

    assert '/binaries/2020/' in water
    assert water.endswith('binary_mapbiomas_300m_seals7_2020_water.tif')


def test_a_foreign_project_path_is_replaced_by_this_run_s():
    """A coefficient file calibrated elsewhere points at a directory that need not exist."""
    out = resolved()

    assert out.loc[0, 'data_location'].startswith('/proj/fine_processed_inputs')
    assert 'somewhere/else' not in out.loc[0, 'data_location']


def test_both_row_naming_conventions_are_resolved():
    out = resolved()

    assert out.loc[1, 'data_location'].endswith('binary_mapbiomas_300m_seals7_2020_other.tif')


def test_regressors_that_are_not_constraints_are_left_alone():
    out = resolved()

    assert out.loc[2, 'data_location'] == '/base_data/soil/bulk_density.tif'
