import pandas as pd

from seals.seals_utils import rebase_project_paths

HERE = '/proj/current/intermediate/fine_processed_inputs'


def table():
    return pd.DataFrame({
        'spatial_regressor_name': ['urban_presence', 'soil_cec', 'forest_gaussian_1',
                                   'water_presence_constraint'],
        'type': ['additive', 'additive', 'gaussian_1', 'multiplicative'],
        'data_location': [
            '/Users/someone/projects/other/intermediate/fine_processed_inputs/lulc/binaries/2015/b.tif',
            '/base_data/soil/soil_cec.tif',
            '/scratch/elsewhere/intermediate/fine_processed_inputs/lulc/convolutions/c.tif',
            '/Users/someone/projects/other/intermediate/fine_processed_inputs/lulc/binaries/2015/w.tif',
        ]})


def test_per_project_layers_are_rerooted_here():
    out = rebase_project_paths(table(), HERE)

    assert out.loc[0, 'data_location'] == HERE + '/lulc/binaries/2015/b.tif'
    assert out.loc[2, 'data_location'] == HERE + '/lulc/convolutions/c.tif'


def test_shared_base_data_layers_are_untouched():
    """Soil and climate live in base_data and are not regenerated per project."""
    out = rebase_project_paths(table(), HERE)

    assert out.loc[1, 'data_location'] == '/base_data/soil/soil_cec.tif'


def test_constraint_rows_are_rerooted_too():
    out = rebase_project_paths(table(), HERE)

    assert out.loc[3, 'data_location'].startswith(HERE)


def test_a_file_already_rooted_here_is_unchanged():
    d = table()
    d.loc[0, 'data_location'] = HERE + '/lulc/binaries/2015/b.tif'

    out = rebase_project_paths(d, HERE)

    assert out.loc[0, 'data_location'] == HERE + '/lulc/binaries/2015/b.tif'
