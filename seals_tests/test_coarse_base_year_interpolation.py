"""Interpolating a coarse state at a year the coarse model does not step at.

The arithmetic matters more than it looks: the coarse map reaches allocation as an anomaly added to
the regional total, so its amplitude is not renormalised away, and getting the weight wrong biases
how sharply change concentrates inside each region.
"""
import os

import numpy as np
import pytest

import hazelbean as hb
from seals import seals_utils


def write_raster(path, array):
    from osgeo import gdal, osr
    hb.create_directories(os.path.dirname(path))
    array = np.asarray(array, dtype='float64')
    dataset = gdal.GetDriverByName('GTiff').Create(
        path, array.shape[1], array.shape[0], 1, gdal.GDT_Float64)
    dataset.SetGeoTransform((0.0, 1.0, 0.0, 0.0, 0.0, -1.0))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(-9999.0)
    band.WriteArray(array)
    dataset = None


@pytest.fixture
def coarse_tree(tmp_path):
    """A coarse directory with states at 2020 and 2025 and nothing between."""
    source_dir = str(tmp_path / 'prop')
    earlier = np.array([[0.0, 10.0], [20.0, 30.0]])
    later = np.array([[0.0, 20.0], [40.0, 80.0]])
    write_raster(os.path.join(source_dir, '2020', 'cropland_prop_2020.tif'), earlier)
    write_raster(os.path.join(source_dir, '2025', 'cropland_prop_2025.tif'), later)
    return source_dir, earlier, later


def test_years_available_reads_the_directory_names(coarse_tree):
    source_dir, _, _ = coarse_tree
    assert seals_utils.coarse_years_available(source_dir) == [2020, 2025]


@pytest.mark.parametrize('target, expected', [
    (2023, (2020, 2025)),      # the case this exists for
    (2020, None),              # already a step, leave it alone
    (2025, None),
    (2015, None),              # cannot be bracketed
    (2030, None),
])
def test_bracketing(coarse_tree, target, expected):
    source_dir, _, _ = coarse_tree
    assert seals_utils.bracketing_coarse_years(source_dir, target) == expected


def test_interpolated_values_use_the_right_weight(coarse_tree):
    """2023 sits 3/5 of the way from 2020 to 2025, so the weight is 0.6."""
    source_dir, earlier, later = coarse_tree
    written = seals_utils.interpolate_coarse_state_at_year(
        source_dir, 2023, (2020, 2025), 'cropland_prop_{year}.tif')

    assert len(written) == 1
    result = hb.as_array(written[0])
    expected = earlier + 0.6 * (later - earlier)      # [[0, 16], [32, 60]]
    assert np.allclose(result, expected)
    # and it must lie between the brackets everywhere, which is the property that matters
    assert np.all(result >= np.minimum(earlier, later) - 1e-9)
    assert np.all(result <= np.maximum(earlier, later) + 1e-9)


def test_rerun_does_not_rewrite(coarse_tree):
    source_dir, _, _ = coarse_tree
    first = seals_utils.interpolate_coarse_state_at_year(
        source_dir, 2023, (2020, 2025), 'cropland_prop_{year}.tif')
    second = seals_utils.interpolate_coarse_state_at_year(
        source_dir, 2023, (2020, 2025), 'cropland_prop_{year}.tif')
    assert len(first) == 1 and second == []


def test_target_outside_the_brackets_raises(coarse_tree):
    source_dir, _, _ = coarse_tree
    with pytest.raises(NameError):
        seals_utils.interpolate_coarse_state_at_year(
            source_dir, 2030, (2020, 2025), 'cropland_prop_{year}.tif')


def test_missing_bracket_file_raises(coarse_tree, tmp_path):
    source_dir, _, _ = coarse_tree
    with pytest.raises(NameError):
        seals_utils.interpolate_coarse_state_at_year(
            source_dir, 2023, (2020, 2025), 'pastureland_prop_{year}.tif')


def test_it_can_write_the_state_somewhere_other_than_beside_its_brackets(tmp_path):
    """The reference trajectory's base-year state belongs in the baseline tree, under the baseline
    naming, because that is the ONE common start every scenario's first difference reads."""
    source_dir = str(tmp_path / 'ref')
    for year, value in ((2020, 10.0), (2030, 20.0)):
        write_raster(os.path.join(source_dir, str(year), 'cropland_prop_ssp2_rcp70_magpie_ref_%d.tif' % year),
                     np.full((2, 2), value))
    baseline_dir = str(tmp_path / 'baseline' / '2023')
    written = seals_utils.interpolate_coarse_state_at_year(
        source_dir, 2023, (2020, 2030), 'cropland_prop_ssp2_rcp70_magpie_ref_{year}.tif',
        dst_dir=baseline_dir, dst_filename_template='cropland_prop_ssp2_magpie_{year}.tif')
    assert written == [os.path.join(baseline_dir, 'cropland_prop_ssp2_magpie_2023.tif')]
    assert np.allclose(hb.as_array(written[0]), 13.0), 'linear at weight 0.3 between 10 and 20'
    # and it did NOT write beside the brackets
    assert not os.path.exists(os.path.join(source_dir, '2023'))


def test_the_default_destination_is_unchanged(tmp_path):
    source_dir = str(tmp_path / 'ref')
    for year, value in ((2020, 10.0), (2030, 20.0)):
        write_raster(os.path.join(source_dir, str(year), 'cropland_{year}.tif'.format(year=year)),
                     np.full((2, 2), value))
    written = seals_utils.interpolate_coarse_state_at_year(
        source_dir, 2023, (2020, 2030), 'cropland_{year}.tif')
    assert written == [os.path.join(source_dir, '2023', 'cropland_2023.tif')]


SCENARIO = dict(exogenous_label='ssp2', climate_label='rcp70', model_label='magpie',
                counterfactual_label='baseline_ignore_dependencies')
SCENARIO_DIR = os.path.join('ssp2', 'rcp70', 'magpie', 'baseline_ignore_dependencies')
SCENARIO_TEMPLATE = 'forest_prop_ssp2_rcp70_magpie_baseline_ignore_dependencies_{year}.tif'


def interpolation_project(tmp_path, monkeypatch, states, key_base_year=2023,
                          scenario_type='baseline_ignore_dependencies'):
    """A project whose reference trajectory has coarse forest states at the given years."""
    import pandas as pd
    from types import SimpleNamespace
    source_dir = str(tmp_path / 'coarse_simplified_proportion')
    for year, value in states.items():
        write_raster(os.path.join(source_dir, SCENARIO_DIR, str(year), SCENARIO_TEMPLATE.format(year=year)),
                     np.full((2, 2), value))
    rows = [dict(scenario_type='baseline', key_base_year=key_base_year, **SCENARIO),
            dict(scenario_type=scenario_type, key_base_year=key_base_year, **SCENARIO)]
    p = SimpleNamespace(run_this=True, cur_dir=str(tmp_path / 'coarse_interpolated_states'),
                        coarse_interpolated_states_dir=str(tmp_path / 'coarse_interpolated_states'),
                        coarse_simplified_proportion_dir=source_dir, coarse_correspondence_path='correspondence.csv',
                        input_dir=str(tmp_path), base_data_dir=str(tmp_path), scenarios_df=pd.DataFrame(rows))
    p.get_path = lambda path, **kwargs: path
    monkeypatch.setattr(hb.utils, 'get_reclassification_dict_from_df', lambda *args: {'dst_labels': ['forest']})
    return p


def test_the_task_writes_the_scenario_state_and_the_common_reference_state(tmp_path, monkeypatch):
    from seals import seals_process_coarse_timeseries as coarse
    p = interpolation_project(tmp_path, monkeypatch, {2020: 0.2, 2025: 0.7})
    assert coarse.coarse_interpolated_states(p) is p

    scenario_path = os.path.join(p.cur_dir, SCENARIO_DIR, '2023', SCENARIO_TEMPLATE.format(year=2023))
    reference_path = os.path.join(p.cur_dir, 'ssp2', 'magpie', '2023', 'forest_prop_ssp2_magpie_2023.tif')
    for path in (scenario_path, reference_path):
        assert np.allclose(hb.as_array(path), 0.5), '0.2 + 0.6 * (0.7 - 0.2)'
    assert not os.path.exists(os.path.join(p.coarse_simplified_proportion_dir, SCENARIO_DIR, '2023'))
    assert seals_utils.coarse_state_dir(p, 'ssp2', 'magpie', '2023') == os.path.dirname(reference_path)
    assert seals_utils.coarse_state_dir(p, 'ssp2', 'magpie', '2020') == os.path.join(
        p.coarse_simplified_proportion_dir, 'ssp2', 'magpie', '2020')


def test_the_task_reuses_existing_states(tmp_path, monkeypatch):
    from seals import seals_process_coarse_timeseries as coarse
    p = interpolation_project(tmp_path, monkeypatch, {2020: 0.2, 2025: 0.7})
    coarse.coarse_interpolated_states(p)
    path = os.path.join(p.cur_dir, SCENARIO_DIR, '2023', SCENARIO_TEMPLATE.format(year=2023))
    stamp = os.stat(path).st_mtime_ns
    coarse.coarse_interpolated_states(p)
    assert os.stat(path).st_mtime_ns == stamp


def test_a_base_year_the_coarse_model_has_is_copied(tmp_path, monkeypatch):
    from seals import seals_process_coarse_timeseries as coarse
    p = interpolation_project(tmp_path, monkeypatch, {2020: 0.2, 2025: 0.7}, key_base_year=2020)
    coarse.coarse_interpolated_states(p)
    path = os.path.join(p.cur_dir, SCENARIO_DIR, '2020', SCENARIO_TEMPLATE.format(year=2020))
    assert np.allclose(hb.as_array(path), 0.2)


def test_a_base_year_outside_the_coarse_years_raises(tmp_path, monkeypatch):
    from seals import seals_process_coarse_timeseries as coarse
    p = interpolation_project(tmp_path, monkeypatch, {2020: 0.2, 2025: 0.7}, key_base_year=2030)
    with pytest.raises(NameError, match='Cannot bracket the coarse base year 2030'):
        coarse.coarse_interpolated_states(p)


def test_a_skipped_task_returns_p_and_writes_nothing(tmp_path, monkeypatch):
    from seals import seals_process_coarse_timeseries as coarse
    p = interpolation_project(tmp_path, monkeypatch, {2020: 0.2, 2025: 0.7})
    p.run_this = False
    assert coarse.coarse_interpolated_states(p) is p
    assert not os.path.exists(p.cur_dir)
