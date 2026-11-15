"""The guard must fire on the real mismatch and stay silent on the real match."""
import pandas as pd
import pytest
from seals import seals_utils


def coefficients(columns, rows):
    return pd.DataFrame({'calibration_block_index': ['%d_%d_1_1' % (c, r) for c in columns for r in rows],
                         'spatial_regressor_name': 'x'})


def test_fires_on_one_degree_coefficients_used_at_four_degrees():
    df = coefficients(range(360), range(6, 146))            # 1 degree, as the manuscript files are
    zones = ['%d_%d' % (c, r) for c in range(90) for r in range(45)]   # 4-degree zone ids
    with pytest.raises(NameError, match='FINER processing grid'):
        seals_utils.assert_calibration_matches_processing_grid(df, zones, 4.0, 'trained.csv')


def test_silent_when_the_grids_agree():
    df = coefficients(range(360), range(6, 146))
    zones = ['%d_%d' % (c, r) for c in range(360) for r in range(6, 146)]
    got = seals_utils.assert_calibration_matches_processing_grid(df, zones, 1.0, 'trained.csv')
    assert got == 1.0


def test_a_regional_coefficient_file_at_the_right_resolution_still_fails_loudly():
    """Brazil-only coefficients used globally cover almost nothing; that is worth refusing too."""
    df = coefficients(range(106, 146), range(84, 124))
    zones = ['%d_%d' % (c, r) for c in range(360) for r in range(6, 146)]
    with pytest.raises(NameError, match='find a calibration key'):
        seals_utils.assert_calibration_matches_processing_grid(df, zones, 1.0, 'brazil.csv')


def test_a_table_without_block_keys_addresses_every_zone_and_is_not_checked():
    df = pd.DataFrame({'spatial_regressor_name': ['x'], 'urban': [1.0]})
    assert seals_utils.assert_calibration_matches_processing_grid(df, ['0_0'], 4.0, 'default.csv') is None
