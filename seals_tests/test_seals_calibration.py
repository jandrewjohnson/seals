"""System test for the seals calibration template: wraps run_seals_calibration.py, never reimplements it.

Drives run_project exactly as run_seals_calibration_test.py (the run_test file) does: the pared RWA
scenarios CSV against the stable seals_calibration_test project dir, run_mode='check' so repeated
runs resume in place. Marked requires_base_data + slow, so it is excluded from the automated tier.
"""
import os

import pandas as pd
import pytest

import hazelbean as hb

from seals import seals_utils

pytestmark = [pytest.mark.requires_base_data, pytest.mark.slow]


def test_run_seals_calibration_wrapped():
    import seals.run_seals_calibration

    p = hb.ProjectFlow(
        project_dir=os.path.join(os.path.expanduser('~'), 'Files', 'seals', 'projects', 'seals_calibration_test'),
        run_mode='check')
    # The definitions CSVs are beside the run file, not beside this test (see test_seals.py).
    p.script_dir = os.path.dirname(os.path.abspath(seals.run_seals_calibration.__file__))
    p.input_template_dir = os.path.join(p.script_dir, 'input_template')
    p.scenario_definitions_filename = 'calibration_scenarios_test.csv'
    p.tasks_to_skip = seals.run_seals_calibration.projection_task_names

    seals.run_seals_calibration.run_project(p)

    assert hb.path_exists(p.combined_calibration_file_path)
    coefficients_df = pd.read_csv(p.combined_calibration_file_path)
    # One table per zone that changed, and a table an allocation in the same class scheme would accept.
    assert coefficients_df['calibration_block_index'].nunique() == 6
    seals_utils.check_coefficients_match_class_scheme(coefficients_df, p.changing_class_labels, p.combined_calibration_file_path)
