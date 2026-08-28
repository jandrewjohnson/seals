"""System test for the seals default template: wraps run_seals.py, never reimplements it.

Imports run_project from seals.run_seals and drives it exactly as
run_seals_test.py (the run_test file) does: the pared RWA scenarios CSV against
the stable seals_test project dir, run_mode='check' so repeated runs resume in
place and only recompute what is missing. Marked requires_base_data + slow, so
it is excluded from the automated tier (pytest -m "not requires_base_data and
not slow") and runs in the full local tier.
"""
import os

import pytest

import hazelbean as hb

pytestmark = [pytest.mark.requires_base_data, pytest.mark.slow]


def test_run_seals_wrapped():
    import seals.run_seals

    p = hb.ProjectFlow(
        project_dir=os.path.join(os.path.expanduser('~'), 'Files', 'seals', 'projects', 'seals_test'),
        run_mode='check')
    # Template seeding copies from the run file's dir (input_template/ beside
    # run_seals.py), not from this test's dir.
    p.script_dir = os.path.dirname(os.path.abspath(seals.run_seals.__file__))
    p.scenario_definitions_filename = 'standard_scenarios_test.csv'

    seals.run_seals.run_project(p)

    stitched_path = os.path.join(
        p.project_dir, 'intermediate', 'stitched_lulc_simplified_scenarios',
        'lulc_esa_seals7_ssp2_rcp45_luh2-message_bau_2030.tif')
    assert hb.path_exists(stitched_path)
