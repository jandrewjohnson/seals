"""Pared test run of run_seals.py with a fine base year the coarse clock has no step for.

Same task tree and same entry point as run_seals_test.py. Only the scenarios CSV and
the project dir differ: the CSV sets seals_key_base_year to 2015 while key_base_year
stays 2017, so coarse_interpolated_states must interpolate the 2017 coarse state and
the first difference must start from it.
"""
import hazelbean as hb

from run_seals import run_project


if __name__ == '__main__':
    p = hb.ProjectFlow(project_name='seals_interpolation_test', run_mode='check')
    p.scenario_definitions_filename = 'standard_scenarios_interpolation_test.csv'

    run_project(p)
