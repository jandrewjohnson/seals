"""Pared test run of run_seals_standard.py: baseline + one bau, one projection year.

Same task tree and same entry point. Only the scenarios CSV and the project dir
differ -- the variant is data and placement, never code.
"""
import hazelbean as hb

from run_seals_standard import run_project


if __name__ == '__main__':
    p = hb.ProjectFlow(project_name='seals_standard_test', run_mode='check')
    p.scenario_definitions_filename = 'standard_scenarios_test.csv'
    # p.tasks_to_skip = ['stitched_lulc_simplified_scenarios']

    run_project(p)
