"""Pared test run of run_seals_calibration.py: the same calibration on a small AOI.

Same task tree and same entry point. Only the scenarios CSV and the project dir differ.
"""
import hazelbean as hb

from run_seals_calibration import projection_task_names, run_project


if __name__ == '__main__':
    p = hb.ProjectFlow(project_name='seals_calibration_test', run_mode='full')
    p.scenario_definitions_filename = 'calibration_scenarios_test.csv'
    p.tasks_to_skip = projection_task_names

    run_project(p)
