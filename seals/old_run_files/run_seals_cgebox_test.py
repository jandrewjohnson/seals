"""Pared test run of seals_cgebox (devstack layout): same task tree, minimal test scenarios CSV, stable dir."""
from run_seals_cgebox import run_project

if __name__ == '__main__':
    run_project(scenario_definitions_filename='seals_cgebox_scenarios_test.csv',
                run_mode='check')
