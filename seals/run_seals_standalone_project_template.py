"""Template for starting a NEW standalone SEALS project.

To use: copy this file into your new project repo as run_<project>.py, change
PROJECT_NAME and the scenarios CSV default, put your definition CSVs in an
input_template/ directory next to the run file (they are copied to the
project's input/ on first run), and evolve build_task_tree() as your project
diverges from stock SEALS. Keep the canonical anatomy: build_task_tree,
run_project, __main__ guard. See the ProjectFlow conventions in
earth_economy_devstack/docs/conventions.qmd.
"""
import os
import hazelbean as hb
from seals import seals_initialize_project


def build_task_tree(p):
    # This project's task tree: starts as stock SEALS standard. Compose additional
    # library subtrees or project-specific tasks here as the project diverges;
    # only tree construction belongs in this function.
    seals_initialize_project.build_standard_task_tree(p)


def run_project(scenario_definitions_filename='standard_scenarios.csv',
                project_name='seals_project_template',
                run_mode='check',
                tasks_to_skip=None):
    """Full run and test run differ only by the scenarios CSV; run_mode='full'
    gives a fresh project dir per run; a stable project_name with
    run_mode='check' resumes in place, skipping completed tasks. Returns p."""
    # Create a ProjectFlow Object to organize directories and enable parallel processing.
    # set_project_dir_for_run_mode validates run_mode and sets the project_dir under
    # ~/<extra_dirs>/<project_name> (see its docstring for the run_mode semantics).
    p = hb.ProjectFlow()
    p.set_project_dir_for_run_mode(project_name, run_mode,
                                   extra_dirs=['Files', 'seals', 'projects'])

    p.base_data_dir = os.path.join(p.user_dir, 'Files', 'base_data')
    p.data_credentials_path = None
    p.input_bucket_name = None

    p.run_in_parallel = 1  # Must be set before building the task tree.

    build_task_tree(p)
    p.skip_tasks(tasks_to_skip)

    # Scenarios CSV drives everything scenario-varying; generated with defaults if absent.
    p.scenario_definitions_filename = scenario_definitions_filename
    p.scenario_definitions_path = os.path.join(p.input_dir, p.scenario_definitions_filename)
    seals_initialize_project.initialize_scenario_definitions(p)

    seals_initialize_project.set_advanced_options(p)

    p.L = hb.get_logger(p.project_name)
    hb.log('Created ProjectFlow object at ' + p.project_dir)

    p.execute()

    return p


if __name__ == '__main__':
    run_project()

# Pre-template work note (preserved from the original stub): fix the
# static_regressors — both pogging them and naming them correctly (no
# minutes_to_market_1m_=30sec etc.) and fixing sand_percent_10sec.tif.
