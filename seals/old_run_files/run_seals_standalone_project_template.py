"""Template for starting a NEW standalone SEALS project.

    Configuration: level 3 of 4 -- what varies is rows in a scenarios CSV
    Code layout:   library package -- tasks and the tree builder come from seals
    Ownership:     downstream user -- seals is a read-only dependency of your repo

To use: copy this file into your new project repo as run_<project>.py, change the
project name and scenarios CSV in the __main__ block, put your definition CSVs in
an input_template/ directory next to the run file (they are copied to the
project's input/ on first run), and evolve build_task_tree() as your project
diverges from stock SEALS. Keep the canonical anatomy: build_task_tree,
run_project(p), __main__ guard. See earth_economy_devstack/docs/conventions.qmd
for the full rules, and examples/run_templates/ for the three-axis model.

THE OWNERSHIP AXIS IS THE POINT OF THIS FILE
    seals is installed, not edited. Your repo owns this run file, your scenarios
    CSV, and any tasks you write; seals owns the pipeline. Two consequences:

    - Your project dir goes under YOUR project's name, not under seals'. That is
      why nothing here passes extra_dirs: ProjectFlow's git-aware inference puts
      the project beside your repo, which is what you want.
    - To extend the stock tree, add your tasks in build_task_tree AFTER the
      library builder returns -- never by editing seals. Attaching to a stock
      task (parent=p.<some_seals>_task) works because builders assign those
      attributes onto p, but which of them are stable public API is not yet
      specified, so prefer attaching at the top level until it is.
"""
import os

import hazelbean as hb

from seals import seals_initialize_project


def build_task_tree(p):
    # This project's task tree: starts as stock SEALS standard. Compose additional
    # library subtrees or your own project tasks here as the project diverges;
    # only tree construction belongs in this function.
    seals_initialize_project.build_standard_task_tree(p)


def run_project(p):
    """Execute this project's pipeline against the ProjectFlow the caller configured.

    Reads p.scenario_definitions_filename, and optionally p.tasks_to_skip. Returns p.
    """
    # Must be set BEFORE the tree is built: the standard tree contains parallel
    # iterator tasks, which read this at construction time.
    p.run_in_parallel = 1

    # IF YOU WANT TO LOOK AT THE MODEL LOGIC, INSPECT THIS FUNCTION.
    build_task_tree(p)
    p.skip_tasks(p.tasks_to_skip)

    # Project constants: no variant changes these, so they are set here rather
    # than by the caller.
    p.base_data_dir = os.path.join(p.user_dir, 'Files', 'base_data')
    p.data_credentials_path = None
    p.input_bucket_name = None

    # Scenarios CSV drives everything scenario-varying; the caller chose which
    # one. SEALS generates a default in your project's input_dir on first run.
    p.scenario_definitions_path = os.path.join(p.input_dir, p.scenario_definitions_filename)
    seals_initialize_project.initialize_scenario_definitions(p)

    seals_initialize_project.set_advanced_options(p)

    p.L = hb.get_logger(p.project_name)
    hb.log('Created ProjectFlow object at ' + p.project_dir +
           '\n    with base_data set at ' + p.base_data_dir)

    p.execute()

    return p


if __name__ == '__main__':
    # Change these two lines and it is your project. run_mode: 'check' resumes in
    # place | 'fresh_intermediate' rebuilds all computation but keeps input/
    # (test projects only) | 'full' timestamps a new dir.
    p = hb.ProjectFlow(project_name='seals_project_template', run_mode='check')
    p.scenario_definitions_filename = 'standard_scenarios.csv'
    # p.tasks_to_skip = ['stitched_lulc_simplified_scenarios']

    run_project(p)


# A variant run is its own file, never a fork. run_<project>_test.py:
#
#     import hazelbean as hb
#     from run_<project> import run_project
#
#     if __name__ == '__main__':
#         p = hb.ProjectFlow(project_name='<project>_test', run_mode='check')
#         p.scenario_definitions_filename = '<project>_scenarios_test.csv'
#         run_project(p)

# Pre-template work note (preserved from the original stub): fix the
# static_regressors — both pogging them and naming them correctly (no
# minutes_to_market_1m_=30sec etc.) and fixing sand_percent_10sec.tif.
