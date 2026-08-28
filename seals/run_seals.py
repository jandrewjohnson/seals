import os

import hazelbean as hb

from seals import seals_initialize_project


def build_task_tree(p):
    seals_initialize_project.build_standard_task_tree(p)


def run_project(p):
    """Execute the standard SEALS pipeline against the ProjectFlow the caller configured.

    Reads p.scenario_definitions_filename, and optionally p.tasks_to_skip. Returns p.
    """

    p.run_in_parallel = 1 # Must be set BEFORE the tree is built: the standard tree contains parallel iterator tasks, which read this at construction time.
    
    build_task_tree(p)
    p.skip_tasks(p.tasks_to_skip)

    # How large a chunk to process at a time. 4 deg is about the max for 64gb systems.
    p.processing_resolution = 1.0 # In degrees. Must be in pyramid_compatible_resolutions

    # Scenarios: the rows of work. The caller chose which CSV, because that is
    # exactly what a variant run varies. The CSV ships in input_template/ beside
    # this run file and is seeded into input/ on first run.
    hb.initialize_scenarios(p, p.scenario_definitions_filename)

    # Seals' model initializer: advanced options, derived attributes, calibration
    # override dict, logger. Must come after the scenarios load (EE Spec ordering rule).
    seals_initialize_project.initialize_project(p)

    p.execute()

    return p


if __name__ == '__main__':
    
    p = hb.ProjectFlow(project_name='seals_standard', run_mode='check')
    p.scenario_definitions_filename = 'standard_scenarios.csv'

    run_project(p)
