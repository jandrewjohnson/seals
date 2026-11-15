import hazelbean as hb

from seals import seals_initialize_project

# The projection half of the complete tree. A calibration run fits coefficients and stops, so its caller skips these.
projection_task_names = ['coarse_change', 'coarse_extraction', 'coarse_simplified_proportion', 'coarse_interpolated_states', 'coarse_simplified_ha', 'coarse_simplified_ha_difference_from_previous_year', 'regional_change', 'allocations', 'allocation_zones', 'allocation', 'stitched_lulc_simplified_scenarios', 'visualization', 'lulc_pngs']


def build_task_tree(p):
    seals_initialize_project.build_complete_run_task_tree(p)


def run_project(p):
    """Execute the SEALS calibration pipeline against the ProjectFlow the caller configured.

    Reads p.scenario_definitions_filename, and optionally p.tasks_to_skip. Returns p.
    """

    p.run_in_parallel = 1 # Must be set BEFORE the tree is built: the tree contains parallel iterator tasks, which read this at construction time.

    build_task_tree(p)
    p.skip_tasks(p.tasks_to_skip)

    # The size of a calibration zone: one set of coefficients is fitted per block of this size.
    p.processing_resolution = 1.0 # In degrees. Must be in pyramid_compatible_resolutions

    # Scenarios: the baseline row names the observed period to fit, as its base_years.
    hb.initialize_scenarios(p, p.scenario_definitions_filename)

    # Seals' model initializer. Must come after the scenarios load (EE Spec ordering rule).
    seals_initialize_project.initialize_project(p)

    p.execute()

    return p


if __name__ == '__main__':

    # run_mode: 'full' timestamps a new project dir | 'check' resumes in place |
    # 'fresh_intermediate' rebuilds all computation but keeps input/ (test projects only).
    p = hb.ProjectFlow(project_name='seals_calibration', run_mode='check')
    p.scenario_definitions_filename = 'calibration_scenarios.csv'
    p.tasks_to_skip = projection_task_names

    run_project(p)
