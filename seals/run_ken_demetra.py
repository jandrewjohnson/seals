"""Kenya (Demetra) SEALS run: ken task tree with Kenya AEZ regions, driven by ken_scenarios_full.csv."""
import os

import hazelbean as hb

from seals import seals_initialize_project


def build_task_tree(p):
    # This project's task tree: delegates unchanged to the shared library builder.
    # Compose additional library subtrees or project-specific tasks here if the
    # project's pipeline ever diverges; only tree construction belongs in this function.
    seals_initialize_project.build_ken_task_tree(p)


def run_project(scenario_definitions_filename='ken_scenarios_full.csv',
                project_name='run_ken_demetra',
                extra_dirs=None,
                run_mode='check',
                tasks_to_skip=None,
                execute=True):
    """Build and execute the Kenya SEALS pipeline against a given scenarios CSV.

    run_mode='full' gives each run its own fresh project dir; the default
    'check' reuses a stable project_name dir so repeated runs resume in place,
    skipping tasks whose outputs already exist. tasks_to_skip pares the tree for
    variant runs; execute=False stops before p.execute(). Returns p.
    """

    valid_run_modes = ('check', 'fresh_intermediate', 'full')
    if run_mode not in valid_run_modes:
        raise ValueError('run_mode must be one of ' + str(valid_run_modes) + ', got ' + repr(run_mode))
    if run_mode == 'fresh_intermediate' and 'test' not in project_name:
        raise ValueError("run_mode='fresh_intermediate' deletes the project's intermediate/ and outputs/ "
                         "dirs, so it is only allowed on dedicated test projects (project_name containing "
                         "'test'), got " + repr(project_name))

    # Create a ProjectFlow Object to organize directories and enable parallel processing.
    p = hb.ProjectFlow()

    # Assign project-level attributes to the p object (such as in p.base_data_dir = ... below)
    # including where the project_dir and base_data are located.
    # The project_name is used to name the project directory below. If the directory exists, each task will not recreate
    # files that already exist.
    p.user_dir = os.path.expanduser('~')
    p.extra_dirs = extra_dirs if extra_dirs is not None else ['Files', 'seals', 'projects']
    p.project_name = project_name
    if run_mode == 'full':
        p.project_name = p.project_name + '_' + hb.pretty_time()  # fresh dir per run; other modes reuse/resume the stable dir.

    # Based on the paths above, set the project_dir. All files will be created in this directory.
    p.project_dir = os.path.join(p.user_dir, os.sep.join(p.extra_dirs), p.project_name)
    p.set_project_dir(p.project_dir)  # NOTE: auto-copies <script_dir>/input_template/ -> input/ (overwrite=False)
    if run_mode == 'fresh_intermediate':
        # Delete in place (rather than timestamping a new dir) so any path derived
        # from project_dir still resolves to the fresh results. input/ is kept: it
        # holds the per-machine backend connection values in parameters.csv that a
        # freshly seeded template would leave blank.
        import shutil
        for stale_dir in [p.intermediate_dir, p.output_dir]:
            if os.path.exists(stale_dir):
                shutil.rmtree(stale_dir)
                print("run_mode='fresh_intermediate': deleted " + stale_dir)


    p.run_in_parallel = 1 # Must be set before building the task tree if the task tree has parralel iterator tasks.

    # Build the task tree via a building function and assign it to p. IF YOU WANT TO LOOK AT THE MODEL LOGIC, INSPECT THIS FUNCTION
    build_task_tree(p)
    p.skip_tasks(tasks_to_skip)

    # Set the base data dir. The model will check here to see if it has everything it needs to run.
    # If anything is missing, it will download it. You can use the same base_data dir across multiple projects.
    p.base_data_dir = os.path.join(p.user_dir, 'Files/base_data')

    # ProjectFlow downloads all files automatically via the p.get_path() function. If you want it to download from a different
    # bucket than default, provide the name and credentials here. Otherwise uses default public data 'gtap_invest_seals_2023_04_21'.
    p.data_credentials_path = None
    p.input_bucket_name = None

    ## Set defaults and generate the scenario_definitions.csv if it doesn't exist.
    # SEALS will run based on the scenarios defined in a scenario_definitions.csv
    # If you have not run SEALS before, SEALS will generate it in your project's input_dir.
    p.scenario_definitions_filename = scenario_definitions_filename
    p.scenario_definitions_path = os.path.join(p.input_dir, p.scenario_definitions_filename)
    seals_initialize_project.initialize_scenario_definitions(p)

    # SEALS is based on an extremely comprehensive region classification system defined in the following geopackage.
    global_regions_vector_ref_path = os.path.join('borders', 'kenya_aez.gpkg')
    p.global_regions_vector_path = p.get_path(global_regions_vector_ref_path)

    # Set Kenya-specific region paths and attributes
    p.regions_path = p.get_path(os.path.join('borders', 'kenya_aez_300m.tif'))
    p.regions_clipped_path = p.get_path(os.path.join('borders', 'kenya_aez_300m_clipped.tif'))
    p.regions_vector_path_epsg8857 = p.get_path(os.path.join('borders', 'kenya_aez_epsg8857.gpkg'))
    p.alt_regions_vector_path_epsg8857 = p.get_path(os.path.join('borders', 'kenya_adm1_epsg8857.gpkg'))
    p.alt_regions_path = p.get_path(os.path.join('borders', 'kenya_adm1.tif'))
    p.alt_regions_clipped_path = p.get_path(os.path.join('borders', 'kenya_adm1_clipped.tif'))
    p.alt_region_id_column = 'CC_1'
    p.alt_region_label_column = 'NAME_1'

    # Set processing resolution: determines how large of a chunk should be processed at a time. 4 deg is about max for 64gb memory systems
    p.processing_resolution = 1.0 # In degrees. Must be in pyramid_compatible_resolutions

    seals_initialize_project.set_advanced_options(p)

    p.L = hb.get_logger('test_run_seals')
    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script + '\n    with base_data set at ' + p.base_data_dir)

    if execute:
        p.execute()

    return p


if __name__ == '__main__':
    run_project()
