import os, sys
import seals_utils
import seals_initialize_project
import pandas as pd

# Set both GDAL_DATA and PROJ_LIB to correct paths
# conda_env = '/Users/mbraaksma/mambaforge/envs/hb_test'
# os.environ['GDAL_DATA'] = os.path.join(conda_env, 'share', 'gdal')
# os.environ['PROJ_LIB'] = os.path.join(conda_env, 'share', 'proj')
import hazelbean as hb
# Re-set after import in case hazelbean corrupts them
# os.environ['GDAL_DATA'] = os.path.join(conda_env, 'share', 'gdal')
# os.environ['PROJ_LIB'] = os.path.join(conda_env, 'share', 'proj')
# # Verify both are correct
# print(f"GDAL_DATA: {os.environ['GDAL_DATA']}")
# print(f"PROJ_LIB: {os.environ['PROJ_LIB']}")

main = ''
if __name__ == '__main__':

    ### ------- ENVIRONMENT SETTINGS -------------------------------
    # Users should only need to edit lines in this section
    
    # Create a ProjectFlow Object to organize directories and enable parallel processing.
    p = hb.ProjectFlow()

    # Assign project-level attributes to the p object (such as in p.base_data_dir = ... below)
    # including where the project_dir and base_data are located.
    # The project_name is used to name the project directory below. If the directory exists, each task will not recreate
    # files that already exist. 
    p.user_dir = os.path.expanduser('~')        
    p.extra_dirs = ['Files', 'seals', 'projects']
    p.project_name = 'run_ken_demetra'
    # p.project_name = p.project_name + '_' + hb.pretty_time() # If don't you want to recreate everything each time, comment out this line.
    
    # Based on the paths above, set the project_dir. All files will be created in this directory.
    p.project_dir = os.path.join(p.user_dir, os.sep.join(p.extra_dirs), p.project_name)
    p.set_project_dir(p.project_dir) 
    
    p.run_in_parallel = 1 # Must be set before building the task tree if the task tree has parralel iterator tasks.

    # Build the task tree via a building function and assign it to p. IF YOU WANT TO LOOK AT THE MODEL LOGIC, INSPECT THIS FUNCTION
    seals_initialize_project.build_ken_task_tree(p)

    # Set the base data dir. The model will check here to see if it has everything it needs to run.
    # If anything is missing, it will download it. You can use the same base_data dir across multiple projects.
    # Additionally, if you're clever, you can move files generated in your tasks to the right base_data_dir
    # directory so that they are available for future projects and avoids redundant processing.
    # The final directory has to be named base_data to match the naming convention on the google cloud bucket.
    p.base_data_dir = os.path.join(p.user_dir, 'Files/base_data')

    # ProjectFlow downloads all files automatically via the p.get_path() function. If you want it to download from a different 
    # bucket than default, provide the name and credentials here. Otherwise uses default public data 'gtap_invest_seals_2023_04_21'.
    p.data_credentials_path = None
    p.input_bucket_name = None
    
    ## Set defaults and generate the scenario_definitions.csv if it doesn't exist.
    # SEALS will run based on the scenarios defined in a scenario_definitions.csv
    # If you have not run SEALS before, SEALS will generate it in your project's input_dir.
    # A useful way to get started is to to run SEALS on the test data without modification
    # and then edit the scenario_definitions.csv to your project needs.   
    p.scenario_definitions_filename = 'ken_scenarios_full.csv' 
    p.scenario_definitions_path = os.path.join(p.input_dir, p.scenario_definitions_filename)
    seals_initialize_project.initialize_scenario_definitions(p)
        
    # SEALS is based on an extremely comprehensive region classification system defined in the following geopackage.
    # global_regions_vector_ref_path = os.path.join('cartographic', 'ee', 'ee_r264_correspondence.gpkg') 'kenya_aez.gpkg'
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
    
    p.execute()

    result = 'Done!'
