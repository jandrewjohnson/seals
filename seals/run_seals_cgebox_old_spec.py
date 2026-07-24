import os, sys, time

import hazelbean as hb
import pandas as pd

from seals import seals_generate_base_data, seals_initialize_project, seals_main, seals_process_coarse_timeseries, seals_tasks, seals_utils, seals_visualization_tasks

### ENVIRONMENT VARIABLES (nothing else should need to be edited besides here)
project_name = 'seals_cgebox_devstack' # Name of the project. Also is used to set the project dir

project_dir = os.path.join('../projects', project_name) # DEVSTACK OPTION. If you're running a standalone repo clone, you probably want to set this to just os.path.join('..')
input_dir = os.path.join(project_dir, 'input') 
input_data_dir = 'input/seals_cgebox_input' # Will look in the repo or base data for this and then copy it to the input_dir above. 
scenario_definitions_path = os.path.join(input_dir, 'seals_cgebox_scenarios.csv') # Path to the scenario definitions file. 


def convert_cgebox_output_to_seals_regional_projections_input(p):   
    """Convert CGEBox output to SEALS regional projections input format.
    
    This function reads CGEBox output files, processes them to calculate year-over-year changes
    in land use categories, and saves the transformed data in a format compatible with SEALS.
    
    This is the only task different from the standard seals workflow and is the only task
    different in the build_bonn_task_tree function.
    
    The hard part is that because the input regional_change csvs are modified, we need also to write
    a replacements dictionary p.regional_projections_input_override_paths that maps scenario labels
    to the new csvs. SEALS checks for an override dictionary here to replace what is defined 
    in the scenarios.csv."""

    
    if p.run_this:
        
        # Define a dictionary to hold override paths for each scenario
        p.regional_projections_input_override_paths = {}        
        
        for index, row in p.scenarios_df.iterrows():
            seals_utils.assign_df_row_to_object_attributes(p, row)

            if p.scenario_type != 'baseline':    
                input_path = p.regional_projections_input_path
                output_path = os.path.join(p.cur_dir, f'regional_projections_input_pivoted_{p.exogenous_label}.csv')     
                
                # START HERE, the override doesn't work cause it doesn't iterate over years.... use a catear?           
                p.regional_projections_input_override_paths[p.scenario_label] = output_path
                    
                if not hb.path_exists(output_path):
                    df = hb.df_read(input_path)
                    
                    # Step 1: Melt the DataFrame to convert year columns into rows.
                    # Get the list of columns to unpivot (years)
                    years_to_unpivot = [col for col in df.columns if col.isdigit()]
                    melted = df.melt(
                        id_vars=[p.regions_column_label, 'LandCover'], # Assumes the land cover column is named 'LandCover'
                        value_vars=years_to_unpivot,
                        var_name='year',
                        value_name='value'
                    )

                    # Step 2: Pivot the melted DataFrame.
                    # We set the region and year as the new index, and create new columns from 'LandCover' categories.
                    merged_pivoted = melted.pivot_table(
                        index=[p.regions_column_label, 'year'],
                        columns='LandCover',
                        values='value'
                    ).reset_index()
                    
                    # Now add nuts_id
                    merged_pivoted['nuts_id'], unique_countries = pd.factorize(merged_pivoted[p.regions_column_label])
                    merged_pivoted['nuts_id'] = merged_pivoted['nuts_id'] + 1
                    
                    # Define the columns for which the year-over-year change should be calculated
                    land_use_columns = ['cropland', 'forest', 'grassland', 'other', 'othernat', 'urban', 'water']

                    # Sort the DataFrame by 'nuts_label' and 'year' to ensure correct chronological order
                    df_sorted = merged_pivoted.sort_values(by=['nuts_label', 'year'])

                    # Group by 'nuts_label' and calculate the difference for the specified columns
                    # .diff() calculates the difference from the previous row within each group
                    # .fillna(0) replaces the initial NaN values with 0
                    df_sorted[land_use_columns] = df_sorted.groupby('nuts_label')[land_use_columns].diff().fillna(0)

                    # The 'df_sorted' DataFrame now contains the year-over-year change.
                    # You can display the first few rows of the result for a specific region to verify.
                    print("Year-over-year changes for CZ03:")
                    print(df_sorted[df_sorted['nuts_label'] == 'CZ03'].head())            
                    
                    # multiply by 1000 because i cgebox outputs in thousands of ha
                    for col in land_use_columns:
                        df_sorted[col] = df_sorted[col] * 1000
                    
                    # Write a new file in the task dir and reassign the project attribute to the new csv
                    hb.df_write(df_sorted, output_path)
                
            
def build_bonn_task_tree(p):

    # Define the project AOI
    p.project_aoi_task = p.add_task(seals_tasks.project_aoi)
    p.convert_cgebox_output_to_seals_regional_projections_input_task = p.add_task(convert_cgebox_output_to_seals_regional_projections_input)
    

    ##### FINE PROCESSED INPUTS #####
    p.fine_processed_inputs_task = p.add_task(seals_generate_base_data.fine_processed_inputs)
    p.generated_kernels_task = p.add_task(seals_generate_base_data.generated_kernels, parent=p.fine_processed_inputs_task, creates_dir=False)
    p.lulc_clip_task = p.add_task(seals_generate_base_data.lulc_clip, parent=p.fine_processed_inputs_task, creates_dir=False)
    p.lulc_simplifications_task = p.add_task(seals_generate_base_data.lulc_simplifications, parent=p.fine_processed_inputs_task, creates_dir=False)
    p.lulc_binaries_task = p.add_task(seals_generate_base_data.lulc_binaries, parent=p.fine_processed_inputs_task, creates_dir=False)
    p.lulc_convolutions_task = p.add_task(seals_generate_base_data.lulc_convolutions, parent=p.fine_processed_inputs_task, creates_dir=False)

    ##### COARSE CHANGE #####
    p.coarse_change_task = p.add_task(seals_process_coarse_timeseries.coarse_change, skip_existing=0)
    p.extraction_task = p.add_task(seals_process_coarse_timeseries.coarse_extraction, parent=p.coarse_change_task, run=1, skip_existing=0)
    p.coarse_simplified_task = p.add_task(seals_process_coarse_timeseries.coarse_simplified_proportion, parent=p.coarse_change_task, skip_existing=0)
    p.coarse_simplified_ha_task = p.add_task(seals_process_coarse_timeseries.coarse_simplified_ha, parent=p.coarse_change_task, skip_existing=0)
    p.coarse_simplified_ha_difference_from_previous_year_task = p.add_task(seals_process_coarse_timeseries.coarse_simplified_ha_difference_from_previous_year, parent=p.coarse_change_task, skip_existing=0)

    ##### REGIONAL
    p.regional_change_task = p.add_task(seals_process_coarse_timeseries.regional_change)

    ##### ALLOCATION #####
    p.allocations_task = p.add_iterator(seals_main.allocations, skip_existing=0)
    p.allocation_zones_task = p.add_iterator(seals_main.allocation_zones, run_in_parallel=p.run_in_parallel, parent=p.allocations_task, skip_existing=0)
    p.allocation_task = p.add_task(seals_main.allocation, parent=p.allocation_zones_task, skip_existing=0)

    ##### STITCH ZONES #####
    p.stitched_lulc_simplified_scenarios_task = p.add_task(seals_main.stitched_lulc_simplified_scenarios)

    ##### VIZUALIZE EXISTING DATA #####
    p.visualization_task = p.add_task(seals_visualization_tasks.visualization)
    p.lulc_pngs_task = p.add_task(seals_visualization_tasks.lulc_pngs, parent=p.visualization_task)


main = ''
if __name__ == '__main__':

    # Create a ProjectFlow Object to organize directories and enable parallel processing.
    p = hb.ProjectFlow()
    
    # Default locations (to be used if local vars not defined above)
    p.user_dir = os.path.expanduser('~')
    p.extra_dirs = ['Files', 'seals', 'projects']   
    
    # Set processing resolution: determines how large of a chunk should be processed at a time. 4 deg is about max for 64gb memory systems
    p.processing_resolution = 1.0 # In degrees. Must be in pyramid_compatible_resolutions
    
    hb.log(f'Running script {__file__} with abs {os.path.abspath(__file__)}.')
    
    if 'project_name' in globals():
        hb.log(f'Using locally set project_name: {project_name}')
        p.project_name = project_name        
        generate_new_project_dir_with_timestamp_for_every_run = False # If true, every run goes into a new and unique folder. This can help with debuging, but also means each run will be very slow as it will not use precalculated results.
        if generate_new_project_dir_with_timestamp_for_every_run:
            p.project_name = p.project_name + '_' + hb.pretty_time()      
    
    if 'project_dir' in globals():
        hb.log(f'Using locally set project_dir: {project_dir} with abspath {os.path.abspath(project_dir)}')
        p.project_dir = project_dir        
    else:
        p.project_dir = os.path.join(p.user_dir, os.sep.join(p.extra_dirs), p.project_name)
    p.set_project_dir(p.project_dir)       
        
    if 'input_dir' in globals():
        hb.log(f'Using locally set input_dir: {input_dir} with abspath {os.path.abspath(input_dir)}, But first we need to check that its different than the one implied by the project_dir')
        if p.input_dir != input_dir:
            hb.log(f'Overriding project input_dir {p.input_dir} with locally set input_dir: {input_dir}')            
            p.input_dir = input_dir
        else:
            pass # Just keep the one implied by the project_dir

    if 'input_data_dir' in globals():
        hb.log(f'Detected locally set input_data_dir: {input_data_dir} with abspath {os.path.abspath(input_data_dir)}. This happens when the data used for project setup (not the raw spatial data) is obtianed by git cloning. The assumed behavior here is that it will copy it from the repo input dir to the project input dir.')
        p.input_data_dir = input_data_dir
        # Copy the input data dir to the project input dir if it exists and the project input dir doesn't exist.
        # When we copy to the local dir, we drop the input_projectname part:
        _, target_dir = os.path.split(p.input_data_dir) 
        if hb.path_exists(p.input_data_dir, verbose=True):
            hb.copy_file_tree_to_new_root(p.input_data_dir, p.input_dir, skip_existing=True)
            hb.log(f'Copied input data from {p.input_data_dir} and target dir {target_dir}, abspath: {os.path.abspath(p.input_data_dir)} to {p.input_dir}, abspath: {os.path.abspath(p.input_dir)}.')
    
    # Check for locally-set versions of base_data_dir, project_dir, and input_dir
    
    # Set the base data dir. The model will check here to see if it has everything it needs to run.
    # If anything is missing, it will download it. You can use the same base_data dir across multiple projects.
    # Additionally, if you're clever, you can move files generated in your tasks to the right base_data_dir
    # directory so that they are available for future projects and avoids redundant processing.
    # The final directory has to be named base_data to match the naming convention on the google cloud bucket.
    # if 'base_data_dir' in globals():
    #     hb.log(f'Using locally set base_data_dir: {base_data_dir}')
    #     p.base_data_dir = base_data_dir
    # else:
    #     p.base_data_dir = os.path.join(p.user_dir, 'Files/base_data')
    # # Actually set the base date, which will also validate that this folder is correct and not a duplicate.
    p.set_base_data_dir()    
    
    ## Set defaults and generate the scenario_definitions.csv if it doesn't exist.
    # SEALS will run based on the scenarios defined in a scenario_definitions.csv
    # If you have not run SEALS before, SEALS will generate it in your project's input_dir.
    # A useful way to get started is to to run SEALS on the test data without modification
    # and then edit the scenario_definitions.csv to your project needs.       
    if 'scenario_definitions_path' in globals():
        hb.log(f'Using locally set scenarios_file_path: {scenario_definitions_path} with abspath {os.path.abspath(scenario_definitions_path)}')
        p.scenario_definitions_path = scenario_definitions_path
    else:
        p.scenario_definitions_path = os.path.join(p.input_dir, 'scenarios.csv')      

    p.run_in_parallel = 1 # Must be set before building the task tree if the task tree has parralel iterator tasks.

    # Build the task tree via a building function and assign it to p. IF YOU WANT TO LOOK AT THE MODEL LOGIC, INSPECT THIS FUNCTION
    build_bonn_task_tree(p)

    # ProjectFlow downloads all files automatically via the p.get_path() function. If you want it to download from a different
    # bucket than default, provide the name and credentials here. Otherwise uses default public data 'gtap_invest_seals_2023_04_21'.
    p.data_credentials_path = None
    p.input_bucket_name = None


    seals_initialize_project.initialize_scenario_definitions(p)       

    seals_initialize_project.set_advanced_options(p)

    p.L = hb.get_logger('test_run_seals')
    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script + '\n    with base_data set at ' + p.base_data_dir)

    p.execute()

    result = 'Done!'

