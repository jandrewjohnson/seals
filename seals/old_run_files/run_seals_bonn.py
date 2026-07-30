"""Bonn ESM CGEBox-SEALS project: downscale CGEBox land cover projections to 300m LULC maps with SEALS."""
import os

import hazelbean as hb
import pandas as pd

from seals import seals_generate_base_data, seals_initialize_project, seals_main, seals_process_coarse_timeseries, seals_tasks, seals_utils, seals_visualization_tasks


def convert_cgebox_output_to_seals_regional_projections_input(p):

    p.regional_projections_input_override_paths = {}
    if p.run_this:
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

                    # multiply by 1000 because i think cgebox outputs in thousands of ha
                    for col in land_use_columns:
                        df_sorted[col] = df_sorted[col] * 1000
                    # 2019 2020 2021 2023 2025 2027 2029 2030 2031 2033 2035 2037 2039 2040 2041 2043 2045 2047 2049 2050
                    # repeat these numbers

                    # Write a new file in the task dir and reassign the project attribute to the new csv
                    hb.df_write(df_sorted, output_path)


def build_task_tree(p):
    # (was build_bonn_task_tree)

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


def run_project(scenario_definitions_filename='scenarios.csv',
                project_name='bonn_esm_cgebox_seals',
                run_mode='check',
                tasks_to_skip=None,
                project_dir=None):
    """Build and execute the Bonn ESM CGEBox-SEALS pipeline against a given scenarios CSV.

    run_mode='full' gives a fresh project dir per run; the default 'check' reuses/
    resumes the stable dir. project_dir=None puts the project at
    ~/Files/seals/projects/<project_name> (the layout p.extra_dirs already declared),
    which unlike the previous '../../Projects' default does not depend on the working
    directory; pass an explicit path to override.
    """

    # Create a ProjectFlow Object to organize directories and enable parallel processing.
    # The constructor validates run_mode and applies its semantics (timestamped dir for
    # 'full', in-place clearing of intermediate/ and outputs/ for 'fresh_intermediate').
    if project_dir is None:
        p = hb.ProjectFlow(project_name=project_name, run_mode=run_mode,
                           extra_dirs=['Files', 'seals', 'projects'])
    else:
        p = hb.ProjectFlow(project_dir=project_dir, run_mode=run_mode)

    # Set processing resolution: determines how large of a chunk should be processed at a time. 4 deg is about max for 64gb memory systems
    p.processing_resolution = 1.0 # In degrees. Must be in pyramid_compatible_resolutions

    # Set the base data dir. The model will check here to see if it has everything it needs to run.
    # If anything is missing, it will download it. You can use the same base_data dir across multiple projects.
    p.base_data_dir = '../../../base_data/' # Automatically downloaded data will go here.


    # Seed data dir: the repo's tracked input/bonn_input dir (scenarios.csv + SSP1-5.csv) is
    # copied into the project's input dir, skipping anything already there.
    p.input_data_dir = 'input/bonn_input'
    if hb.path_exists(p.input_data_dir, verbose=True):
        hb.copy_file_tree_to_new_root(p.input_data_dir, p.input_dir, skip_existing=True)
        hb.log(f'Copied input data from {p.input_data_dir} to {p.input_dir}.')

    # SEALS will run based on the scenarios defined in this csv. If you have not run SEALS before,
    # SEALS will generate it in your project's input_dir.
    p.scenario_definitions_path = os.path.join(p.input_dir, scenario_definitions_filename)

    p.run_in_parallel = 1 # Must be set before building the task tree if the task tree has parralel iterator tasks.

    # Build the task tree via a building function and assign it to p. IF YOU WANT TO LOOK AT THE MODEL LOGIC, INSPECT THIS FUNCTION
    build_task_tree(p)
    p.skip_tasks(tasks_to_skip)

    # ProjectFlow downloads all files automatically via the p.get_path() function. If you want it to download from a different
    # bucket than default, provide the name and credentials here. Otherwise uses default public data 'gtap_invest_seals_2023_04_21'.
    p.data_credentials_path = None
    p.input_bucket_name = None

    seals_initialize_project.initialize_scenario_definitions(p)

    seals_initialize_project.set_advanced_options(p)

    p.L = hb.get_logger('bonn_esm_cgebox_seals')
    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script + '\n    with base_data set at ' + p.base_data_dir)

    p.execute()

    return p


if __name__ == '__main__':
    run_project()
