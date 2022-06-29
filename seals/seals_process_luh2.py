import os
import hazelbean as hb
import numpy as np
import pandas as pd
import multiprocessing
from matplotlib import pyplot as plt
import geopandas as gpd

import seals_utils

L = hb.get_logger()

def download_base_data(p):

    if p.run_this:

        p.required_base_data_urls = []
        p.required_base_data_dst_paths = []


        flattened_list = hb.flatten_nested_dictionary(p.required_base_data_paths, return_type='values')
        # flattened_list = list(flattened_list.values())

        p.L.info('Script requires the following Base Data to be in your base_data_dir\n' + hb.pp(flattened_list, return_as_string=True))
        for path in flattened_list:
            if not hb.path_exists(path) and not path == 'use_generated':
                p.L.info('Path did not exist, so adding it to urls to download: ' + str(path))

                # HACK, should have made this cleaner

                url_from_path =  path.split(os.path.split(p.base_data_dir)[1])[1].replace('\\', '/')
                url_from_path = 'base_data' + url_from_path

                p.required_base_data_urls.append(url_from_path)
                p.required_base_data_dst_paths.append(path)

        if len(p.required_base_data_urls) > 0:
            for c, blob_url in enumerate(p.required_base_data_urls):

                # The data_credentials_path file needs to be given to the user with a specific service account email attached. Generated via the gcloud CMD line, described in new_computer.
                filename = os.path.split(blob_url)[1]
                dst_path = p.required_base_data_dst_paths[c]
                if not hb.path_exists(dst_path): # Check one last time to ensure that it wasn't added twice.
                    hb.download_google_cloud_blob(p.input_bucket_name, blob_url, p.data_credentials_path, dst_path)


def luh2_extraction(p):
    """Get subset of NC data from LUH for the desired scenarios, years, etc. Based on the release used, this should generate 14 layers of
    30km grid-cells for "state" of the grid-cell, which is a proportion of grid-cell in one of the 14 states.
    All luh data extracted from the luh2 project and Hurtt et al. 2017."""
    p.luh_data_dir = hb.luh_data_dir
    p.luh_scenario_management_paths = hb.luh_scenario_management_paths
    if p.run_this:

        for base_year in p.base_years:

            """2015 is the first year in the non-historical data, so if your base year is past that, have to draw from the 
            projections. We assumeed SSP2 represents years before the "policy scenarios" start."""

            output_dir = os.path.join(p.cur_dir, 'baseline', str(base_year))
            hb.create_directories(output_dir)

            if base_year <= 2014:
                base_year_path = p.luh_scenario_states_paths['historical']
            else:
                base_year_path = p.luh_scenario_states_paths['rcp45_ssp2']

            # VALIDATION: Should extract 14 layers, the 14 layers of LUH.
            if len(hb.list_filtered_paths_nonrecursively(output_dir, include_extensions='.tif')) != 14:
                p.L.info('Extracting from ' + base_year_path)
                if base_year <= 2014:
                    hb.extract_luh_netcdf_to_geotiffs(base_year_path, output_dir, base_year - 850)  # 0 = 2015, last year is 85=2100
                else:
                    hb.extract_luh_netcdf_to_geotiffs(base_year_path, output_dir, base_year - 2015)

        # for scenario_name in p.policy_scenario_labels:
        for luh_scenario_label in p.luh_scenario_labels:
            for year in p.scenario_years:
                if year < 2015:
                    states_path = p.luh_scenario_states_paths['historical']
                else:
                    states_path = p.luh_scenario_states_paths[luh_scenario_label]

                output_dir = os.path.join(p.cur_dir, luh_scenario_label, str(year))
                hb.create_directories(output_dir)

                if len(hb.list_filtered_paths_nonrecursively(output_dir, include_extensions='.tif')) != 14:
                    p.L.info('Extracting from ' + states_path)

                    if year <= 2014:
                        hb.extract_luh_netcdf_to_geotiffs(states_path, output_dir,
                                                          year - 850)  # 0 = 2015, last year is 85=2100
                    else:
                        hb.extract_luh_netcdf_to_geotiffs(states_path, output_dir, year - 2015)

            ## KEEP for when i want to incorporate fertilizer.
            # extract_management = 0
            # if extract_management:
            #     management_path = p.luh_scenario_management_paths[scenario_name]
            #
            #     for year in p.years:
            #         output_dir = os.path.join(p.cur_dir, scenario_name, str(year), 'management')
            #         hb.create_directories(output_dir)
            #         p.L.info('Extracting from ' + management_path)
            #         hb.extract_luh_netcdf_to_geotiffs(management_path, output_dir, year - 2015)  # 0 = 2015, last year is 85=2100


def luh2_difference_from_base_year(p):
    # p.luh2_extraction_dir = r'C:\Files\Research\base_data\ipbes\processed_data\luh2_extraction_'
    # p.match_15m_float_path = os.path.join(p.luh2_extraction_dir, 'baseline', str(p.base_year), 'c3ann.tif')
    p.match_15m_global_float_path = os.path.join(p.luh2_extraction_dir, 'baseline', str(p.base_year), 'c3ann.tif')
    p.coarse_aoi_ha_per_cell_path = os.path.join(p.cur_dir, 'coarse_aoi_ha_per_cell.tif')


    if p.run_this:
        p.ha_per_cell_array = hb.load_geotiff_chunk_by_bb(p.coarse_ha_per_cell_path, p.bb, output_path=p.coarse_aoi_ha_per_cell_path)
        # p.match_15m_float_af = hb.ArrayFrame(p.match_15m_global_float_path)
        for scenario_year in p.scenario_years:  # Skip first cause it is 2015
            for base_year in p.base_years:
                for scenario_name in p.luh_scenario_labels:
                    state_baseline_dir = os.path.join(p.luh2_extraction_dir, 'baseline', str(base_year))
                    states_future_dir = os.path.join(p.luh2_extraction_dir, scenario_name, str(scenario_year))
                    for state_name in hb.luh_state_names:
                        difference_dir = os.path.join(p.cur_dir, scenario_name)
                        difference_path = os.path.join(difference_dir, state_name + '_' + str(scenario_year) + '_' + str(base_year) + '_ha_difference.tif')
                        if not hb.path_exists(difference_path):

                            hb.create_directories(difference_dir)

                            state_baseline_path = os.path.join(state_baseline_dir, state_name + '.tif')
                            state_future_path = os.path.join(states_future_dir, state_name + '.tif')
                            proportion_states_baseline_array = hb.load_geotiff_chunk_by_bb(state_baseline_path, p.bb)
                            proportion_states_baseline_ndv = hb.get_ndv_from_path(state_baseline_path)
                            proportion_states_future_array = hb.load_geotiff_chunk_by_bb(state_future_path, p.bb)
                            # proportion_states_baseline_ndv = hb.get_ndv_from_path(state_baseline_path)
                            # proportion_states_baseline = hb.ArrayFrame(state_baseline_path)
                            # proportion_states_future = hb.ArrayFrame(state_future_path)

                            # NOTE, tested several methods and this was the fastest and most flexible. Note that because it doesn't call valid_mask, that never has to load
                            difference_array = np.where(proportion_states_baseline_array != proportion_states_baseline_ndv, proportion_states_future_array - proportion_states_baseline_array, proportion_states_baseline_ndv)

                            difference = hb.save_array_as_geotiff(difference_array, difference_path, p.coarse_aoi_ha_per_cell_path)


def luh2_as_seals7_proportion(p):
    """Convert the luh2, .25 degree proportion of each of 13 classes into the hectares of a simplified set of classes
    (SEALS5). """


    p.luh2_to_seals7_correspondence = {}
    p.luh2_to_seals7_correspondence['urban'] = 'urban'
    p.luh2_to_seals7_correspondence['c4per'] = 'cropland'
    p.luh2_to_seals7_correspondence['c3ann'] = 'cropland'
    p.luh2_to_seals7_correspondence['c3nfx'] = 'cropland'
    p.luh2_to_seals7_correspondence['c3per'] = 'cropland'
    p.luh2_to_seals7_correspondence['c4ann'] = 'cropland'
    p.luh2_to_seals7_correspondence['range'] = 'grassland'
    p.luh2_to_seals7_correspondence['pastr'] = 'grassland'
    p.luh2_to_seals7_correspondence['primf'] = 'forest'
    p.luh2_to_seals7_correspondence['secdf'] = 'forest'
    p.luh2_to_seals7_correspondence['primn'] = 'nonforestnatural'
    p.luh2_to_seals7_correspondence['secdn'] = 'nonforestnatural'

    p.seals7_to_luh2_correspondence = {}
    p.seals7_to_luh2_correspondence['urban'] = ['urban']
    p.seals7_to_luh2_correspondence['cropland'] = ['c4per', 'c3ann', 'c3nfx', 'c3per', 'c4ann']
    p.seals7_to_luh2_correspondence['grassland'] = ['range', 'pastr']
    p.seals7_to_luh2_correspondence['forest'] = ['primf', 'secdf']
    p.seals7_to_luh2_correspondence['nonforestnatural'] = ['primn', 'secdn']


    if p.run_this:

        p.ha_per_cell_array = hb.load_geotiff_chunk_by_bb(p.coarse_ha_per_cell_path, p.bb, output_path=p.coarse_aoi_ha_per_cell_path)

        for base_year in p.base_years:

            for class_label in p.class_labels:
                luh_dir = os.path.join(p.luh2_extraction_dir, 'baseline', str(base_year))
                current_path = os.path.join(p.cur_dir, 'baseline', str(base_year), class_label +  '.tif')

                arrays = []

                for luh_label in p.seals7_to_luh2_correspondence[class_label]:
                    a = hb.load_geotiff_chunk_by_bb(os.path.join(luh_dir, luh_label + '.tif'), p.bb)
                    arrays.append(a)
                ndv = hb.get_ndv_from_path(os.path.join(luh_dir, luh_label + '.tif'))
                # LEARNING POINT: in the np.sum list comprehension, need an extra , 0 to indicate that it is a sum ALONG the list comprehensions' dimension. Otherwise, this will return a singleton sum.
                
                sum_of_class_proportions = np.zeros(a.shape)
                for a in arrays:
                    sum_of_class_proportions += a

                hb.create_directories(current_path)

                hectares = np.where(arrays[0] != ndv, sum_of_class_proportions * p.ha_per_cell_array, ndv)
                hb.save_array_as_geotiff(hectares, current_path, p.coarse_aoi_ha_per_cell_path)

        for luh_scenario_label in p.luh_scenario_labels:
            for scenario_year in p.scenario_years:
                for class_label in p.class_labels:
                    luh_dir = os.path.join(p.luh2_extraction_dir, luh_scenario_label, str(scenario_year))
                    current_path = os.path.join(p.cur_dir, luh_scenario_label, str(scenario_year), class_label + '.tif')

                    arrays = []
                    for luh_label in p.seals7_to_luh2_correspondence[class_label]:
                        a = hb.load_geotiff_chunk_by_bb(os.path.join(luh_dir, luh_label + '.tif'), p.bb)
                        arrays.append(a)
                    ndv = hb.get_ndv_from_path(os.path.join(luh_dir, luh_label + '.tif'))
                    sum_of_class_proportions = np.zeros(a.shape)
                    for a in arrays:
                        sum_of_class_proportions += a

                    hectares = np.where(arrays[0] != ndv, sum_of_class_proportions * p.ha_per_cell_array, ndv)
                    hb.create_directories(current_path)
                    hb.save_array_as_geotiff(hectares, current_path, p.coarse_aoi_ha_per_cell_path)


def seals7_difference_from_base_year(p):
    # p.luh2_extraction_dir = r'C:\Files\Research\base_data\ipbes\processed_data\luh2_extraction_'
    # p.match_15m_float_path = os.path.join(p.luh2_extraction_dir, 'baseline', str(p.base_year), 'c3ann.tif')


    if p.run_this:
        # p.match_15m_float_af = hb.ArrayFrame(p.match_15m_float_path)
        for scenario_year in p.scenario_years:  # Skip first cause it is 2015
            for scenario_name in p.luh_scenario_labels:
                state_baseline_dir = os.path.join(p.luh2_as_seals7_proportion_dir, 'baseline', str(p.base_year))
                states_future_dir = os.path.join(p.luh2_as_seals7_proportion_dir, scenario_name, str(scenario_year))
                for file_path in hb.list_filtered_paths_nonrecursively(state_baseline_dir, include_extensions='.tif'):
                    file_root = hb.file_root(file_path)
                    state_baseline_path = os.path.join(state_baseline_dir, file_root + '.tif')
                    proportion_states_baseline_array = hb.load_geotiff_chunk_by_bb(state_baseline_path, p.bb)
                    proportion_states_baseline_ndv = hb.get_ndv_from_path(state_baseline_path)
                    state_future_path = os.path.join(states_future_dir, file_root + '.tif')
                    proportion_states_future_array = hb.load_geotiff_chunk_by_bb(state_future_path, p.bb)

                    # proportion_states_baseline = hb.ArrayFrame(state_baseline_path)
                    # proportion_states_future = hb.ArrayFrame(state_future_path)

                    # NOTE, tested several methods and this was the fastest and most flexible. Note that because it doesn't call valid_mask, that never has to load
                    difference_array = np.where(proportion_states_baseline_array != proportion_states_baseline_ndv, proportion_states_future_array - proportion_states_baseline_array, proportion_states_baseline_ndv)

                    difference_dir = os.path.join(p.cur_dir, scenario_name, str(scenario_year))
                    hb.create_directories(difference_dir)

                    difference_path = os.path.join(difference_dir, file_root + '_' + str(scenario_year) + '_' + str(p.base_year) + '_ha_difference.tif')
                    difference = hb.save_array_as_geotiff(difference_array, difference_path, p.coarse_aoi_ha_per_cell_path)

