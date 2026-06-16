import collections
import copy
import logging
import math
import multiprocessing
import os
import sys
import time
import warnings
from collections import OrderedDict

import hazelbean as hb
import hazelbean.pyramids
import matplotlib
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from osgeo import gdal

# from hazelbean.pyramids import *
# from hazelbean import *


# from seals_visualization_functions import *


# IMPORT NOTE: This shuold not be needed if the user is installing to site-packages via PIP or whatever. However, to link to the developer version,
# It would always break cross-referencing SEALS vs GTAP-InVEST. Putting this before a relative import fixes it.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(parent_dir)))


from seals import seals_utils
import pandas as pd
import time
import math
from hazelbean import config as hb_config

try:
    from setuptools import Extension
    from setuptools import setup
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

L = hb_config.get_logger()


env_name = sys.executable.split(os.sep)[-2]

from seals import seals_utils

### Currently disabled because now i have people do it on install.
# if env_name is not None:
#     try:
#         seals_utils.recompile_cython(env_name)
#     except:
#         raise NameError('Failed to import a cython-enabled library. You can try manually recompiling the library with "python compile_cython_files.py build_ext --inplace". This requires a C compiler installed. See https://justinandrewjohnson.com/earth_economy_devstack/installation.html. ')

try:
    from seals.seals_cython_functions import calibrate as calibrate
except:
    print('Failed to import a cython-enabled library. You can try manually recompiling the library with "python compile_cython_files.py build_ext --inplace". This requires a C compiler installed. See https://justinandrewjohnson.com/earth_economy_devstack/installation.html. Alternatively, you might want to clone the git repo and then install it as an editable install via  pip install -e .')

try:
    from seals import seals_cython_functions as seals_cython_functions
except:
    print('Failed to import a cython-enabled library. You can try manually recompiling the library with "python compile_cython_files.py build_ext --inplace". This requires a C compiler installed. See https://justinandrewjohnson.com/earth_economy_devstack/installation.html. Alternatively, you might want to clone the git repo and then install it as an editable install via  pip install -e .')

try:
    from seals.seals_cython_functions import calibrate_from_change_matrix
except:
    print('Failed to import a cython-enabled library. You can try manually recompiling the library with "python compile_cython_files.py build_ext --inplace". This requires a C compiler installed. See https://justinandrewjohnson.com/earth_economy_devstack/installation.html. Alternatively, you might want to clone the git repo and then install it as an editable install via  pip install -e .')


def initialize_tasks(p):
    # how do tasks? Do this vs.
    5


def full_change_matrices(passed_p=None):
    if passed_p is None:
        global p
    else:
        p = passed_p

    if p.run_this:

        src_dir = p.stitched_lulc_simplified_scenarios_dir


        if p.scenario_definitions_path is not None:
            p.scenarios_df = pd.read_csv(p.scenario_definitions_path)

            lulc_1_path = None
            for index, row in p.scenarios_df.iterrows():
                seals_utils.assign_df_row_to_object_attributes(p, row)
                seals_utils.set_derived_attributes(p)

                if p.scenario_type !=  'baseline':
                    for c, year in enumerate(p.years):

                        # Get the correct starting year path
                        lulc_1_path = os.path.join(src_dir, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(p.years[c-1]) + '.tif')
                        if not hb.path_exists(lulc_1_path):
                            lulc_1_path = os.path.join(src_dir, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.model_label + '_' + str(p.key_base_year) + '.tif')


                        # if lulc_1_path is None:
                        #     lulc_1_path = os.path.join(src_dir, 'lulc_baseline_' + p.model_label + '_' + str(p.key_base_year) + '.tif')
                        # else:
                        #     lulc_1_path = os.path.join(src_dir, 'lulc_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(p.years[c-1]) + '.tif')

                        lulc_2_path = os.path.join(src_dir, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(year) + '.tif')

                        coarse_ha_per_cell_path = p.aoi_ha_per_cell_coarse_path
                        classes_that_might_change = p.changing_class_indices
                        output_dir = os.path.join(p.cur_dir, str(year))

                        if not hb.path_exists(output_dir): # Note shortcut of only checking for folder
                            hb.create_directories(output_dir)
                            seals_utils.calc_observed_lulc_change_for_two_lulc_paths(lulc_1_path, lulc_2_path, coarse_ha_per_cell_path, classes_that_might_change, output_dir)

def target_zones_matrices(p):

    if p.run_this:

        if p.scenario_definitions_path is not None:
            p.scenarios_df = pd.read_csv(p.scenario_definitions_path)

            for index, row in p.scenarios_df.iterrows():
                seals_utils.assign_df_row_to_object_attributes(p, row)
                seals_utils.set_derived_attributes(p)

                if p.scenario_type !=  'baseline':

                    zones_to_plot = 'first' # one of first, all, or four
                    if zones_to_plot == 'all':
                        target_zones = p.global_processing_blocks_list
                        offsets = [p.global_coarse_blocks_list[0]]
                        offsets = [[int(i) for i in j] for j in target_zones]

                    elif zones_to_plot == 'four':
                        target_zones = [p.global_processing_blocks_list[0], p.global_processing_blocks_list[int(len(p.global_processing_blocks_list)/4)], p.global_processing_blocks_list[int(len(p.global_processing_blocks_list)/2)], p.global_processing_blocks_list[int(len(p.global_processing_blocks_list)*3/4)]]
                        offsets = [p.global_coarse_blocks_list[0]]
                        offsets = [[int(i) for i in j] for j in target_zones]



                    elif zones_to_plot == 'first':
                        target_zones = [p.global_processing_blocks_list[0]]
                        offsets = [p.global_coarse_blocks_list[0]]
                        offsets = [[int(i) for i in offsets[0]]]
                    else:
                        raise ValueError('zones_to_plot must be one of first, all, or four')


                    # full_change_matrix_no_diagonal = hb.as_array(full_change_matrix_no_diagonal_path)

                    for c, offset in enumerate(offsets):
                        target_zone = target_zones[c]
                        target_zone_string = str(target_zone)
                        # target_zone_string = target_zone[0] + '_' + target_zone[1]

                        # full_change_matrix_no_diagonal = hb.load_geotiff_chunk_by_cr_size(full_change_matrix_no_diagonal_path, offset)
                        # src_dir = os.path.join(p.allocation_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(p.year), offset[0] + '_' + offset[1])
                        for year_c, year in enumerate(p.years):
                            src_dir = os.path.join(p.intermediate_dir, 'allocations', p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year), 'allocation_zones', target_zone_string, 'allocation')


                            # Get the correct starting year path
                            lulc_1_path = os.path.join(src_dir, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(p.years[c-1]) + '.tif')
                            if not hb.path_exists(lulc_1_path):
                                if year_c == 0:
                                    previous_year = p.key_base_year
                                    correct_dir_year = year
                                    previous_dir = os.path.join(p.intermediate_dir, 'allocations', p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(correct_dir_year), 'allocation_zones', target_zone_string, 'allocation')
                                    lulc_1_path = os.path.join(previous_dir, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.model_label + '_' + str(previous_year) + '.tif')


                                else:
                                    previous_year = p.years[year_c - 1]
                                    correct_dir_year = previous_year
                                    previous_dir = os.path.join(p.intermediate_dir, 'allocations', p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(correct_dir_year), 'allocation_zones', target_zone_string, 'allocation')
                                    lulc_1_path = os.path.join(previous_dir, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(previous_year) + '.tif')




                            # if lulc_1_path is None:
                            #     lulc_1_path = os.path.join(src_dir, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.model_label + '_' + str(p.key_base_year) + '.tif')
                            # else:
                            #     lulc_1_path = os.path.join(src_dir, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(p.years[c-1]) + '.tif')

                            lulc_2_path = os.path.join(src_dir, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(year) + '.tif')

                            coarse_ha_per_cell_path = p.aoi_ha_per_cell_coarse_path
                            classes_that_might_change = p.changing_class_indices
                            output_dir = os.path.join(p.cur_dir, str(year))

                            if not hb.path_exists(output_dir): # Note shortcut of only checking for folder
                                hb.create_directories(output_dir)
                                seals_utils.calc_observed_lulc_change_for_two_lulc_paths(lulc_1_path, lulc_2_path, coarse_ha_per_cell_path, classes_that_might_change, output_dir)



def combined_trained_coefficients(p):

    # Can reuse this task to extract from a different project by setting below. Otherwise it draws from the current project.
    p.project_override_for_extraction = None
    # p.project_override_for_extraction = 'seals_manuscript_jaj_workstation2022'

    if p.project_override_for_extraction is not None:
        current_project_name = p.project_override_for_extraction
        extraction_dir = os.path.join(p.project_dir, '..', p.project_override_for_extraction)
    else:
        current_project_name = hb.file_root(p.project_dir)
        extraction_dir = p.project_dir


    calibration_dir = os.path.join(extraction_dir, 'intermediate', 'calibration')

    p.combined_calibration_file_path = os.path.join(p.cur_dir, 'trained_coefficients_' + current_project_name + ' .csv')

    if p.run_this:

        if not hb.path_exists(p.combined_calibration_file_path):

            block_indices = hb.get_global_block_list_indices_from_block_size(p.processing_resolution)

            extant_block_calibration_paths = []

            df = None
            list_of_dfs = []
            for c, block_index in enumerate(block_indices):
                block_calibration_path = os.path.join(calibration_dir, str(block_index[0]) + '_' + str(block_index[1]), 'calibration_zones', 'trained_coefficients_zone_' + str(block_index[0]) + '_' + str(block_index[1]) + '.csv')
                if hb.path_exists(block_calibration_path):
                    extant_block_calibration_paths.append(block_calibration_path)

                    new_df = pd.read_csv(block_calibration_path, header=0)
                    block_index_df_input_list = [str(int(block_indices[c][0])) + '_' + str(int(block_indices[c][1])) + '_' + str(int(p.processing_resolution)) + '_' + str(int(p.processing_resolution))] * len(new_df)
                    new_df['calibration_block_index'] = block_index_df_input_list
                    list_of_dfs.append(new_df)
                    # hb.print_in_place('Reading calibration files: ' + str(c / len(block_indices) * 100.0) + '% ' + block_calibration_path)

            # LEARNING POINT: When concatenating a donkload of dataframes, calling concat once on a long list of DFs is fastest.
            df = pd.concat(list_of_dfs, axis=0, ignore_index=True)

            hb.log('extract_calibration_from_project() found ' + str(len(extant_block_calibration_paths)) + ' calibration files.')
            df.to_excel(p.combined_calibration_file_path)



# def calibration_generated_inputs(p):
#     """DEPRECATED IN FAVOR OF regressors_starting_values, possibly can delete. Create an xls with starting-guess parameters for the SEALS calibration run. This identifies where the
#     input data (and generated base data) is stored, parsed to the class-simpliicaiton scheme used."""

#     p.coefficients_training_starting_value_path = os.path.join(p.cur_dir, 'coefficients_training_starting_value.csv')
#     if p.run_this:

#         column_headers = ['spatial_regressor_name', 'data_location', 'type']
#         column_headers.extend(['class_' + str(i) for i in p.class_labels])

#         df_input_2d_list = []


#         # Write the default starting coefficients

#         # Set Multiplicative (constraint) coefficients
#         # For barren and water binaries set it to zero or 1 for others.
#         for c, label in enumerate(p.all_class_labels):
#             row = [label + '_presence_constraint', os.path.join(
#                 p.base_data_dir, 'lulc', 'esa', p.lulc_simplification_label, 'binaries', str(p.training_start_year), 'class_' + str(p.all_class_indices[c]) + '.tif'),
#                    'multiplicative'] + \
#                   [0 if i == p.all_class_indices[c] or p.all_class_indices[c] in [6, 7] else 1 for i in p.class_indices]
#             df_input_2d_list.append(row)

#         # Set additive coefficients
#         # for class binaries
#         for c, label in enumerate(p.all_class_labels):
#             row = [label + '_presence', os.path.join(
#                 p.base_data_dir, 'lulc', 'esa', 'simplified', 'binaries', str(p.training_start_year), 'class_' + str(p.all_class_indices[c]) + '.tif'),
#                    'additive'] + [0 if i == p.all_class_indices[c] else 0 for i in p.class_indices]
#             df_input_2d_list.append(row)

#         # for class convolutions of sigma 1
#         for c, label in enumerate(p.all_class_labels):
#             row = [label + '_gaussian_1', os.path.join(
#                 p.base_data_dir, 'lulc', 'esa', 'simplified', 'convolutions', str(p.training_start_year), 'class_' + str(p.all_class_indices[c]) + '_gaussian_' + str(1) + '.tif'),
#                    'additive'] + [0 if i == p.all_class_indices[c] else 0 for i in p.class_indices]
#             df_input_2d_list.append(row)

#         # for class convolutions of sigma 5, set to zero except for diagonal (self edge expansion)
#         for c, label in enumerate(p.all_class_labels):
#             row = [label + '_gaussian_5', os.path.join(
#                 p.base_data_dir, 'lulc', 'esa', 'simplified', 'convolutions',  str(p.training_start_year), 'class_' + str(p.all_class_indices[c]) + '_gaussian_' + str(5) + '.tif'),
#                    'additive'] + [1 if i == p.all_class_indices[c] else 0 for i in p.class_indices]
#             df_input_2d_list.append(row)

#         # for all static variables, set to zero, except for as a hack one of them so that the it is edefined everyone.
#         for static_regressor_label, path in p.static_regressor_paths.items():
#             row = [static_regressor_label, path,
#                    'additive'] + [1 if static_regressor_label == 'soil_organic_content' else 0 for i in p.class_indices]
#             df_input_2d_list.append(row)

#         df = pd.DataFrame(df_input_2d_list, columns=column_headers)
#         df.set_index('spatial_regressor_name', inplace=True)

#         df.to_csv(p.coefficients_training_starting_value_path)



def calibration(p):

    # BROKEN AFTER SERIALIZATION
    # Need to set projected_coarse_change_dir differently for magpie vs gtap runs because they have different tasks that generate their coarse change files.
    # if p.is_magpie_run:
    #     p.projected_coarse_change_dir = os.path.join(p.magpie_as_simplified_proportion_dir, p.luh_scenario_labels[0], str(p.base_years[0]))
    # elif p.is_gtap1_run:
    #     p.projected_coarse_change_dir = os.path.join(p.gtap_results_joined_with_luh_change_pnas_dir, p.current_luh_scenario_label, str(p.current_year), p.current_policy_scenario_label)
    # elif p.is_calibration_run:
    #     p.projected_coarse_change_dir = os.path.join(p.coarse_simplified_ha_difference_from_base_year_dir, p.luh_scenario_labels[0], str(p.scenario_years[0]))
    # elif p.is_standard_seals_run:
    #     p.projected_coarse_change_dir = p.input_dir
    # else:
    #     raise NameError('Unhandled.')

    # Generate lists of which zones change and thus need to be rerun
    p.combined_block_lists_paths = {
        'fine_blocks_list': os.path.join(p.cur_dir, 'fine_blocks_list.csv'),
        'coarse_blocks_list': os.path.join(p.cur_dir, 'coarse_blocks_list.csv'),
        'processing_blocks_list': os.path.join(p.cur_dir, 'processing_blocks_list.csv'),
        'global_fine_blocks_list': os.path.join(p.cur_dir, 'global_fine_blocks_list.csv'),
        'global_coarse_blocks_list': os.path.join(p.cur_dir, 'global_coarse_blocks_list.csv'),
        'global_processing_blocks_list': os.path.join(p.cur_dir, 'global_processing_blocks_list.csv'),
    }

    try:
        if all(hb.path_exists(i) for i in p.combined_block_lists_paths):
            blocks_lists_already_exist = True
        else:
            blocks_lists_already_exist = False
    except:
        blocks_lists_already_exist = False

    if blocks_lists_already_exist:
        fine_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'fine_blocks_list.csv'), '2d_list'))
        coarse_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'coarse_blocks_list.csv'), '2d_list'))
        processing_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'processing_blocks_list.csv'), '2d_list'))
        global_fine_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'global_fine_blocks_list.csv'), '2d_list'))
        global_coarse_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'global_coarse_blocks_list.csv'), '2d_list'))
        global_processing_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'global_processing_blocks_list.csv'), '2d_list'))

    else:
        if p.aoi == 'global':

            fine_blocks_list = hb.get_global_block_list_from_resolution(p.processing_resolution, p.fine_resolution)
            coarse_blocks_list = hb.get_global_block_list_from_resolution(p.processing_resolution, p.coarse_resolution)
            processing_blocks_list = hb.get_global_block_list_from_resolution(p.processing_resolution, p.processing_resolution)
            global_fine_blocks_list = fine_blocks_list
            global_coarse_blocks_list = coarse_blocks_list
            global_processing_blocks_list = hb.get_global_block_list_from_resolution_and_bb(p.processing_resolution,
                                                                                            p.processing_resolution, p.bb)

        else:
            fine_blocks_list = hb.pyramids.get_subglobal_block_list_from_resolution_and_bb(p.processing_resolution, p.fine_resolution, p.bb)
            coarse_blocks_list = hb.pyramids.get_subglobal_block_list_from_resolution_and_bb(p.processing_resolution, p.coarse_resolution, p.bb)
            processing_blocks_list = hb.pyramids.get_subglobal_block_list_from_resolution_and_bb(p.processing_resolution, p.processing_resolution, p.bb)
            global_fine_blocks_list = hb.pyramids.get_global_block_list_from_resolution_and_bb(p.processing_resolution, p.fine_resolution, p.bb)
            global_coarse_blocks_list = hb.pyramids.get_global_block_list_from_resolution_and_bb(p.processing_resolution, p.coarse_resolution, p.bb)
            global_processing_blocks_list = hb.pyramids.get_global_block_list_from_resolution_and_bb(p.processing_resolution, p.processing_resolution, p.bb)

            # global_processing_blocks_list = hb.get_global_block_list_from_resolution(p.processing_resolution, p.coarse_resolution)
        hb.log('Length of iterator before pruning in task calibration:', len(fine_blocks_list))

    if p.subset_of_blocks_to_run is not None:
        old_fine_blocks_list = fine_blocks_list
        old_coarse_blocks_list = coarse_blocks_list
        old_processing_blocks_list = processing_blocks_list

        fine_blocks_list = []
        coarse_blocks_list = []
        processing_blocks_list = []

        for i in p.subset_of_blocks_to_run:
            fine_blocks_list.append(old_fine_blocks_list[i])
            coarse_blocks_list.append(old_coarse_blocks_list[i])
            processing_blocks_list.append(old_processing_blocks_list[i])

    hb.log('Length of iterator after considering subset_of_blocks_to_run:', len(fine_blocks_list))

    combined_block_lists_dict = {
        'fine_blocks_list': fine_blocks_list,
        'coarse_blocks_list': coarse_blocks_list,
        'processing_blocks_list': processing_blocks_list,
        'global_fine_blocks_list': global_fine_blocks_list,
        'global_coarse_blocks_list': global_coarse_blocks_list,
        'global_processing_blocks_list': global_processing_blocks_list,
    }

    if not all([hb.path_exists(i) for i in p.combined_block_lists_paths.values()]):

        # Pare down the number of blocks to run based on if there is change in the projected_coarse_change
        old_fine_blocks_list = fine_blocks_list
        old_coarse_blocks_list = coarse_blocks_list
        old_processing_blocks_list = processing_blocks_list
        old_global_fine_blocks_list = global_fine_blocks_list
        old_global_coarse_blocks_list = global_coarse_blocks_list
        old_global_processing_blocks_list = global_processing_blocks_list
        fine_blocks_list = []
        coarse_blocks_list = []
        processing_blocks_list = []
        global_fine_blocks_list = []
        global_coarse_blocks_list = []
        global_processing_blocks_list = []

        # Conceptual problem here: should i only run where there was observed changes or projected changes. I've decided that CALIBRATION should
        L.debug('Checking existing blocks for change in the LUH data and excluding if no change.')
        for c, block in enumerate(old_coarse_blocks_list):
            progress_percent = float(c) / float(len(old_coarse_blocks_list)) * 100.0
            print ( 'Percent finished: ' + str(progress_percent), end='\r', flush=False)
            skip = []
            current_coarse_change_rasters = []
            for class_label in p.class_labels:
                gen_path = os.path.join(p.projected_coarse_change_dir, class_label + '_' + str(p.scenario_years[0]) + '_' + str(p.training_end_year) + '_ha_difference.tif')
                current_coarse_change_rasters.append(gen_path)
            for path in current_coarse_change_rasters:
                block = old_coarse_blocks_list[c]
                a = hb.load_geotiff_chunk_by_cr_size(path, block)
                changed = np.where((a != 0) & (a != -9999.) & (~np.isnan(a)), 1, 0)
                hb.debug('Checking to see if there is change in ', path)
                if np.nansum(changed) == 0:
                    hb.debug('Skipping because no change in coarse projections:', path)
                    skip.append(True)
                else:
                    skip.append(False)

            if not all(skip):
                fine_blocks_list.append(old_fine_blocks_list[c])
                coarse_blocks_list.append(old_coarse_blocks_list[c])
                processing_blocks_list.append(old_processing_blocks_list[c])
                global_fine_blocks_list.append(old_global_fine_blocks_list[c])
                global_coarse_blocks_list.append(old_global_coarse_blocks_list[c])
                global_processing_blocks_list.append(old_global_processing_blocks_list[c])

        # Write the blockslists to csvs to avoid future reprocessing (actually is quite slow (2 mins) when 64000 tiles).
        for block_name, block_list in combined_block_lists_dict.items():
            hb.python_object_to_csv(block_list, os.path.join(p.cur_dir, block_name + '.csv'), '2d_list')

    else:
        fine_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'fine_blocks_list.csv'), '2d_list'))
        coarse_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'coarse_blocks_list.csv'), '2d_list'))
        processing_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'processing_blocks_list.csv'), '2d_list'))
        global_fine_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'global_fine_blocks_list.csv'), '2d_list'))
        global_coarse_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'global_coarse_blocks_list.csv'), '2d_list'))
        global_processing_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'global_processing_blocks_list.csv'), '2d_list'))

    hb.log('Length of iterator after removing non-changing zones:', len(fine_blocks_list))


    # Remove from iterator lists that have already been computed.
    old_fine_blocks_list = fine_blocks_list
    old_coarse_blocks_list = coarse_blocks_list
    old_processing_blocks_list = processing_blocks_list
    old_global_fine_blocks_list = global_fine_blocks_list
    old_global_coarse_blocks_list = global_coarse_blocks_list
    old_global_processing_blocks_list = global_processing_blocks_list
    fine_blocks_list = []
    coarse_blocks_list = []
    processing_blocks_list = []
    global_fine_blocks_list = []
    global_coarse_blocks_list = []
    global_processing_blocks_list = []

    for c, fine_block in enumerate(old_fine_blocks_list):
        tile_dir = str(fine_block[4]) + '_' + str(fine_block[5])
        expected_path = os.path.join(p.cur_dir, tile_dir, 'calibration_zones', 'trained_coefficients_zone_' + tile_dir + '.csv')
        if not hb.path_exists(expected_path):
            fine_blocks_list.append(old_fine_blocks_list[c])
            coarse_blocks_list.append(old_coarse_blocks_list[c])
            processing_blocks_list.append(old_processing_blocks_list[c])
            global_fine_blocks_list.append(old_global_fine_blocks_list[c])
            global_coarse_blocks_list.append(old_global_coarse_blocks_list[c])
            global_processing_blocks_list.append(old_global_processing_blocks_list[c])

    hb.log('Length of iterator after removing finished zones:', len(fine_blocks_list))

    # Process for each block which calibration file it should use.
    nyi = True

    if not nyi:
        # NOTE an interesting quirk here. Although I want to make sure nothing runs globally when there is a target AOI set
        #, I do let this one run globally because its fast, and then the aoi-specific run just needs to use the right ID.
        p.aezreg_zones_raster_path = os.path.join(p.cur_dir, 'aezreg_zones.tif')
        p.processing_zones_raster_path = os.path.join(p.cur_dir, 'processing_zones.tif')
        p.processing_zones_to_calibration_chunk_path = os.path.join(p.cur_dir, 'processing_zones_to_calibration_chunk.csv')
        p.processing_zones_match_path = p.match_paths[3600.0]
        if p.run_this:
            if not hb.path_exists(p.aezreg_zones_raster_path):
                hb.convert_polygons_to_id_raster(p.calibration_zone_polygons_path, p.aezreg_zones_raster_path, p.coarse_match_path, id_column_label='pyramid_id', data_type=5, ndv=-9999, all_touched=True, compress=True)
            if not hb.path_exists(p.processing_zones_raster_path):
                hb.convert_polygons_to_id_raster(p.calibration_zone_polygons_path, p.processing_zones_raster_path, p.processing_zones_match_path, id_column_label='pyramid_id', data_type=5, ndv=-9999, all_touched=True, compress=True)

            if not hb.path_exists(p.processing_zones_to_calibration_chunk_path):
                calibration_zones_to_calibration_chunk = {}

                zones_raster = hb.as_array(p.processing_zones_raster_path)
                uniques = np.unique(zones_raster)
                r, c = hb.calculate_zone_to_chunk_list_lookup_dict(zones_raster)

                zone_calibration_block_lookup_dict = {}
                for u in uniques[uniques != -9999]:
                    n_in_zone = len(r[u][r[u] > 0])
                    selected_id = math.floor(n_in_zone / 2)
                    zone_calibration_block_lookup_dict[u] = (r[u, selected_id], c[u, selected_id])


                with open(p.processing_zones_to_calibration_chunk_path, "w") as f:
                    for k, line in zone_calibration_block_lookup_dict.items():
                        f.write(str(k) + ',' + str(line[0]) + '_' + str(line[1]) + '\n')

    p.iterator_replacements = {}
    p.iterator_replacements['fine_blocks_list'] = fine_blocks_list
    p.iterator_replacements['coarse_blocks_list'] = coarse_blocks_list
    p.iterator_replacements['processing_blocks_list'] = processing_blocks_list
    p.iterator_replacements['global_fine_blocks_list'] = global_fine_blocks_list
    p.iterator_replacements['global_coarse_blocks_list'] = global_coarse_blocks_list
    p.iterator_replacements['global_processing_blocks_list'] = global_processing_blocks_list

    # Trickier replacement that will redefine the parent dir for each task so that it also WRITES in the correct output location

    # NOTE: global_processing_blocks_list is currently broken, writing all zero values to the CSV except for cols 5 and 6.
    p.iterator_replacements['cur_dir_parent_dir'] = [p.intermediate_dir+ '/calibration/' + str(i[0]) + '_' + str(i[1]) for i in global_processing_blocks_list]
    # p.iterator_replacements['cur_dir_parent_dir'] = [p.intermediate_dir + '/calibration/' + str(i[4]) + '_' + str(i[5]) for i in fine_blocks_list]

    p.iterator_replacements['current_scenario_pairing_label'] = ['rcp45_ssp2' for i in fine_blocks_list]
    p.iterator_replacements['current_year'] = [p.key_base_year for i in fine_blocks_list]
    p.iterator_replacements['current_policy_scenario_label'] = ['calibration' for i in fine_blocks_list]


    if p.run_only_first_element_of_each_iterator:
        p.iterator_replacements['fine_blocks_list'] = [p.iterator_replacements['fine_blocks_list'][0]]
        p.iterator_replacements['coarse_blocks_list'] = [p.iterator_replacements['coarse_blocks_list'][0]]
        p.iterator_replacements['processing_blocks_list'] = [p.iterator_replacements['processing_blocks_list'][0]]
        p.iterator_replacements['global_fine_blocks_list'] = [p.iterator_replacements['global_fine_blocks_list'][0]]
        p.iterator_replacements['global_coarse_blocks_list'] = [p.iterator_replacements['global_coarse_blocks_list'][0]]
        p.iterator_replacements['global_processing_blocks_list'] = [p.iterator_replacements['global_processing_blocks_list'][0]]
        p.iterator_replacements['cur_dir_parent_dir'] = [p.iterator_replacements['cur_dir_parent_dir'][0]]
        p.iterator_replacements['current_scenario_pairing_label'] = [p.iterator_replacements['current_scenario_pairing_label'][0]]
        p.iterator_replacements['current_year'] = [p.iterator_replacements['current_year'][0]]
        p.iterator_replacements['current_policy_scenario_label'] = [p.iterator_replacements['current_policy_scenario_label'][0]]

def calibration_prepare_lulc(passed_p=None):
    if passed_p is None:
        global p
    else:
        p = passed_p

    p.zone_esa_simplified_lulc_base_year_path = os.path.join(p.cur_dir, 'zone_esa_'+p.lulc_simplification_label + '_lulc_' + str(p.key_base_year) + '.tif')

    p.chunk_ha_per_cell_course_path = os.path.join(p.cur_dir, 'chunk_ha_per_cell_coarse.tif')

    p.lulc_class_types_path = r"C:\OneDrive\Projects\cge\seals\projects\ipbes\input\lulc_class_types.csv"

    # Problem here: Change vector method needs to be expanded to Change matrix, full from-to relationships
    # but when doing from-to, that only works when doing observed time-period validation. What would be the assumption for going into
    # the future? Possibly attempt to match prior change matrices, but only as a slight increase in probability? Secondly, why is my
    # search algorithm not itself finding the from-to relationships just by minimizing difference? Basically, need to take seriously deallocation.

    full_change_matrix_no_diagonal_path = os.path.join(p.cur_dir, 'full_change_matrix_no_diagonal.tif')

    if p.run_this:
        from hazelbean.calculation_core.cython_functions import \
            calc_change_matrix_of_two_int_arrays

        # Clip ha_per_cell and use it as the match
        # TODOO Is this needed? delete if so.
        chunk_ha_per_cell_coarse = hb.load_geotiff_chunk_by_cr_size(p.aoi_ha_per_cell_coarse_path, p.processing_blocks_list, output_path=p.chunk_ha_per_cell_course_path)

        p.chunk_ha_per_cell_coarse = hb.ArrayFrame(p.chunk_ha_per_cell_course_path)
        p.chunk_coarse_match = hb.ArrayFrame(p.chunk_ha_per_cell_course_path)

        # Get the processing zone ID
        current_processing_zone_path = os.path.join(p.cur_dir, 'processing_zone.tif')
        # processing_zone = hb.load_geotiff_chunk_by_cr_size(p.processing_zones_raster_path, p.global_processing_blocks_list, output_path=current_processing_zone_path)

        # calibration_zone =

        p.lulc_base_year_chunk_10sec_path = os.path.join(p.cur_dir, 'lulc_esa_' + p.lulc_simplification_label + '_' + str(p.key_base_year) + '.tif')
        p.lulc_training_start_year_chunk_10sec_path = os.path.join(p.cur_dir, 'lulc_esa_' + p.lulc_simplification_label + '_' + str(p.training_start_year) + '.tif')

        # Clip ha_per_cell and use it as the match
        hb.load_geotiff_chunk_by_cr_size(p.lulc_simplified_paths['lulc_esa_' + p.lulc_simplification_label + '_' + str(p.key_base_year)], p.fine_blocks_list, output_path=p.lulc_base_year_chunk_10sec_path)
        hb.load_geotiff_chunk_by_cr_size(p.lulc_simplified_paths['lulc_esa_' + p.lulc_simplification_label + '_' + str(p.training_start_year)], p.fine_blocks_list, output_path=p.lulc_training_start_year_chunk_10sec_path)
        p.fine_match = hb.ArrayFrame(p.lulc_base_year_chunk_10sec_path)

        p.ha_per_cell_coarse = hb.ArrayFrame(p.global_ha_per_cell_course_path)
        # p.coarse_match = hb.ArrayFrame(p.global_ha_per_cell_course_path)

        fine_cells_per_coarse_cell = round((p.chunk_ha_per_cell_coarse.cell_size / p.fine_match.cell_size) ** 2)
        aspect_ratio = int(p.fine_match.num_cols / p.chunk_coarse_match.num_cols)

        p.lulc_training_start_year_chunk = hb.ArrayFrame(p.lulc_training_start_year_chunk_10sec_path)

        # TODOOO later on, figure out why this failed.
        # if aspect_ratio != p.fine_match.num_cols / p.coarse_match.num_cols:
        #     hb.log('aspect ratio ' + str(aspect_ratio) + ' not same as non-inted version ' + str(p.fine_match.num_cols / p.coarse_match.num_cols) + '. This could indicate non pyramidal data.')

        if p.output_writing_level >= 3:
            p.calculate_change_matrix = 1
        else:
            p.calculate_change_matrix = 0
        if p.calculate_change_matrix or True: # I think this always needs to be run
            net_change_output_arrays = np.zeros((len(p.class_indices), p.chunk_coarse_match.shape[0], p.chunk_coarse_match.shape[1]))
            full_change_matrix = np.zeros((len(p.class_indices * p.chunk_coarse_match.n_rows), len(p.class_indices) * p.chunk_coarse_match.n_cols))
            full_change_matrix_no_diagonal = np.zeros((len(p.class_indices * p.chunk_coarse_match.n_rows), len(p.class_indices) * p.chunk_coarse_match.n_cols))
            for r in range(p.chunk_coarse_match.num_rows):
                for c in range(p.chunk_coarse_match.num_cols):

                    t1_subarray = p.lulc_training_start_year_chunk.data[int(r * aspect_ratio): int((r + 1) * aspect_ratio), int(c * aspect_ratio): int((c + 1) * aspect_ratio)]
                    t2_subarray = p.fine_match.data[int(r * aspect_ratio): int((r + 1) * aspect_ratio), int(c * aspect_ratio): int((c + 1) * aspect_ratio)]
                    # ha_per_cell_subarray = chunk_ha_per_cell_coarse.data[int(r * aspect_ratio): int((r + 1) * aspect_ratio), int(c * aspect_ratio): int((c + 1) * aspect_ratio)]

                    ha_per_cell_coarse_this_subarray = p.chunk_ha_per_cell_coarse.data[r, c]

                    change_matrix, counters = calc_change_matrix_of_two_int_arrays(t1_subarray.astype(np.int), t2_subarray.astype(np.int), p.class_indices)

                    vector = seals_utils.calc_change_vector_of_change_matrix(change_matrix)

                    ha_per_cell_this_subarray = p.chunk_ha_per_cell_coarse.data[r, c] / fine_cells_per_coarse_cell

                    if vector:
                        for i in p.class_indices:
                            net_change_output_arrays[i - 1, r, c] = vector[i - 1] * ha_per_cell_this_subarray
                    else:
                        net_change_output_arrays[i, r, c] = 0.0

                    n_classes = len(p.class_indices)
                    full_change_matrix[r * n_classes: (r + 1) * n_classes, c * n_classes: (c + 1) * n_classes] = change_matrix

                    # Fill diagonal with zeros.
                    for i in range(n_classes):
                        change_matrix[i, i] = 0

                    full_change_matrix_no_diagonal[r * n_classes: (r + 1) * n_classes, c * n_classes: (c + 1) * n_classes] = change_matrix

            for c, class_label in enumerate(p.class_labels):
                current_net_change_array_path = os.path.join(p.cur_dir, class_label + '_observed_change.tif')
                hb.save_array_as_geotiff(net_change_output_arrays[c], current_net_change_array_path, p.chunk_coarse_match.path)

            write_change_matrix_rasters = 1
            if write_change_matrix_rasters:
                calibration_full_change_matrix_path = os.path.join(p.cur_dir, 'calibration_full_change_matrix.tif')
                hb.save_array_as_geotiff(full_change_matrix, calibration_full_change_matrix_path, p.chunk_coarse_match.path, n_rows=full_change_matrix.shape[1], n_cols=full_change_matrix.shape[1])
                full_change_matrix_no_diagonal_path = os.path.join(p.cur_dir, 'full_change_matrix_no_diagonal.tif')
                hb.save_array_as_geotiff(full_change_matrix_no_diagonal, full_change_matrix_no_diagonal_path, p.chunk_coarse_match.path, n_rows=full_change_matrix_no_diagonal.shape[1], n_cols=full_change_matrix_no_diagonal.shape[1])

        # TODOO make this work between gtap1 and magpie. maybe by making it coarse_land_change?
        magpie_long_label = 'SSP2_test_cell.land_0.5_primother_share_'
        magpie_short_label = 'magpie'

        # Build a dict of where each LUC is projected at the coarse resolution.
        # In the event that there's a class that doesn't have a change scenario, just take from the underlying SSP/RCP map
        if p.is_magpie_run:
            current_class_labels = p.shortened_class_labels
        else:
            current_class_labels = p.class_labels

        p.projected_coarse_change_files_adjustment_run = {}
        if p.is_magpie_run:
            for scenario_label in ['baseline']: # TODOO This needs to be generalized to coarse_focusing_at_time_t etc
                p.projected_coarse_change_files_adjustment_run[scenario_label] = {}
                for year in p.key_base_years:
                    p.projected_coarse_change_files_adjustment_run[scenario_label][year] = {}
                    for policy_scenario_label in ['baseline']:
                        p.projected_coarse_change_files_adjustment_run[scenario_label][year][policy_scenario_label] = {}
                        for class_label in current_class_labels:
                            implied_magpie_path = os.path.join(p.projected_coarse_change_dir, magpie_short_label + '_' + class_label + '_' + str(p.key_base_year) + '_ha_difference_from_esa.tif')
                            p.projected_coarse_change_files_adjustment_run[scenario_label][year][policy_scenario_label][class_label] = implied_magpie_path


        p.projected_coarse_change_files = {}
        for luh_scenario_label in p.luh_scenario_labels: # TODOO This needs to be generalized to coarse_focusing_at_time_t etc
            p.projected_coarse_change_files[luh_scenario_label] = {}
            for year in p.scenario_years:
                p.projected_coarse_change_files[luh_scenario_label][year] = {}
                for policy_scenario_label in p.policy_scenario_labels:
                    p.projected_coarse_change_files[luh_scenario_label][year][policy_scenario_label] = {}
                    for class_label in current_class_labels:
                        if p.is_gtap1_run:
                            implied_gtap1_path = os.path.join(p.projected_coarse_change_dir, 'gtap1_' + class_label + '_ha_change.tif')
                            if hb.path_exists(implied_gtap1_path):
                                p.projected_coarse_change_files[luh_scenario_label][year][policy_scenario_label][class_label] = implied_gtap1_path
                            else:
                                implied_luh_path = os.path.join(p.coarse_simplified_ha_difference_from_previous_year_dir, luh_scenario_label, str(year),
                                                                class_label + '_' + str(year) + '_' + str(p.key_base_year) + '_difference.tif')
                                p.projected_coarse_change_files[luh_scenario_label][year][policy_scenario_label][class_label] = implied_luh_path
                        elif p.is_magpie_run:
                            implied_magpie_path = os.path.join(p.projected_coarse_change_dir, magpie_short_label + '_' + class_label + '_' + str(year) + '_' + str(p.key_base_year) + '_ha_difference.tif')
                            p.projected_coarse_change_files[luh_scenario_label][year][policy_scenario_label][class_label] = implied_magpie_path
                        else:
                            implied_ssp_path = os.path.join(p.projected_coarse_change_dir, class_label + '_' + str(year) + '_' + str(p.key_base_year) + '_ha_difference.tif')
                            p.projected_coarse_change_files[luh_scenario_label][year][policy_scenario_label][class_label] = implied_ssp_path

        # TODOO FIX THIS HACK. How should I deal with adjustment runs... would this fit into the overall dynamic scenario iteration?
        if 'esa_luh_baseline_lulc_adjustment' in p.cur_dir_parent_dir:
            p.projected_coarse_change_files = p.projected_coarse_change_files_adjustment_run
            p.is_first_pass = True
        else:
            p.is_first_pass = False

        hb.log('prepare_lulc looked for projected_coarse_change_files and found ', p.projected_coarse_change_files)
        if p.write_projected_coarse_change_chunks:
            for luh_scenario_label, v in p.projected_coarse_change_files.items():
                for year_label, vv in v.items():
                    for policy_scenario_label, vvv in vv.items():
                        for class_label, coarse_change_path in vvv.items():
                            current_net_change_array_path = os.path.join(p.cur_dir, class_label + '_projected_change.tif')

                            hb.load_geotiff_chunk_by_cr_size(coarse_change_path, p.coarse_blocks_list, output_path=current_net_change_array_path)

                            # STOPPED HERE, waiting for discussion from patrick:
                            # baseline adjustment is working, but the old change figs do'nt seem to look right. esp.
                            # C:\Files\Research\cge\gtap_invest\projects\test_magpie_seals\intermediate\magpie_as_simplified_proportion\rcp45_ssp2\2050\SSP2_BiodivPol_ClimPol_NCPpol_LPJmL5\magpie_forest_2050_2015_ha_difference.tif
                            # Also, need to hear from Patrick on just using change figs




def calibration_zones_logit(passed_p=None):

    NOTE = "ABANDONED RIGHT AFTER I GOT THE LOGIT TO RUN."


    if passed_p is None:
        global p
    else:
        p = passed_p

    final_coefficients_path = os.path.join(p.cur_dir, 'trained_coefficients_zone_' + os.path.split(os.path.split(p.cur_dir)[0])[1] + '.csv')

    if p.run_this:
        import hazelbean.stats
        import sklearn.model_selection
        from hazelbean.stats import RegressionFrame


        starting_coefficients_path = p.local_data_regressors_starting_values_path
        spatial_regressors_df = pd.read_csv(starting_coefficients_path)

        hb.log('calibration_zones_logit loaded local_data_regressors_starting_values_path from ' + str(p.local_data_regressors_starting_values_path) + ' of size ' + str(spatial_regressors_df.shape))

        # For now, I am declaring that including the word 'presence' in a regressor name means it is derived from the lulc map
        # and is not an independent variable in the logit regression.
        static_regressors_df = spatial_regressors_df.loc[~spatial_regressors_df['spatial_regressor_name'].str.contains('presence')]


        rf = RegressionFrame(project=p)

        df = None  # KEY VARIABLE. Set here to allow conditional loading of df if not set? How generalize this to multiple dfs?


        rf.max_cells_to_load_to_df = 1E7

        p.train_data_csv_path = os.path.join(p.cur_dir, 'train_data.csv')
        p.test_data_csv_path = os.path.join(p.cur_dir, 'test_data.csv')

        if hb.path_exists(p.train_data_csv_path):
            df = pd.read_csv(p.train_data_csv_path)

        # Reset results so that results are specific to this zone's run.
        rf.results = {}

        # p.current_bounding_box = [float(i) for i in p.region_bounding_boxes[p.zone_name].replace(' ', '').split(',')]

        p.regressions_to_run = {}


        # Iterate through each of the classes that we are going to predict
        for c, target_class_index in enumerate(p.class_indices):

            rf.dependent_variable_label = p.class_labels[c] + '_presence_constraint'
            dependent_row = spatial_regressors_df.loc[spatial_regressors_df['spatial_regressor_name'] == rf.dependent_variable_label]
            rf.dependent_variable_input_path = dependent_row['data_location'].values[0]

            rf.add_aligned_input(rf.dependent_variable_label, rf.dependent_variable_input_path)


            # Iterate through each row of the static regressors df
            for r in static_regressors_df.iterrows():

                regressor_type = r[1]['type']
                if 'gaussian' not in regressor_type and regressor_type == 'additive':
                    regressor_label = r[1]['spatial_regressor_name']
                    regressor_path = r[1]['data_location']

                    try:
                        hb.path_exists(regressor_path, verbose=False)
                    except:
                        raise NameError('RegressionFrame in seals_main unable to find on disk: ' + str(regressor_path))

                    rf.add_aligned_input(regressor_label, regressor_path)



            p.data_csv_path = os.path.join(p.cur_dir, 'tabular_data.csv')
            p.all_valid_sample_resolution_path = os.path.join(p.cur_dir, 'all_valid_sample_resolution.tif')

            # p.zone_dependent_variable_path = os.path.join(p.cur_dir, str(p.zone_name) + '_carbon_dependent_variable.tif')

            rf_path = os.path.join(p.cur_dir, 'rf.pkl')

            # We initialize it here because below there is an optimization that initialize_df_from_equation can append to an existing df passed in.
            # df = pd.DataFrame()
            summary = None

            p.standard_equation_string = rf.dependent_variable_label + """
            ~ intercept"""
            # p.standard_equation_string = rf.dependent_variable_label + """
            # ~ intercept"""

            # HACKY HERE: These variables made perfect summation to 1.
            p.skip_names = [
                'sand_percent',
                'nonforestnatural_presence_constraint',
                'urban_presence_constraint',
                'cropland_presence_constraint',
                'grassland_presence_constraint',
                'forest_presence_constraint',
                ]

            for k, v in rf.aligned_inputs.items():
                if k not in p.standard_equation_string and k not in p.skip_names:
                    p.standard_equation_string += ' + ' + k



            # Note option to append to an existing df passed in.
            # rf.train_test_split_strategy = 'spatial_tile'
            df, all_valid_array = rf.initialize_df_from_equation(p.standard_equation_string, p.bb, df)

            p.regressions_to_run[rf.dependent_variable_label + '_logit'] = p.standard_equation_string


        # Idea for next iteration of RegressionFrame: have nested options so that if one doesn't define regressions to run, it will just run a single one by default for each equation set?
        for name, equation in p.regressions_to_run.items():
            L.info('Running regression ' + str(name) + ' with equation ' + str(equation))

            # TODOO Fair amount of optimization/rationalization possible by not DROPPING things from the DF and just modifying the regression.
            df, all_valid_array = rf.initialize_df_from_equation(p.regressions_to_run[name], p.bb, df)

            for column in df.columns:
                depvar_as_2d_raster = rf.array_uncolumnize(df[column].values, all_valid_array.shape, method='quad_tile')

                plot_indvars = False
                if plot_indvars:
                    from hazelbean import visualization
                    output_path = os.path.join(p.cur_dir, column + '.png')
                    if not hb.path_exists(output_path):
                        hb.visualization.show(depvar_as_2d_raster, output_path=output_path)

            # # This is a hack. It's necessary to get rid of -inf, but I couldn't think of any way better to do it that doesn't require per-variable information and calculation.
            df_dropped = df.copy()
            drop_additional = False
            if drop_additional:
                # df_dropped[df_dropped == -9999.0] = np.nan
                df_dropped[df_dropped == 65535] = np.nan
                df_dropped[df_dropped == np.inf] = np.nan
                df_dropped[np.isinf(df_dropped)] = np.nan
                df_dropped[df_dropped <= -9999] = np.nan
                df_dropped[df_dropped >= 9999999999] = np.nan
                df_dropped = df_dropped.dropna()

            if 'lasso' in name:
                rf.run_lasso(name, df_dropped, rf.equation_dict, output_dir=p.cur_dir)
            elif 'logit' in name:
                rf.run_logit(name, df_dropped, rf.equation_dict, output_dir=p.cur_dir)

            else:
                summary, regression_result = rf.run_sm_lm(name, df_dropped, rf.equation_dict, output_dir=p.cur_dir)

                # Write summaries and df files
                write_summary_txt = 1
                summary_txt_path = os.path.join(p.cur_dir, name + '_summary.txt')
                if summary is not None and write_summary_txt:
                    hb.write_to_file(str(summary), summary_txt_path)

        # Save all valid array
        p.all_valid_path = os.path.join(p.cur_dir, 'all_valid.tif')
        rf.all_valid_path = p.all_valid_path

        # NOT WORKING because of shape is still sparse, but is used for projection
        # hb.save_array_as_geotiff(all_valid_array.reshape(zone_dependent_variable.shape), p.all_valid_path, p.zone_dependent_variable_path, data_type=1, ndv=255)


        p.final_data_csv_path = os.path.join(p.cur_dir, 'final_data.csv')
        save_data_csv = 1
        if save_data_csv:
            if not hb.path_exists(p.final_data_csv_path):
                try:
                    df_dropped.to_csv(p.final_data_csv_path)
                except:
                    L.critical('Unable to save df. Maybe it was None?')

        save_pickle = 0
        if  save_pickle:
            # Save the whole RF as a pickle. But first unload the data so we just save the important shit
            for k, v in rf.sources.items():
                v.data = None
            rf.save_to_path(rf_path)



def calibration_zones(passed_p=None):
    if passed_p is None:
        global p
    else:
        p = passed_p

    final_coefficients_path = os.path.join(p.cur_dir, 'trained_coefficients_zone_' + os.path.split(os.path.split(p.cur_dir)[0])[1] + '.csv')

    if p.run_this and not hb.path_exists(final_coefficients_path):
        starting_coefficients_path = p.local_data_regressors_starting_values_path
        spatial_regressors_df = pd.read_csv(starting_coefficients_path)

        spatial_layer_names = spatial_regressors_df['spatial_regressor_name'].values
        spatial_layer_paths = spatial_regressors_df['data_location'].values
        spatial_layer_types = spatial_regressors_df['type'].values

        # TODOO For now, this doesnt allow for more than just a single parameter per layer, but next phase would extend this so that it was class_1_1, class_1_2, class_2_1, class_2_2, etc.
        p.seals_class_names = spatial_regressors_df.columns.values[3:]

        p.lulc_baseline_path = os.path.join(p.cur_dir, 'lulc_baseline.tif')
        p.zone_esa_simplified_lulc_base_year_path = os.path.join(p.cur_dir, 'zone_esa_' + p.lulc_simplification_label + '_lulc_base_year.tif')
        p.lulc_ndv = hb.get_ndv_from_path(p.base_year_lulc_path)

        p.fine_match_path = p.lulc_baseline_path

        p.loss_function_sigma = np.float64(7.0)

        p.generation_parameters = OrderedDict()
        p.generation_parameter_notations = OrderedDict()

        generation_best_parameters = None

        additive_coefficients_modulo = .1
        additive_coefficients_modulo = 1.
        multiplicative_coefficients_modulo = .1

        # For now, i chose to just start with the gtap values so that i don't have to create a newly build right-size spreadsheet
        spatial_regressor_starting_coefficients_read = pd.read_csv(starting_coefficients_path, index_col=0)
        # spatial_regressor_starting_coefficients_read = pd.read_csv(os.path.join(p.input_dir, 'spatial_regressor_starting_coefficients.csv'), index_col=0)
        spatial_regressor_starting_coefficients = spatial_regressor_starting_coefficients_read[p.seals_class_names].values.astype(np.float64).T

        # TODOO I have inconsistent usage of p.lulc_simplification_label. Use this throughout.
        observed_lulc_array = hb.load_geotiff_chunk_by_cr_size(p.lulc_simplified_paths['lulc_esa_'+ p.lulc_simplification_label + '_' + str(p.training_end_year)], p.fine_blocks_list).astype(np.int64)
        p.lulc_ndv = hb.get_ndv_from_path(p.lulc_simplified_paths['lulc_esa_'+ p.lulc_simplification_label + '_' + str(p.training_end_year)])
        valid_mask_array = np.where((observed_lulc_array != p.lulc_ndv), 1, 0).astype(np.int64)

        p.observed_current_coarse_change_input_paths = hb.list_filtered_paths_nonrecursively(p.calibration_prepare_lulc_dir, include_strings='observed', include_extensions='.tif')


        coarse_n_c, coarse_n_r = int(p.coarse_blocks_list[2]), int(p.coarse_blocks_list[3])
        n_c, n_r = int(p.fine_blocks_list[2]), int(p.fine_blocks_list[3])

        spatial_layers_3d = np.zeros((len(spatial_layer_paths), n_r, n_c)).astype(np.float64)


        normalize_inputs = True
        for c, path in enumerate(spatial_layer_paths):
            hb.debug('Loading spatial layer at path ' + path)
            current_bb = hb.get_bounding_box(path)
            if current_bb == hb.global_bounding_box:
                correct_fine_block_list = p.global_fine_blocks_list
                correct_coarse_block_list = p.global_coarse_blocks_list
            else:
                correct_fine_block_list = p.fine_blocks_list
                correct_coarse_block_list = p.coarse_blocks_list
            if spatial_layer_types[c] == 'additive' or spatial_layer_types[c] == 'multiplicative':
                if normalize_inputs is True:
                    spatial_layers_3d[c] = hb.normalize_array(hb.load_geotiff_chunk_by_cr_size(path, correct_fine_block_list))
                else:
                    spatial_layers_3d[c] = hb.load_geotiff_chunk_by_cr_size(path, correct_fine_block_list)
            elif spatial_layer_types[c][0:8] == 'gaussian':
                # updated_path = os.path.join(p.calibration_zones_dir, 'class_' + spatial_layer_names[c].split('_')[1] + '_gaussian_' + spatial_layer_names[c].split('_')[3] + '_convolution.tif')
                # L.debug('updated_path', updated_path)
                L.debug('path', path)
                if normalize_inputs is True:
                    spatial_layers_3d[c] = hb.normalize_array(hb.load_geotiff_chunk_by_cr_size(path, correct_fine_block_list))
                else:
                    L.debug('fine_blocks_list', p.fine_blocks_list)
                    spatial_layers_3d[c] = hb.load_geotiff_chunk_by_cr_size(path, correct_fine_block_list)  # NOTE assumes already clipped
                    # spatial_layers_3d[c] = hb.as_array(path)  # NOTE assumes already clipped
            else:
                raise NameError('unspecified type')


        # Set how much change for each class needs to be allocated.
        coarse_change_matrix_2d = hb.as_array(p.calibration_full_change_matrix_path)
        change_matrix_edge_length = len(p.class_labels)
        coarse_change_matrix_4d = np.zeros((coarse_n_r, coarse_n_c, change_matrix_edge_length, change_matrix_edge_length))
        for coarse_r in range(coarse_n_r):
            for coarse_c in range(coarse_n_c):
                coarse_change_matrix_4d[coarse_r, coarse_c] = coarse_change_matrix_2d[coarse_r * change_matrix_edge_length: (coarse_r + 1) * change_matrix_edge_length,
                                                                                      coarse_c * change_matrix_edge_length: (coarse_c + 1) * change_matrix_edge_length]




        # observed_coarse_change_3d = np.zeros((len(p.observed_current_coarse_change_input_paths), coarse_n_r, coarse_n_c)).astype(np.float64)
        # for c, path in enumerate(p.observed_current_coarse_change_input_paths):
        #     # # Scaling is unnecessary if you use stricly pyramidal zones... but i'm not sure i want to lose this yet e.g. for intersecting zones and country boundaries.
        #     # unscaled = hb.as_array(path)
        #     # # unscaled = hb.load_geotiff_chunk_by_cr_size(path, p.coarse_blocks_list)
        #     #
        #     # p.proportion_valid_fine_per_coarse_cell = hb.calc_proportion_of_coarse_res_with_valid_fine_res(unscaled, valid_mask_array).astype(np.float64)
        #     #
        #     # scaled_proportion_to_allocate = p.proportion_valid_fine_per_coarse_cell * unscaled
        #     #
        #     # scaled_proportion_to_allocate_path = os.path.join(p.cur_dir, os.path.split(path)[1])
        #     # hb.save_array_as_geotiff(scaled_proportion_to_allocate, scaled_proportion_to_allocate_path, p.fine_match_path)
        #     # observed_coarse_change_3d[c] = scaled_proportion_to_allocate.astype(np.float64)
        #     #
        #     # unscaled = hb.load_geotiff_chunk_by_cr_size(path, p.coarse_blocks_list)
        #     # observed_coarse_change_3d[c] = unscaled.astype(np.float64)
        #     observed_coarse_change_3d[c] = hb.as_array(path).astype(np.float64)
        #


        # MAYBE STILL USEFUL?! Doesn't yet work with DataRef method
        plot_observed_coarse_change_3d = 1
        if plot_observed_coarse_change_3d:
            observed_coarse_change_3d = np.zeros((len(p.class_labels), coarse_n_r, coarse_n_c)).astype(np.float64)
            for c, class_label in enumerate(p.class_labels):
                path = os.path.join(p.cur_dir, '../calibration_prepare_lulc', class_label + '_observed_change.tif')
                observed_coarse_change_3d[c] = hb.as_array(path).astype(np.float64)
                
            from seals import seals_visualization_functions
            seals_visualization_functions.plot_coarse_change_3d(p.cur_dir, observed_coarse_change_3d)


        for generation_id in range(p.num_generations):
            hb.log('Starting generation ' + str(generation_id) + ' for location ' + str(p.fine_blocks_list))
            p.generation_parameters[generation_id] = OrderedDict()
            p.generation_parameter_notations[generation_id] = OrderedDict()


            # The first entry (try 0) always refers to the generation's starting parameters
            if generation_best_parameters is None:
                p.generation_parameters[generation_id][0] = np.copy(spatial_regressor_starting_coefficients)
                p.generation_parameter_notations[generation_id][0] = {'new_coefficient': '', 'old_coefficient': '', 'spatial_layer_label': '', 'spatial_layer_id': '', 'change_class_id': '', 'change_class_label': '', 'spatial_layer_type': ''}

            else:
                p.generation_parameters[generation_id][0] = np.copy(generation_best_parameters)
                p.generation_parameter_notations[generation_id][0] = {'new_coefficient': '', 'old_coefficient': '', 'spatial_layer_label': '', 'spatial_layer_id': '', 'change_class_id': '', 'change_class_label': '', 'spatial_layer_type': ''}



            # Mixed methods here: Eventually this should work with the scenarios tree input structure.
            # p.change_class_labels_list = [int(i.split('_')[1]) for i in spatial_regressors_df.columns[3:]]
            p.change_class_labels = np.asarray(p.class_indices, dtype=np.int64)  # For Cythonization, load these as the "labels", which is used for writing.

            # NOTE: This and the previous 3 attempts dealt with what I think is an inconsistency on how load_geotiff_chunk_by_cr_size deals with datatypes. Updating the environment made the 2nd one fail until I took out the datatype argument. I think it was expecting a gdal number.
            lulc_baseline_array = hb.load_geotiff_chunk_by_cr_size(p.lulc_simplified_paths['lulc_esa_' + p.lulc_simplification_label + '_' + str(p.training_start_year)], p.fine_blocks_list, output_path=p.lulc_baseline_path).astype(np.int64)
            # lulc_baseline_array = hb.load_geotiff_chunk_by_cr_size(p.lulc_simplified_paths['lulc_esa_simplified_' + str(p.training_start_year)], p.fine_blocks_list, datatype=np.int64, output_path=p.lulc_baseline_path).astype(np.int64)
            # lulc_baseline_array = hb.load_geotiff_chunk_by_cr_size(p.training_start_year_simplified_lulc_path, p.fine_blocks_list, datatype=np.int64, output_path=p.lulc_baseline_path).astype(np.int64)

            # Load spatial_regressors

            # # Consider adding a function like this to avoid all the loading bloat.
            # seals_utils.load_zone_fast_inputs(p)

            spatial_layer_names = spatial_regressors_df['spatial_regressor_name'].values
            spatial_layer_paths = spatial_regressors_df['data_location'].values
            spatial_layer_types = spatial_regressors_df['type'].values


            spatial_layer_types_to_codes = {'multiplicative': 1,
                                            'additive': 2,
                                            }
            # QUIRCK, adjacency is really just additive with preprocessing.
            # LEARNING POINT, Here I could have used the set-concatenate function of |=
            spatial_layer_types_to_codes.update({'gaussian_' + str(sigma): 2 for sigma in p.gaussian_sigmas_to_test})



            spatial_layer_function_types_1d = np.asarray([spatial_layer_types_to_codes[i] for i in spatial_layer_types], np.int64)

            # p.spatial_layer_chunk_paths = []
            # for c, path in enumerate(spatial_layer_paths):
            #     if spatial_regressors_df['type'].values[c] == 'gaussian_parametric_1':
            #         _, class_id, _, sigma = spatial_regressors_df['spatial_regressor_name'].values[c].split('_')
            #         filename = os.path.split(path)[1]
            #         spatial_chunk_path = os.path.join(p.cur_dir, os.path.split(spatial_regressors_df['data_location'].values[c])[1])
            #         # spatial_chunk_path = os.path.join(p.cur_dir, spatial_regressors_df['spatial_regressor_name'].values[c] + '.tif')
            #         if not os.path.exists(spatial_chunk_path):
            #             # hb.load_geotiff_chunk_by_cr_size(path, p.fine_blocks_list)
            #             hb.load_geotiff_chunk_by_cr_size(path, p.fine_blocks_list, output_path=spatial_chunk_path)
            #         p.spatial_layer_chunk_paths.append(spatial_chunk_path)
            #
            # spatial_layer_chunk_counter = 0
            # for c, class_label in enumerate(spatial_regressors_df['spatial_regressor_name'].values):
            #     if spatial_regressors_df['type'].values[c] == 'gaussian_parametric_1':
            #         _, class_id, _, sigma = class_label.split('_')
            #
            #         kernel_path = os.path.join(p.generated_kernels_dir, 'gaussian_' + str(sigma) + '.tif')
            #         output_path = os.path.join(p.cur_dir, class_label + '_convolution.tif')
            #
            #         # NOTE, fft_gaussian has to write to disk, which i think i have to embrace.
            #         if not os.path.exists(output_path):
            #             seals_utils.fft_gaussian(p.spatial_layer_chunk_paths[spatial_layer_chunk_counter], kernel_path, output_path, -9999.0, True)
            #
            #         spatial_layer_chunk_counter += 1

            # hb.clip_raster_by_cr_size(p.lulc_baseline_input.path, p.fine_blocks_list, p.lulc_baseline_path)

            # Load things that dont ever change over generations or final run
            hectares_per_grid_cell = hb.load_geotiff_chunk_by_cr_size(p.aoi_ha_per_cell_fine_path, p.fine_blocks_list).astype(np.float64)


            # Generate tries by +/- on each of the IxJ adjacency parameters
            try_id = 1 # Not zero because zero is the generation starting param
            for spatial_layer_id, spatial_layer_label in enumerate(spatial_layer_names):
                for change_class_id, change_class_label in enumerate(p.change_class_labels):
                    if spatial_layer_types[spatial_layer_id] == 'additive' or spatial_layer_types[spatial_layer_id][0:8] == 'gaussian':
                        # QUIRK, notice that we copy the generation starting parameters EACH TIME we update a parameter. This is so that we have a fresh, unmodified set for the single thing we change.
                        # Increment it down
                        p.generation_parameters[generation_id][try_id] = np.copy(p.generation_parameters[generation_id][0])
                        new_coefficient = p.generation_parameters[generation_id][0][change_class_id, spatial_layer_id] - additive_coefficients_modulo
                        p.generation_parameters[generation_id][try_id][change_class_id, spatial_layer_id] = new_coefficient
                        p.generation_parameter_notations[generation_id][try_id] = {'new_coefficient': new_coefficient, 'old_coefficient': new_coefficient + additive_coefficients_modulo, 'spatial_layer_label': spatial_layer_label, 'spatial_layer_id': spatial_layer_id, 'change_class_id': change_class_id, 'change_class_label': change_class_label, 'spatial_layer_type': spatial_layer_types[spatial_layer_id]}

                        try_id += 1

                        # Increment it up
                        p.generation_parameters[generation_id][try_id] = np.copy(p.generation_parameters[generation_id][0])
                        new_coefficient = p.generation_parameters[generation_id][0][change_class_id, spatial_layer_id] + additive_coefficients_modulo
                        p.generation_parameters[generation_id][try_id][change_class_id, spatial_layer_id] = new_coefficient

                        p.generation_parameter_notations[generation_id][try_id] = {'new_coefficient': new_coefficient, 'old_coefficient': new_coefficient - additive_coefficients_modulo, 'spatial_layer_label': spatial_layer_label, 'spatial_layer_id': spatial_layer_id, 'change_class_id': change_class_id, 'change_class_label': change_class_label, 'spatial_layer_type': spatial_layer_types[spatial_layer_id]}

                        try_id += 1
                    elif spatial_layer_types[spatial_layer_id] == 'multiplicative':
                        pass
                        # # Increment it down
                        # p.generation_parameters[generation_id][try_id] = np.copy(p.generation_parameters[generation_id][0])
                        # p.generation_parameters[generation_id][try_id][change_class_id, spatial_layer_id] \
                        #     = p.generation_parameters[generation_id][0][change_class_id, spatial_layer_id] * multiplicative_coefficients_modulo
                        # try_id += 1
                        #
                        # # Increment it up
                        # p.generation_parameters[generation_id][try_id] = np.copy(p.generation_parameters[generation_id][0])
                        # p.generation_parameters[generation_id][try_id][change_class_id, spatial_layer_id] \
                        #     = p.generation_parameters[generation_id][0][change_class_id, spatial_layer_id] / multiplicative_coefficients_modulo

            benchmark_score = None
            current_best_score = 1e+100

            try_scores = OrderedDict()
            try_coefficients = OrderedDict()

            # Run the model repeatedly, iterating through individual parameter changes
            hb.debug('Starting to run allocation iteratively for individual parameter changes. ')
            for k, spatial_layer_coefficients_2d in p.generation_parameters[generation_id].items():
                p.call_string = str(k) + '_' + str(generation_id)

                hb.debug('coarse_change_matrix_4d', coarse_change_matrix_4d.shape, coarse_change_matrix_4d.dtype)
                hb.debug('lulc_baseline_array', lulc_baseline_array.shape, lulc_baseline_array.dtype)
                hb.debug('spatial_layers_3d', spatial_layers_3d.shape, spatial_layers_3d.dtype)
                hb.debug('spatial_layer_coefficients_2d', spatial_layer_coefficients_2d.shape, spatial_layer_coefficients_2d.dtype)
                hb.debug('spatial_layer_function_types_1d', spatial_layer_function_types_1d.shape, spatial_layer_function_types_1d.dtype)
                hb.debug('valid_mask_array', valid_mask_array.shape, valid_mask_array.dtype)
                hb.debug('change_class_labels', p.change_class_labels.shape, p.change_class_labels.dtype)
                hb.debug('observed_lulc_array', observed_lulc_array.shape, observed_lulc_array.dtype)
                hb.debug('hectares_per_grid_cell', hectares_per_grid_cell.shape, hectares_per_grid_cell.dtype)
                hb.debug('cur_dir', p.cur_dir)
                hb.debug('calibration_reporting_level', p.calibration_reporting_level)
                hb.debug('call_string', p.call_string)

                # Run the model repeatedly, iterating through individual parameter changes
                overall_similarity_score, lulc_projected_array, overall_similarity_plot, class_similarity_scores, class_similarity_plots = \
                    calibrate_from_change_matrix(coarse_change_matrix_4d,
                              lulc_baseline_array,
                              spatial_layers_3d,
                              spatial_layer_coefficients_2d,
                              spatial_layer_function_types_1d,
                              valid_mask_array,
                              p.change_class_labels,
                              observed_lulc_array,
                              hectares_per_grid_cell,
                              p.cur_dir,
                              p.calibration_reporting_level,
                              p.loss_function_sigma,
                              p.call_string)

                # Calculate score adjusting for number of predicted chagnes

                # TODOO Review this logic
                new_array = lulc_baseline_array - lulc_projected_array
                uniques = hb.enumerate_array_as_odict(new_array)
                total_change = sum([v for kk, v in uniques.items() if kk != 0])
                weighted_score = total_change / (overall_similarity_score + 1)

                hb.log('  Sweep found score ' + str(weighted_score) + ' for ' + str(p.generation_parameter_notations[generation_id][k]['spatial_layer_label']) +
                       ' with coeff ' + str(p.generation_parameter_notations[generation_id][k]['new_coefficient']) + ' on class '
                       + str(p.generation_parameter_notations[generation_id][k]['change_class_label']) + ' for try ' + str(k) + ' on generation ' + str(generation_id))
                try_scores[k] = weighted_score

            # Iterate through all score-improving changes from best to worst, seeing if they further improve the score in conjunction.
            hb.log('Iterating through all scores based on intial sweep value, testing to see if they make improvements in combination.')
            ranked_tries = OrderedDict(sorted(try_scores.items(), key=lambda x: x[1], reverse=True))
            best_score = 0
            starting_spatial_layer_coefficients_2d = copy.deepcopy(p.generation_parameters[generation_id][0])
            kept_spatial_layer_coefficients_2d = copy.deepcopy(p.generation_parameters[generation_id][0])

            for k, score in ranked_tries.items():
                changed_spatial_layer_coefficients_2d = p.generation_parameters[generation_id][k]
                current_spatial_layer_coefficients_2d = np.where(changed_spatial_layer_coefficients_2d != starting_spatial_layer_coefficients_2d,
                                                                 changed_spatial_layer_coefficients_2d,
                                                                 kept_spatial_layer_coefficients_2d)

                overall_similarity_score, lulc_projected_array, overall_similarity_plot, class_similarity_scores, class_similarity_plots = \
                    calibrate_from_change_matrix(coarse_change_matrix_4d,
                              lulc_baseline_array,
                              spatial_layers_3d,
                              current_spatial_layer_coefficients_2d,
                              spatial_layer_function_types_1d,
                              valid_mask_array,
                              p.change_class_labels,
                              observed_lulc_array,
                              hectares_per_grid_cell,
                              p.cur_dir,
                              p.calibration_reporting_level,
                              p.loss_function_sigma,
                              p.call_string)

                new_array = lulc_baseline_array - lulc_projected_array
                uniques = hb.enumerate_array_as_odict(new_array)
                total_change = sum([v for k, v in uniques.items() if k != 0])
                weighted_score = total_change / (overall_similarity_score + 1)

                hb.log('  Score iterate found score ' + str(weighted_score) + ' for ' + str(p.generation_parameter_notations[generation_id][k]['spatial_layer_label']) +
                       ' with coeff ' + str(p.generation_parameter_notations[generation_id][k]['new_coefficient']) + ' on class '
                       + str(p.generation_parameter_notations[generation_id][k]['change_class_label']) + ' for try ' + str(k) + ' on generation ' + str(generation_id))

                if weighted_score > best_score:
                    kept_spatial_layer_coefficients_2d = copy.deepcopy(current_spatial_layer_coefficients_2d)

                    best_score = weighted_score

                    hb.log('    Score improved by adding in ' + str(p.generation_parameter_notations[generation_id][k]['spatial_layer_label']) +
                       ' with coeff ' + str(p.generation_parameter_notations[generation_id][k]['new_coefficient']) + ' on class '
                       + str(p.generation_parameter_notations[generation_id][k]['change_class_label']))

                    # while True:
                    for permutation_coefficient in [0.00001, 0.0001, 0.001, .01, .1, .5, .75, 1.5, 2, 10, 100, 1000, 10000, 100000]:
                        current_spatial_layer_coefficients_2d = np.where(changed_spatial_layer_coefficients_2d != starting_spatial_layer_coefficients_2d,
                                                                         changed_spatial_layer_coefficients_2d - additive_coefficients_modulo * permutation_coefficient,
                                                                         kept_spatial_layer_coefficients_2d)

                        overall_similarity_score, lulc_projected_array, overall_similarity_plot, class_similarity_scores, class_similarity_plots = \
                            calibrate_from_change_matrix(coarse_change_matrix_4d,
                                      lulc_baseline_array,
                                      spatial_layers_3d,
                                      current_spatial_layer_coefficients_2d,
                                      spatial_layer_function_types_1d,
                                      valid_mask_array,
                                      p.change_class_labels,
                                      observed_lulc_array,
                                      hectares_per_grid_cell,
                                      p.cur_dir,
                                      p.calibration_reporting_level,
                                      p.loss_function_sigma,
                                      p.call_string)

                        new_array = lulc_baseline_array - lulc_projected_array
                        uniques = hb.enumerate_array_as_odict(new_array)
                        total_change = sum([v for k, v in uniques.items() if k != 0])
                        weighted_score_1 = total_change / (overall_similarity_score + 1)

                        if weighted_score_1 > best_score:
                            kept_spatial_layer_coefficients_2d = copy.deepcopy(current_spatial_layer_coefficients_2d)
                            best_score = weighted_score_1
                            hb.log('      Found improvement in permutations by further scaling ' +
                                   str(p.generation_parameter_notations[generation_id][k]['spatial_layer_label'])
                                   + ' coefficient by ' + str(permutation_coefficient))


                        current_spatial_layer_coefficients_2d = np.where(changed_spatial_layer_coefficients_2d != starting_spatial_layer_coefficients_2d,
                                                                         changed_spatial_layer_coefficients_2d + additive_coefficients_modulo * permutation_coefficient,
                                                                         kept_spatial_layer_coefficients_2d)

                        overall_similarity_score, lulc_projected_array, overall_similarity_plot, class_similarity_scores, class_similarity_plots = \
                            calibrate_from_change_matrix(coarse_change_matrix_4d,
                                      lulc_baseline_array,
                                      spatial_layers_3d,
                                      current_spatial_layer_coefficients_2d,
                                      spatial_layer_function_types_1d,
                                      valid_mask_array,
                                      p.change_class_labels,
                                      observed_lulc_array,
                                      hectares_per_grid_cell,
                                      p.cur_dir,
                                      p.calibration_reporting_level,
                                      p.loss_function_sigma,
                                      p.call_string)


                        new_array = lulc_baseline_array - lulc_projected_array
                        uniques = hb.enumerate_array_as_odict(new_array)
                        total_change = sum([v for k, v in uniques.items() if k != 0])
                        weighted_score_2 = total_change / (overall_similarity_score + 1)

                        if weighted_score_2 > best_score:
                            kept_spatial_layer_coefficients_2d = copy.deepcopy(current_spatial_layer_coefficients_2d)
                            best_score = weighted_score_2
                            hb.log('      SECOND PASS Found improvement in permutations by further scaling ' +
                                   str(p.generation_parameter_notations[generation_id][k]['spatial_layer_label'])
                                   + ' coefficient by ' + str(permutation_coefficient))


            # After finding best parameters, need to run 1 last time at the end of the generation to save the right layer.
            overall_similarity_score, lulc_projected_array, overall_similarity_plot, class_similarity_scores, class_similarity_plots = \
                calibrate_from_change_matrix(coarse_change_matrix_4d,
                          lulc_baseline_array,
                          spatial_layers_3d,
                          kept_spatial_layer_coefficients_2d,
                          spatial_layer_function_types_1d,
                          valid_mask_array,
                          p.change_class_labels,
                          observed_lulc_array,
                          hectares_per_grid_cell,
                          p.cur_dir,
                          p.calibration_reporting_level,
                          p.loss_function_sigma,
                          p.call_string)


            if p.write_calibration_generation_arrays:
                p.lulc_projected_gen_path = os.path.join(p.cur_dir, 'lulc_projected_array_gen' + str(generation_id) + '.tif')
                hb.save_array_as_geotiff(lulc_projected_array, p.lulc_projected_gen_path, p.lulc_baseline_path, ndv=-9999., data_type=1)

                p.overall_similarity_plot_path = os.path.join(p.cur_dir, 'overall_similarity_plot_gen' + str(generation_id) + '.tif')
                hb.save_array_as_geotiff(overall_similarity_plot, p.overall_similarity_plot_path, p.lulc_baseline_path, ndv=-9999., data_type=6)

                for c, plot in enumerate(class_similarity_plots):
                    class_similarity_plot_path = os.path.join(p.cur_dir, 'class_' + p.class_labels[c] + '_similarity_plot.tif')
                    hb.save_array_as_geotiff(plot, class_similarity_plot_path, p.lulc_baseline_path, ndv=-9999., data_type=6)

            # Update for best at end of generation
            generation_best_parameters = copy.deepcopy(kept_spatial_layer_coefficients_2d)


            output_df_2 = copy.deepcopy(spatial_regressors_df)
            for c, class_name in enumerate(p.seals_class_names):
                output_df_2[class_name] = generation_best_parameters[c, 0:]

            output_df_2.to_csv(os.path.join(p.cur_dir, 'trained_coefficients_gen' + str(generation_id) + '.csv'), index=False)

        # Write final coefficients
        output_df_2.to_csv(final_coefficients_path, index=False)

        # Write final lulc
        p.lulc_projected_path = os.path.join(p.cur_dir, 'lulc_projected.tif')
        hb.save_array_as_geotiff(lulc_projected_array, p.lulc_projected_path, p.lulc_baseline_path, ndv=-9999., data_type=1)

    else:
        pass
        # NOTE, this doesnt persist because it is an iterated child.
        # p.generation_best_parameters = pd.read_csv(final_coefficients_path)
        # output_df_2.to_csv(final_coefficients_path, index=False)
def calibration_plots(passed_p=None):
    if passed_p is None:
        global p
    else:
        p = passed_p

    if p.run_this:

        for c, class_label in enumerate(p.class_labels):
            baseline_array = hb.as_array(os.path.join(p.cur_dir, '../calibration_prepare_lulc', 'lulc_esa_' + p.lulc_simplification_label + '_' + str(p.training_start_year) + '.tif'))
            observed_array = hb.as_array(os.path.join(p.cur_dir, '../calibration_prepare_lulc', 'lulc_esa_' + p.lulc_simplification_label + '_' + str(p.key_base_year) + '.tif'))
            # os.path.join(p.cur_dir, '../calibration_zones', 'lulc_simplified_projected.tif')

            listed_paths = hb.list_filtered_paths_nonrecursively(os.path.join(p.cur_dir, '../calibration_zones'), include_strings='lulc_projected', include_extensions='.tif')
            projected_path_last_gen = sorted(listed_paths)[-1]
            projected_array = hb.as_array(projected_path_last_gen)
            lulc_class = p.class_indices[c]
            difference_metric_path = os.path.join(p.cur_dir, '../calibration_zones', 'class_' + p.class_labels[c] + '_similarity_plot.tif')

            also_plot_binary_results = 0
            if also_plot_binary_results: # Requires configuring above.
                change_array = hb.as_array(os.path.join(p.cur_dir, '../calibration_prepare_lulc', class_label + '_observed_change.tif'))
                annotation_text = "asdf"
                output_path = os.path.join(p.cur_dir, class_label + '_calibration_plot.png')
                similarity_array = hb.as_array(difference_metric_path)
                from seals import seals_visualization_functions
                seals_visualization_functions.show_lulc_class_change_difference(baseline_array, observed_array, projected_array, lulc_class, similarity_array, change_array, annotation_text, output_path)


def allocations(p):
    # Iterate through different scenarios to be allocated.
    """Create task to group downscaling of different scenarios."""

    if p.run_this:
        p.iterator_replacements = collections.OrderedDict()
        p.iterator_replacements['scenario_type'] = []
        p.iterator_replacements['exogenous_label'] = []
        p.iterator_replacements['climate_label'] = []
        p.iterator_replacements['model_label'] = []
        p.iterator_replacements['counterfactual_label'] = []
        p.iterator_replacements['year'] = []
        p.iterator_replacements['key_base_year'] = []
        p.iterator_replacements['previous_year'] = []

        if hasattr(p, 'seals_years'):
            p.iterator_replacements['seals_year'] = []

        if hasattr(p, 'seals_key_base_year'):
            p.iterator_replacements['seals_key_base_year'] = []

        p.iterator_replacements['calibration_parameters_source'] = []
        p.iterator_replacements['fine_resolution'] = []
        p.iterator_replacements['coarse_resolution'] = []
        p.iterator_replacements['regional_projections_input_path'] = []

        p.iterator_replacements['projected_coarse_change_dir'] = []
        p.iterator_replacements['cur_dir_parent_dir'] = []



        for index, row in p.scenarios_df.iterrows():
            seals_utils.assign_df_row_to_object_attributes(p, row)

            # HACK Seals_years, cannot use typical hack here cause has both
            # if hasattr(p, 'seals_years'):
            #     p.years = p.seals_years

            # if hasattr(p, 'seals_key_base_year'):
            #     p.key_base_year = p.seals_key_base_year[0]
            years_to_use = p.years
            if hasattr(p, 'seals_years'):
                years_to_use = p.seals_years

            # scenario_type = row['scenario_type']
            # p.regional_projections_input_path
            if p.scenario_type != 'baseline':
                # years = str(row['years']).split(' ')


                for c, year in enumerate(years_to_use):
                    if hasattr(p, 'regional_projections_input_path'):
                        
                        p.regional_projections_input_path = hb.replace_cat_ears_with_dict(p.regional_projections_input_path, {'year':year})

                        if p.regional_projections_input_path:
                            got_path = p.get_path(p.regional_projections_input_path)
                            task_dir = os.path.join(p.intermediate_dir, p.regional_projections_input_path)


                            # Tricky case here, because there was cat_ears in the refpath, it never found it and thus assumed it was an input to be created
                            # This means the path has the extra cur_dir derived paths. Hack here to find the refpath and merge it with intermediate
                            replace_dict = {'<^year^>': str(p.years[0])}
                            regional_change_classes_path1 = hb.replace_in_string_via_dict(got_path, replace_dict)
                                
                            if hb.path_exists(regional_change_classes_path1):
                                regional_change_classes_path = regional_change_classes_path1
                            else:
                                split = regional_change_classes_path1.split(os.path.split(p.cur_dir)[1])[1].replace('\\', '/')[1:]
                                regional_change_classes_path = os.path.join(p.intermediate_dir, split)                            
                            # split = regional_change_classes_path1.split(os.path.split(p.cur_dir)[1])[1].replace('\\', '/')[1:]
                            # regional_change_classes_path = os.path.join(p.intermediate_dir, split)                        

                            if hb.path_exists(regional_change_classes_path): # If it's a path, it is required to be based on the regional_change_dir task
                                projected_coarse_change_dir = os.path.join(p.regional_change_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year))
                            elif hb.path_exists(task_dir): # if it's a task that exists on the task tree, then that is the scenario root.
                                projected_coarse_change_dir = os.path.join(task_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year))
                            else:
                                raise NotImplementedError('No interpretation found for regional_projections_input_path of ' + p.regional_projections_input_path)

                        else:
                            projected_coarse_change_dir = os.path.join(p.intermediate_dir, 'coarse_change', 'coarse_simplified_ha_difference_from_previous_year', p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year))


                    else: # If it's blank, then assume there is no regional change projected and it is only run with a coarse_projected. Might need to rename this.
                        projected_coarse_change_dir = os.path.join(p.intermediate_dir, 'coarse_change', 'coarse_simplified_ha_difference_from_previous_year', p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year))

                    coarse_match_path = p.ha_per_cell_coarse_path
                    coarse_resolution = hb.get_cell_size_from_path(coarse_match_path)

                    # TODOOO, this got resent when called assign_df_row_to_object_attributes()
                    fine_resolution = hb.get_cell_size_from_path(p.base_year_lulc_path)

                    fine_resolution_arcseconds = hb.pyramid_compatible_resolution_to_arcseconds[fine_resolution]
                    coarse_resolution_arcseconds = hb.pyramid_compatible_resolution_to_arcseconds[coarse_resolution]


                    p.iterator_replacements['scenario_type'].append(p.scenario_type)
                    p.iterator_replacements['exogenous_label'].append(p.exogenous_label)
                    p.iterator_replacements['climate_label'].append(p.climate_label)
                    p.iterator_replacements['model_label'].append(p.model_label)
                    p.iterator_replacements['counterfactual_label'].append(p.counterfactual_label)
                    p.iterator_replacements['key_base_year'].append(p.key_base_year)
                    if hasattr(p, 'seals_key_base_year'):
                        p.iterator_replacements['seals_key_base_year'].append(p.seals_key_base_year[0])

                    if hasattr(p, 'seals_years'):
                        p.iterator_replacements['seals_year'].append(year)

                    p.iterator_replacements['year'].append(year)

                    if c == 0:

                        if hasattr(p, 'seals_key_base_year'):
                            p.iterator_replacements['previous_year'].append(p.seals_key_base_year[0])
                        else:
                            p.iterator_replacements['previous_year'].append(p.key_base_year)

                    else:
                        p.iterator_replacements['previous_year'].append(p.years[c-1])


                        if hasattr(p, 'seals_years'):
                            p.iterator_replacements['previous_year'].append(p.seals_years[c-1])
                        else:
                            p.iterator_replacements['previous_year'].append(p.years[c-1])


                    p.iterator_replacements['calibration_parameters_source'].append(p.calibration_parameters_source)

                    p.iterator_replacements['coarse_resolution'].append(coarse_resolution)
                    p.iterator_replacements['fine_resolution'].append(fine_resolution)
                    
                    # TODO BIG POINT, everytime you enter a new variable in the scenarios dir, you need to add a line here
                    # to make it update the iterator_projections. But, this is breaky and could be automated, e.g.
                    # for all things that aren't specified, they just get automatically added.
                    # if not hasattr(p, 'regional_projections_input_path'):
                    #     p.regional_projections_input_path = ''
                    p.iterator_replacements['regional_projections_input_path'].append(p.regional_projections_input_path)

                    p.iterator_replacements['projected_coarse_change_dir'].append(projected_coarse_change_dir)
                    p.iterator_replacements['cur_dir_parent_dir'].append(os.path.join(p.cur_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year)))



def allocation_zones(p):

    # # Define the zones overwhich parallel allocation should be run
    # Generate lists of which zones change and thus need to be rerun. Note however that this is SUPER RISKY because if you have a partial run, it fails.
    p.combined_block_lists_paths = {
        'fine_blocks_list': os.path.join(p.cur_dir, 'fine_blocks_list.csv'),
        'coarse_blocks_list': os.path.join(p.cur_dir, 'coarse_blocks_list.csv'),
        'processing_blocks_list': os.path.join(p.cur_dir, 'processing_blocks_list.csv'),
        'global_fine_blocks_list': os.path.join(p.cur_dir, 'global_fine_blocks_list.csv'),
        'global_coarse_blocks_list': os.path.join(p.cur_dir, 'global_coarse_blocks_list.csv'),
        'global_processing_blocks_list': os.path.join(p.cur_dir, 'global_processing_blocks_list.csv'),
    }
    
    if p.run_this:
    
        if hasattr(p, 'regional_projections_input_path'):

            # CANT IMPLEMENT THE STANDARD HACK CAUSE THIS USES BOTH DO IT AT RUNTIME
            # # Implement seals_years hack
            # if hasattr(p, 'seals_years'):
            #     p.years = p.seals_years

            # if hasattr(p, 'seals_key_base_year'):
            #     p.key_base_year = p.seals_key_base_year
            
            if p.regional_projections_input_path:
                has_cat_ears = hb.has_cat_ears(p.regional_projections_input_path)
                if has_cat_ears:
                    cat_ears_present = hb.parse_to_ce_list(p.regional_projections_input_path)
                    if 'year' in cat_ears_present:
                        replace_dict = {'<^year^>': str(p.years[0])}
                        regional_projections_input_path_pre1 = hb.replace_in_string_via_dict(p.regional_projections_input_path, replace_dict)                
                    p.regional_projections_input_path = hb.replace_cat_ears_with_object_attributes(p.regional_projections_input_path, p)

                # Check if it's a dir
                if os.path.isdir(p.regional_projections_input_path):
                    # Add the correct filename
                    fegional_projections_input_path = os.path.join(p.regional_projections_input_path, 'coarse_simplified_ha_difference_from_previous_year.tif')

                    # regional_change_classes_path = os.path.join(p.intermediate_dir, 'coarse_change', 'coarse_simplified_ha_difference_from_previous_year', p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year))

                regional_projections_input_path_pre1 = p.regional_projections_input_path
                got_path = p.get_path(regional_projections_input_path_pre1)
                task_dir = os.path.join(p.intermediate_dir, p.regional_projections_input_path)                            


                # Tricky case here, because there was cat_ears in the refpath, it never found it and thus assumed it was an input to be created
                # This means the path has the extra cur_dir derived paths. Hack here to find the refpath and merge it with intermediate
                replace_dict = {'<^year^>': str(p.years[0])}
                regional_change_classes_path_pre2 = hb.replace_in_string_via_dict(got_path, replace_dict)
                
                regional_change_classes_path3 = p.get_path(regional_change_classes_path_pre2)
                    
                if hb.path_exists(regional_change_classes_path3):
                    regional_change_classes_path = regional_change_classes_path3
                else:
                    split = regional_change_classes_path3.split(os.path.split(p.cur_dir)[1])[1].replace('\\', '/')[1:]
                    regional_change_classes_path = os.path.join(p.intermediate_dir, split)                            
                # split = regional_change_classes_path1.split(os.path.split(p.cur_dir)[1])[1].replace('\\', '/')[1:]
                # regional_change_classes_path = os.path.join(p.intermediate_dir, split)                        

                if hb.path_exists(regional_change_classes_path): # If it's a path, it is required to be based on the regional_change_dir task
                    projected_coarse_change_dir = os.path.join(p.regional_change_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(p.year))
                elif hb.path_exists(task_dir): # if it's a task that exists on the task tree, then that is the scenario root.
                    projected_coarse_change_dir = os.path.join(task_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(p.year))
                else:
                    raise NotImplementedError('No interpretation found for regional_projections_input_path of ' + p.regional_projections_input_path)
                
            else:
                projected_coarse_change_dir = os.path.join(p.intermediate_dir, 'coarse_change', 'coarse_simplified_ha_difference_from_previous_year', p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(p.year))

        
        else: # If it's blank, then assume there is no regional change projected and it is only run with a coarse_projected. Might need to rename this.                        
            projected_coarse_change_dir = os.path.join(p.intermediate_dir, 'coarse_change', 'coarse_simplified_ha_difference_from_previous_year', p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(p.year))
        a = p.counterfactual_label
        b = p.regional_projections_input_path
        c = p.projected_coarse_change_dir
        p.projected_coarse_change_dir = projected_coarse_change_dir
        hb.log(f'\n\nChecking if can use a precached block list for counterfactual_label:\n {p.combined_block_lists_paths}')
        try:
            if all(hb.path_exists(i) for i in p.combined_block_lists_paths.values()):
                blocks_lists_already_exist = True
                hb.log(f'    Found all required so using them. Note that this can make for strange behavior if you changed the AOI but then used precached blocklists.')
            else:
                blocks_lists_already_exist = False
                hb.log(f'    Did not find all required block lists so will recalculate them.')
        except:
            hb.log(f'    Error checking for existing block lists, so will recalculate them.')
            blocks_lists_already_exist = False


        disable_precached_block_lists = False # This is surprisingly hard to chagne for speedups because ie if you do disable, it will then niavely run on ocean blocks.
        
        if blocks_lists_already_exist and not disable_precached_block_lists:
            hb.log('Using precached block lists. This is much faster than recalculating them, BUT, if you change the AOI, processing resolution, or any of the block list parameters, you will need to delete the existing block lists files in order to recalculate them.')
            fine_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'fine_blocks_list.csv'), '2d_list'))
            coarse_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'coarse_blocks_list.csv'), '2d_list'))
            processing_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'processing_blocks_list.csv'), '2d_list'))
            global_fine_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'global_fine_blocks_list.csv'), '2d_list'))
            global_coarse_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'global_coarse_blocks_list.csv'), '2d_list'))
            global_processing_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'global_processing_blocks_list.csv'), '2d_list'))

        else:
            if p.aoi == 'global':

                fine_blocks_list = hb.get_global_block_list_from_resolution(p.processing_resolution, p.fine_resolution)
                coarse_blocks_list = hb.get_global_block_list_from_resolution(p.processing_resolution, p.coarse_resolution)
                processing_blocks_list = hb.get_global_block_list_from_resolution(p.processing_resolution, p.processing_resolution)
                global_fine_blocks_list = fine_blocks_list
                global_coarse_blocks_list = coarse_blocks_list
                global_processing_blocks_list = hb.get_global_block_list_from_resolution_and_bb(p.processing_resolution, p.processing_resolution, p.bb)
            else:
                fine_blocks_list = hb.pyramids.get_subglobal_block_list_from_resolution_and_bb(p.processing_resolution, p.fine_resolution, p.bb, verbose=0)
                coarse_blocks_list = hb.pyramids.get_subglobal_block_list_from_resolution_and_bb(p.processing_resolution, p.coarse_resolution, p.bb, verbose=0)
                processing_blocks_list = hb.pyramids.get_subglobal_block_list_from_resolution_and_bb(p.processing_resolution, p.processing_resolution, p.bb, verbose=0)
                global_fine_blocks_list = hb.get_global_block_list_from_resolution_and_bb(p.processing_resolution, p.fine_resolution, p.bb, verbose=0)
                global_coarse_blocks_list = hb.get_global_block_list_from_resolution_and_bb(p.processing_resolution, p.coarse_resolution, p.bb, verbose=0)
                global_processing_blocks_list = hb.get_global_block_list_from_resolution_and_bb(p.processing_resolution, p.processing_resolution, p.bb, verbose=0)

            hb.log('Length of iterator before pruning in task calibration:', len(fine_blocks_list))

            if p.subset_of_blocks_to_run is not None:
                old_fine_blocks_list = fine_blocks_list
                old_coarse_blocks_list = coarse_blocks_list
                old_processing_blocks_list = processing_blocks_list
                old_global_fine_blocks_list = global_fine_blocks_list
                old_global_coarse_blocks_list = global_coarse_blocks_list
                old_global_processing_blocks_list = global_processing_blocks_list

                fine_blocks_list = []
                coarse_blocks_list = []
                processing_blocks_list = []
                global_fine_blocks_list = []
                global_coarse_blocks_list = []
                global_processing_blocks_list = []
                for i in p.subset_of_blocks_to_run:
                    fine_blocks_list.append(old_fine_blocks_list[i])
                    coarse_blocks_list.append(old_coarse_blocks_list[i])
                    processing_blocks_list.append(old_processing_blocks_list[i])
                    global_fine_blocks_list.append(old_global_fine_blocks_list[i])
                    global_coarse_blocks_list.append(old_global_coarse_blocks_list[i])
                    global_processing_blocks_list.append(old_global_processing_blocks_list[i])

        hb.log('Length of iterator after considering subset_of_blocks_to_run:', len(fine_blocks_list))

        combined_block_lists_dict = {
            'fine_blocks_list': fine_blocks_list,
            'coarse_blocks_list': coarse_blocks_list,
            'processing_blocks_list': processing_blocks_list,
            'global_fine_blocks_list': global_fine_blocks_list,
            'global_coarse_blocks_list': global_coarse_blocks_list,
            'global_processing_blocks_list': global_processing_blocks_list,
        }

        if not all([hb.path_exists(i) for i in p.combined_block_lists_paths.values()]) or disable_precached_block_lists:

            # Pare down the number of blocks to run based on if there is change in the projected_coarse_change
            old_fine_blocks_list = fine_blocks_list
            old_coarse_blocks_list = coarse_blocks_list
            old_processing_blocks_list = processing_blocks_list
            old_global_fine_blocks_list = global_fine_blocks_list
            old_global_coarse_blocks_list = global_coarse_blocks_list
            old_global_processing_blocks_list = global_processing_blocks_list
            fine_blocks_list = []
            coarse_blocks_list = []
            processing_blocks_list = []
            global_fine_blocks_list = []
            global_coarse_blocks_list = []
            global_processing_blocks_list = []

            hb.log('Checking existing blocks for change in the LUH data and excluding if no change. TODO: Update this for other nonLUH approaches. Also this is really slow. Optimize it.')
            for c, block in enumerate(old_coarse_blocks_list):
                progress_percent = float(c) / float(len(old_coarse_blocks_list)) * 100.0
                
                # Log on a single line (don't create new lines) the percentage.
                if c % 100 == 0:
                    hb.log('    Checking block ' + str(c) + ' of ' + str(len(old_coarse_blocks_list)) + ' (' + str(round(progress_percent, 2)) + '%)', same_line_as_previous=True)
                skip = []

                current_coarse_change_rasters = []
                for class_label in p.changing_class_labels:


                    filename = class_label + '_' + str(p.year) + '_' + str(p.previous_year) + '_ha_diff_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '.tif'
                    
                    
                    # p.aggregation_method_string = 'covariate_multiply_regional_change_sum'
                    filename = hb.suri(filename, p.aggregation_method_string)
                    
                    gen_path = os.path.join(p.projected_coarse_change_dir, filename)
                    current_coarse_change_rasters.append(gen_path)


                for path in current_coarse_change_rasters:
                    # for path in hb.list_filtered_paths_nonrecursively(p.projected_coarse_change_dir, include_extensions='.tif'):
                    block = old_coarse_blocks_list[c]
                    a = hb.load_geotiff_chunk_by_cr_size(path, block)
                    changed = np.where((a != 0) & (a != -9999.) & (~np.isnan(a)), 1, 0)
                    # hb.show(a)
                    # 'C:\\Users\\jajohns\\Files\\Research\\cge\\seals\\projects\\test_seals_magpie\\intermediate\\magpie_as_simplified_proportion\\rcp45_ssp2\\2050\\SSP2_BiodivPol_ClimPol_NCPpol_LPJmL5\\urban_2050_2015_ha_difference.tif'

                    # hb.show(changed)
                    hb.debug('Checking to see if there is change in ', path)
                    if np.nansum(changed) == 0:
                        hb.debug('Skipping because no change in coarse projections:', path)
                        skip.append(True)
                    else:
                        skip.append(False)

                if not all(skip):
                    fine_blocks_list.append(old_fine_blocks_list[c])
                    coarse_blocks_list.append(old_coarse_blocks_list[c])
                    processing_blocks_list.append(old_processing_blocks_list[c])
                    global_fine_blocks_list.append(old_global_fine_blocks_list[c])
                    global_coarse_blocks_list.append(old_global_coarse_blocks_list[c])
                    global_processing_blocks_list.append(old_global_processing_blocks_list[c])
                    
                combined_block_lists_dict = {
                    'fine_blocks_list': fine_blocks_list,
                    'coarse_blocks_list': coarse_blocks_list,
                    'processing_blocks_list': processing_blocks_list,
                    'global_fine_blocks_list': global_fine_blocks_list,
                    'global_coarse_blocks_list': global_coarse_blocks_list,
                    'global_processing_blocks_list': global_processing_blocks_list,
                }
   
                    

            # WORKS but disabled for troubleshooting
            # Write the blockslists to csvs to avoid future reprocessing (actually is quite slow (2 mins) when 64000 tiles).
            for block_name, block_list in combined_block_lists_dict.items():
                hb.python_object_to_csv(block_list, os.path.join(p.cur_dir, block_name + '.csv'), '2d_list')

        else:
            hb.log('Using precached block lists from ' + str(p.combined_block_lists_paths))

        hb.log('Length of iterator after removing non-changing zones:', len(fine_blocks_list))

        # Remove from iterator lists that have already been computed.
        old_fine_blocks_list = fine_blocks_list
        old_coarse_blocks_list = coarse_blocks_list
        old_processing_blocks_list = processing_blocks_list
        old_global_fine_blocks_list = global_fine_blocks_list
        old_global_coarse_blocks_list = global_coarse_blocks_list
        old_global_processing_blocks_list = global_processing_blocks_list
        fine_blocks_list = []
        coarse_blocks_list = []
        processing_blocks_list = []
        global_fine_blocks_list = []
        global_coarse_blocks_list = []
        global_processing_blocks_list = []

        for c, fine_block in enumerate(old_global_processing_blocks_list):
            tile_dir = str(fine_block[0]) + '_' + str(fine_block[1])
            expected_path = os.path.join(p.cur_dir, tile_dir, 'allocation', 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(p.year) + '.tif')
            if not hb.path_exists(expected_path):
                fine_blocks_list.append(old_fine_blocks_list[c])
                coarse_blocks_list.append(old_coarse_blocks_list[c])
                processing_blocks_list.append(old_processing_blocks_list[c])
                global_fine_blocks_list.append(old_global_fine_blocks_list[c])
                global_coarse_blocks_list.append(old_global_coarse_blocks_list[c])
                global_processing_blocks_list.append(old_global_processing_blocks_list[c])

        hb.log('Length of iterator after removing finished zones:', len(fine_blocks_list))

        # Process for each block which calibration file it should use.
        nyi = True

        if not nyi:
            # NOTE an interesting quirk here. Although I want to make sure nothing runs globally when there is a target AOI set
            # , I do let this one run globally because its fast, and then the aoi-specific run just needs to use the right ID.
            p.aezreg_zones_raster_path = os.path.join(p.cur_dir, 'aezreg_zones.tif')
            p.processing_zones_raster_path = os.path.join(p.cur_dir, 'processing_zones.tif')
            p.processing_zones_to_calibration_chunk_path = os.path.join(p.cur_dir, 'processing_zones_to_calibration_chunk.csv')
            p.processing_zones_match_path = p.match_paths[3600.0]
            if p.run_this:
                if not hb.path_exists(p.aezreg_zones_raster_path):
                    hb.convert_polygons_to_id_raster(p.calibration_zone_polygons_path, p.aezreg_zones_raster_path, p.coarse_match_path, id_column_label='pyramid_id', data_type=5, ndv=-9999, all_touched=True, compress=True)
                if not hb.path_exists(p.processing_zones_raster_path):
                    hb.convert_polygons_to_id_raster(p.calibration_zone_polygons_path, p.processing_zones_raster_path, p.processing_zones_match_path, id_column_label='pyramid_id', data_type=5, ndv=-9999, all_touched=True, compress=True)

                if not hb.path_exists(p.processing_zones_to_calibration_chunk_path):
                    calibration_zones_to_calibration_chunk = {}

                    zones_raster = hb.as_array(p.processing_zones_raster_path)
                    uniques = np.unique(zones_raster)
                    r, c = hb.calculate_zone_to_chunk_list_lookup_dict(zones_raster)

                    zone_calibration_block_lookup_dict = {}
                    for u in uniques[uniques != -9999]:
                        n_in_zone = len(r[u][r[u] > 0])
                        selected_id = math.floor(n_in_zone / 2)
                        zone_calibration_block_lookup_dict[u] = (r[u, selected_id], c[u, selected_id])

                    with open(p.processing_zones_to_calibration_chunk_path, "w") as f:
                        for k, line in zone_calibration_block_lookup_dict.items():
                            f.write(str(k) + ',' + str(line[0]) + '_' + str(line[1]) + '\n')

        # Load the calibration variable used for all the zones if relying ona  precalcualted one.
        # TODO I never fixed the case where the data is just in the default base data and was resolved by get_path() the desired way. Could probably delete the code below cause the p.get_path found by the scenarios iteration should be sufficient.
        if hb.path_exists(p.calibration_parameters_source):
            calibration_parameters_path = p.calibration_parameters_source
        elif hb.path_exists(os.path.join(p.input_dir, p.calibration_parameters_source)):
            calibration_parameters_path = os.path.join(p.input_dir, p.calibration_parameters_source)
        elif hb.path_exists(os.path.join(p.base_data_dir, p.calibration_parameters_source)):
            calibration_parameters_path = os.path.join(p.base_data_dir, p.calibration_parameters_source)
        else:
            hb.log('Could not find calibration parameters file at ' + os.path.join(p.input_dir, p.calibration_parameters_source) + ' or ' + os.path.join(p.base_data_dir, p.calibration_parameters_source))

        if hb.path_exists(calibration_parameters_path):
            hb.log('Starting to read ' + calibration_parameters_path)
            df = pd.read_csv(calibration_parameters_path)

            # TODO This is bad. Fix it.
            # TODOOO, YES IT WAS A BAD IDEA YOU DUMMY.
            # TODOOO AGAIN. Indeed, still bad.
            year_replacement_dict = {}
            year_replacement_dict['2014'] = p.key_base_year
            for src, dst in year_replacement_dict.items():

                df['data_location'] = df.loc[:, 'data_location'].replace({str(src): str(dst)}, regex=True)

            # TODOOO Consider renaming this.
            p.combined_calibration_parameters_df = df

            hb.log('Finished reading ' + calibration_parameters_path)

        p.iterator_replacements = collections.OrderedDict()
        p.iterator_replacements['fine_blocks_list'] = fine_blocks_list
        p.iterator_replacements['coarse_blocks_list'] = coarse_blocks_list
        p.iterator_replacements['processing_blocks_list'] = processing_blocks_list
        p.iterator_replacements['global_fine_blocks_list'] = global_fine_blocks_list
        p.iterator_replacements['global_coarse_blocks_list'] = global_coarse_blocks_list
        p.iterator_replacements['global_processing_blocks_list'] = global_processing_blocks_list

        # Trickier replacement that will redefine the parent dir for each task so that it also WRITES in the correct output location
        p.iterator_replacements['cur_dir_parent_dir'] = [p.cur_dir_parent_dir+ '/allocation_zones/' + str(i[0]) + '_' + str(i[1]) for i in global_processing_blocks_list]


        if p.run_only_first_element_of_each_iterator:
            p.iterator_replacements['fine_blocks_list'] = [p.iterator_replacements['fine_blocks_list'][0]]
            p.iterator_replacements['coarse_blocks_list'] = [p.iterator_replacements['coarse_blocks_list'][0]]
            p.iterator_replacements['processing_blocks_list'] = [p.iterator_replacements['processing_blocks_list'][0]]
            p.iterator_replacements['global_fine_blocks_list'] = [p.iterator_replacements['global_fine_blocks_list'][0]]
            p.iterator_replacements['global_coarse_blocks_list'] = [p.iterator_replacements['global_coarse_blocks_list'][0]]
            p.iterator_replacements['global_processing_blocks_list'] = [p.iterator_replacements['global_processing_blocks_list'][0]]
            p.iterator_replacements['cur_dir_parent_dir'] = [p.iterator_replacements['cur_dir_parent_dir'][0]]


def allocation(passed_p=None):
    # Actually do the allocation

    if passed_p is None:
        global p
    else:
        p = passed_p

    start = time.time()
    if p.run_this:
        # Set where CHUNK-specific maps will be saved.
        p.initial_lulc_baseline_path = os.path.join(p.cur_dir, 'lulc_' + p.lulc_simplification_label + '_baseline_' + p.model_label + '_' + str(p.key_base_year) + '.tif')

        # p.lulc_baseline_path = os.path.join(p.cur_dir, 'lulc_baseline.tif')
        p.zone_esa_simplified_lulc_base_year_path = os.path.join(p.cur_dir, 'zone_esa_' + p.lulc_simplification_label + '_lulc_base_year.tif')
        p.fine_match_path = p.initial_lulc_baseline_path

        if isinstance(p.key_base_year, list):
            p.key_base_year = p.key_base_year[0]

        p.lulc_ndv = hb.get_ndv_from_path(p.aoi_lulc_simplified_paths[p.key_base_year])

        p.loss_function_sigma = np.float64(7.0) # Set how much closeness vs farness matters in assessing accuracy. Sigma = 1 means you need to be REALLY close to count as a food prediction.

        # Load the coefficients as DF from either calibration dir or a prebuilt dir.
        zone_string = os.path.split(p.cur_dir_parent_dir)[1]
        if p.calibration_parameters_source == 'calibration_task':
            current_pretrained_coefficients_path = os.path.join(p.calibration_dir, zone_string, 'calibration_zones', 'trained_coefficients_zone_' + zone_string + '.csv')
            hb.log('Setting current_pretrained_coefficients_path to one generated for zone ' + str(zone_string) + ' at  ' + current_pretrained_coefficients_path)
            spatial_regressors_df = pd.read_csv(current_pretrained_coefficients_path)

            if len(p.calibration_parameters_override_dict) > 0:
                try:
                    current_override_path = p.calibration_parameters_override_dict[p.current_scenario_pairing_label][p.current_year][p.current_policy_scenario_label]
                except:
                    raise NameError('Unable to find override for this scenario.')


                raise NameError('NYI for calibration_task')

            spatial_regressors_used_for_this_tile_file_root = 'trained_coefficients_'+p.current_scenario_pairing_label+'_'+str(p.current_year)+'_'+p.current_policy_scenario_label+'_'+p.current_policy_scenario_label
            spatial_regressors_used_for_this_tile_path = os.path.join(p.cur_dir, spatial_regressors_used_for_this_tile_file_root + '.csv')
            spatial_regressors_df.to_csv(spatial_regressors_used_for_this_tile_path)

        elif len(p.combined_calibration_parameters_df):
            if 'calibration_block_index' in p.combined_calibration_parameters_df.columns: # Then it is the global source we pull from
                current_calibration_block_index = zone_string + '_1_1'
                p.calibrated_parameters_df = p.combined_calibration_parameters_df[p.combined_calibration_parameters_df['calibration_block_index'] == current_calibration_block_index]
                spatial_regressors_df = p.calibrated_parameters_df
                # spatial_regressors_df = p.calibrated_parameters_df.drop(columns=['data_location'])
                # p.calibrated_parameters_df = p.calibrated_parameters_df.drop(columns=['data_location'])
                # p.calibrated_parameters_df['merge_col'] = p.calibrated_parameters_df['spatial_regressor_name']
            else:
                current_calibration_block_index = 'from_input'
                spatial_regressors_df = p.combined_calibration_parameters_df
                # p.calibrated_parameters_df = p.combined_calibration_parameters_df

            # Check if there is a suitable training tile. IF not, use default.
            # if p.calibrated_parameters_df['type'].isnull().values.any() or len(a) == 0: # NOTE: This is an optimized way to check if there's a nan in an array
            # # if np.isnan(np.sum(p.calibrated_parameters_df['type'].values)): # NOTE: This is an optimized way to check if there's a nan in an array
            #     p.calibrated_parameters_df = pd.read_csv(p.local_data_regressors_starting_values_path)
            #     p.calibrated_parameters_df['calibration_block_index'] = p.calibrated_parameters_df.shape[0] * [current_calibration_block_index]


            # #### OLD But possibly relevant after I redo the global calibration.
            # # Merge in local filenames
            # p.local_data_paths_df = pd.read_csv(p.local_data_regressors_starting_values_path)
            # p.local_data_paths_df = p.local_data_paths_df[['spatial_regressor_name', 'data_location']]
            # # p.local_data_paths_df['merge_col'] = p.local_data_paths_df['spatial_regressor_name']


            # spatial_regressors_df = hb.df_merge(p.calibrated_parameters_df, p.local_data_paths_df, left_on='spatial_regressor_name', right_on='spatial_regressor_name')
            # # spatial_regressors_df = pd.merge(p.calibrated_parameters_df, p.local_data_paths_df, on='merge_col')
            # if 'calibration_block_index' in spatial_regressors_df.columns:
            #     first_cols = ['calibration_block_index', 'spatial_regressor_name', 'type', 'data_location']
            # else:
            #     first_cols = ['spatial_regressor_name', 'type', 'data_location']

            # ordered_cols = first_cols + [i for i in spatial_regressors_df.columns if i not in first_cols and 'Unnamed' not in i]


            # spatial_regressors_df = spatial_regressors_df[ordered_cols]

            # # Replace year specific things with the current year
            # # TODOO Fix this hack by having the information in the spreadsheets be programatically generated per year via a function (like how it is done for the local_csv path but specific to this year).
            # spatial_regressors_df['data_location'] = spatial_regressors_df['data_location'].str.replace('2000', '2015')


            if len(p.calibration_parameters_override_dict) > 0:

                try:
                    current_override_path = p.calibration_parameters_override_dict[p.current_scenario_pairing_label][p.current_year][p.current_policy_scenario_label]
                    has_override = True
                except:
                    hb.debug('Unable to find override for this scenario.')
                    has_override = False


                if has_override:
                    left_df = spatial_regressors_df
                    # left_df = spatial_regressors_df[[i for i in spatial_regressors_df.columns if i not in ['class_' + j for j in p.class_labels]]]
                    right_df = pd.read_csv(current_override_path)
                    join_col = 'spatial_regressor_name'
                    rename_dict ={i: i + '_right' for i in right_df.columns if i != join_col}
                    right_df = right_df.rename(columns=rename_dict)
                    spatial_regressors_df = hb.df_merge(left_df, right_df, left_on=join_col, right_on=join_col, verbose=False, supress_warnings=True)

                    for column_label in spatial_regressors_df.columns:
                        if column_label[-6:] != '_right': # HACK
                            if column_label + '_right' in spatial_regressors_df.columns:

                                new_col = np.where(pd.isnull(spatial_regressors_df[column_label + '_right']), spatial_regressors_df[column_label], spatial_regressors_df[column_label + '_right'])
                                spatial_regressors_df[column_label] = new_col
                                # spatial_regressors_df[column_label].values = np.where(not np.isnan(spatial_regressors_df[column_label + '_right'].values), spatial_regressors_df[column_label + '_right'].values, spatial_regressors_df[column_label].values)
                    to_drop = []
                    for i in spatial_regressors_df.columns:
                        if i[-6:] == '_right':
                            to_drop.append(i)

                    spatial_regressors_df = spatial_regressors_df[[i for i in spatial_regressors_df.columns if i not in to_drop]]

            try:
                current_calibration_block_index
            except:
                current_calibration_block_index = 'here'

            spatial_regressors_used_for_this_tile_file_root = 'trained_coefficients_'+current_calibration_block_index
            spatial_regressors_used_for_this_tile_path = os.path.join(p.cur_dir, spatial_regressors_used_for_this_tile_file_root + '.csv')

            spatial_regressors_df.to_csv(spatial_regressors_used_for_this_tile_path, index=False)
        else:
            raise NameError('calibration_parameters_source doesnt make sense.')


        if p.use_calibration_created_coefficients:
            if p.use_calibration_from_zone_centroid_tile:
                # OLD NOTE: calibration_zone_polygons_path points to the correct GTAP37_AEZ18 gpkg. Rasterize it to create coarse-tile to AZREG correspondence then calculate centroids.
                p.calibration_zone_polygons_path
                zone_string = os.path.split(p.cur_dir_parent_dir)[1]
                current_pretrained_coefficients_path = os.path.join(p.calibration_dir, zone_string, 'calibration_zones', 'trained_coefficients_zone_' + zone_string + '.csv')
                hb.log('Setting current_pretrained_coefficients_path to one generated for zone ' + str(zone_string)  + ' at  ' + current_pretrained_coefficients_path)

            else:

                zone_string = os.path.split(p.cur_dir_parent_dir)[1]
                current_pretrained_coefficients_path = os.path.join(p.calibration_dir, zone_string, 'calibration_zones', 'trained_coefficients_zone_' + zone_string + '.csv')
                hb.log('Setting current_pretrained_coefficients_path to one generated in this project, at ' + current_pretrained_coefficients_path)
        else:
            current_pretrained_coefficients_path = p.calibration_parameters_source
                    # current_pretrained_coefficients_path = p.pretrained_coefficients_path_dict[p.current_scenario_pairing_label][p.current_year][p.current_policy_scenario_label]
            hb.log('Setting current_pretrained_coefficients_path to one specified in run configuration, at ' + current_pretrained_coefficients_path)



        changing_class_indices_array = np.asarray(p.changing_class_indices, dtype=np.int64)  # For Cythonization, load these as the "labels", which is used for writing.


        lulc_projected_path = os.path.join(p.cur_dir, 'lulc_' + p.lulc_src_label + '_'  + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(p.year) + '.tif')

        if p.skip_created_downscaling_zones:
            skip_this_zone = os.path.exists(lulc_projected_path)
        else:
            skip_this_zone = False

        if p.run_this and not skip_this_zone:

            # Tricky logic here: I implemented an optimization that skips doing zones that have no change. But this means
            # that the previous year lulc will not always be there. The logic below finds the most recent LULC map that
            # exists, reverting to the key_base_year if needed.
            previous_year_dir = p.cur_dir.replace('\\', '/').replace('/' + str(p.year) + '/', '/' + str(p.previous_year) + '/')
            if p.previous_year in p.lulc_simplified_paths: # Then it is the base year

                # On the first year, we need to clip from the AOI-wide LULC to the current processing tile.
                aoi_previous_year_path = p.lulc_simplified_paths[p.previous_year]

                # HACK IUCN. This can only be dealt with by fixing/eliminating the lulc_simplified_paths[] dicts
                if 'restoration' in p.scenario_label:
                    clipped_path = os.path.join(p.cur_dir, 'base_lulc.tif')
                    hb.clip_raster_by_bb(p.base_year_lulc_path, p.bb, clipped_path)
                    aoi_previous_year_path = clipped_path
                    # still need to clip tho



                tile_starting_lulc_path = os.path.join(p.cur_dir, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.model_label + '_' + str(p.key_base_year) + '.tif')



                lulc_baseline_array = hb.load_geotiff_chunk_by_cr_size(aoi_previous_year_path, p.fine_blocks_list, output_path=tile_starting_lulc_path).astype(np.int64)
                tile_match_path = tile_starting_lulc_path
            else:

                # On not-first years, we need to reference the previous year's and only write it if writing_level is sufficiently high.
                previous_year_filename = 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(p.previous_year) + '.tif'
                tile_starting_lulc_path = os.path.join(previous_year_dir, previous_year_filename)

                # If the previous year didnt change, then there wont be a file there. In that case, we search for the previous year that DID change.
                if not hb.path_exists(tile_starting_lulc_path):
                    found_it = False
                    reversed_years = p.years.copy()
                    reversed_years.reverse()

                    for test_year in reversed_years:
                        test_year_dir = p.cur_dir.replace('\\', '/').replace('/' + str(p.year) + '/', '/' + str(test_year) + '/')
                        test_year_filename = 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(test_year) + '.tif'
                        try_path = os.path.join(test_year_dir, test_year_filename)
                        if hb.path_exists(try_path):
                            tile_starting_lulc_path = try_path
                            found_it = True
                            break
                    # if you still cant find it, just use the starting year
                    if not found_it:
                        # Tricky: Note that it looks for it now in the first NON BASE YEAR, cause that's the only one i save it in.
                        test_year_dir = p.cur_dir.replace('\\', '/').replace('/' + str(p.year) + '/', '/' + str(p.years[0]) + '/')
                        tile_starting_lulc_path = os.path.join(test_year_dir, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.model_label + '_' + str(p.key_base_year) + '.tif')
                        if hb.path_exists(tile_starting_lulc_path):
                            found_it = True

                    if not found_it:

                        tile_starting_lulc_path = 6

                        aoi_previous_year_path = p.lulc_simplified_paths[p.key_base_year]
                        tile_starting_lulc_path = os.path.join(p.cur_dir, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.model_label + '_' + str(p.key_base_year) + '.tif')
                        lulc_baseline_array = hb.load_geotiff_chunk_by_cr_size(aoi_previous_year_path, p.fine_blocks_list, output_path=tile_starting_lulc_path).astype(np.int64)
                        tile_match_path = tile_starting_lulc_path

                lulc_baseline_array = hb.as_array(tile_starting_lulc_path).astype(np.int64)
                tile_match_path = tile_starting_lulc_path
                if p.output_writing_level > 1:
                    if not hb.path_exists(tile_starting_lulc_path):
                        hb.save_array_as_geotiff(lulc_baseline_array, tile_starting_lulc_path, tile_match_path, ndv=p.lulc_ndv, compress=True)




            spatial_layer_names = spatial_regressors_df['spatial_regressor_name'].dropna().values
            spatial_layer_paths = spatial_regressors_df['data_location'].dropna().values
            spatial_layer_types = spatial_regressors_df['type'].dropna().values



            # QUIRCK, adjacency is really just additive with preprocessing.
            spatial_layer_types_to_codes = {'multiplicative': 1,
                                            'additive': 2,
                                            }
            spatial_layer_types_to_codes.update({'gaussian_' + str(sigma): 2 for sigma in p.gaussian_sigmas_to_test})

            spatial_layer_function_types_1d = np.asarray([spatial_layer_types_to_codes[i] for i in spatial_layer_types], np.int64)

            # # CREATE GAUSSIANS for all variables tagged as that type. Note this is a large performance loss and I use precached global convolutions in all cases to date.
            # p.spatial_layer_chunk_paths = []
            # for c, path in enumerate(spatial_layer_paths):
            #     if spatial_regressors_df['type'].values[c][0:8] == 'gaussian':
            #         _, class_id, _, sigma = spatial_regressors_df['spatial_regressor_name'].values[c].split('_')
            #         filename = os.path.split(path)[1]
            #         spatial_chunk_path = os.path.join(p.cur_dir, os.path.split(spatial_regressors_df['data_location'].values[c])[1])
            #         if not os.path.exists(spatial_chunk_path):
            #             hb.load_geotiff_chunk_by_cr_size(path, p.fine_blocks_list, output_path=spatial_chunk_path)
            #         p.spatial_layer_chunk_paths.append(spatial_chunk_path)
            #
            # spatial_layer_chunk_counter = 0
            # for c, class_label in enumerate(spatial_regressors_df['spatial_regressor_name'].values):
            #     if spatial_regressors_df['type'].values[c][0:8] == 'gaussian':
            #         _, class_id, _, sigma = class_label.split('_')
            #
            #         kernel_path = os.path.join(p.generated_kernels_dir, 'gaussian_' + str(sigma) + '.tif')
            #         output_path = os.path.join(p.cur_dir, class_label + '_convolution.tif')
            #
            #         # NOTE, fft_gaussian has to write to disk
            #         if not os.path.exists(output_path):
            #             seals_utils.fft_gaussian(p.spatial_layer_chunk_paths[spatial_layer_chunk_counter], kernel_path, output_path, -9999.0, True)
            #
            #         spatial_layer_chunk_counter += 1

            n_c, n_r = int(p.fine_blocks_list[2]), int(p.fine_blocks_list[3])
            coarse_n_c, coarse_n_r = int(p.coarse_blocks_list[2]), int(p.coarse_blocks_list[3])

            # Load things that dont ever change over generations or final run
            block_ha_per_cell_fine_path = os.path.join(p.cur_dir, 'block_ha_per_cell_fine.tif')
            block_ha_per_cell_coarse_path = os.path.join(p.cur_dir, 'block_ha_per_cell_coarse.tif')

            # This has to be written to a file so that it can define the aoi of the coarse grid. I could optimize this as
            # it creates n-zones number of bloat files. Don't need fine file because that georeference we can
            # get from the baseline lulc.
            hectares_per_grid_coarse_cell = hb.load_geotiff_chunk_by_cr_size(p.aoi_ha_per_cell_coarse_path, p.coarse_blocks_list, output_path=block_ha_per_cell_coarse_path).astype(np.float64)

            if p.output_writing_level > 0:
                hectares_per_grid_cell = hb.load_geotiff_chunk_by_cr_size(p.aoi_ha_per_cell_fine_path, p.fine_blocks_list, output_path=block_ha_per_cell_fine_path).astype(np.float64)
            else:
                hectares_per_grid_cell = hb.load_geotiff_chunk_by_cr_size(p.aoi_ha_per_cell_fine_path, p.fine_blocks_list).astype(np.float64)

            # IMPORTANT NOTE CAVEAT: Although the final allocation does properly start from the 2015 lulc, the underlying ranking is still based on the 2000 class binaries.
            # Figure out a way to make the spatial_layers_3d smartly update the "new state" variables in a way that is forward looking for year-by-year iteration

            # Build the numpy array for spatial layers.
            spatial_layers_3d = np.zeros((len(spatial_layer_paths), n_r, n_c)).astype(np.float64)

            # Chose not to normalize anything.
            normalize_inputs = False
            # Add either the normalized or not normalized array to the spatial_layers_3d
            for c, path in enumerate(spatial_layer_paths):
                hb.debug('Loading spatial layer at path ' + path)

                # path = hb.get_first_extant_path(path, [p.fine_processed_inputs_dir, p.input_dir, p.base_data_dir])
                # if 'binary_esa_seals7_2015_urban' in path:
                #     pass

                # if 'soil_organic_content' in path:
                #     pass

                # PROBLEM Sometimes it NEEDS to look in fine_processed_inputs_dir, but other times it needs to download it no matter what. how deal with this?
                # Am I possibly using get_path to deal with THREE types of data
                # 1. Data that needs to be created
                # 2. Data that needs t be put in base_data_dir
                # I confused....
                possible_dirs = [p.intermediate_dir, p.fine_processed_inputs_dir, p.input_dir, p.base_data_dir]
                
                
                path = p.get_path(path, possible_dirs=possible_dirs)
                current_bb = hb.get_bounding_box(path)

                if current_bb == hb.global_bounding_box:
                    correct_fine_block_list = p.global_fine_blocks_list
                    correct_coarse_block_list = p.global_coarse_blocks_list
                else:
                    correct_fine_block_list = p.fine_blocks_list
                    correct_coarse_block_list = p.coarse_blocks_list

                add_randomness = 0 # DECIDED NOT TO DO THIS. Don't try it again. I'm warning you! Increment the following comment up 1 for each unsuccessful attempt at doing this: 3
                if spatial_layer_types[c] == 'additive' or spatial_layer_types[c] == 'multiplicative':
                    if normalize_inputs is True:
                        spatial_layers_3d[c] = hb.normalize_array(hb.load_geotiff_chunk_by_cr_size(path, correct_fine_block_list))
                    else:

                        if add_randomness:
                            a = hb.load_geotiff_chunk_by_cr_size(path, correct_fine_block_list)
                            a = a * (1 - (np.random.random(a.shape)/100))
                            spatial_layers_3d[c] = a
                        else:
                            spatial_layers_3d[c] = hb.load_geotiff_chunk_by_cr_size(path, correct_fine_block_list)

                # NOTE: Currently gaussian is not used as it is just considered additive
                elif spatial_layer_types[c][0:8] == 'gaussian':
                    # updated_path = os.path.join(p.cur_dir, 'class_' + spatial_layer_names[c].split('_')[1] + '_gaussian_' + spatial_layer_names[c].split('_')[3] + '_convolution.tif')

                    if normalize_inputs is True:
                        spatial_layers_3d[c] = hb.normalize_array(hb.load_geotiff_chunk_by_cr_size(path, correct_fine_block_list))
                        # spatial_layers_3d[c] = hb.normalize_array(hb.as_array(updated_path))
                    else:
                        if add_randomness:
                            a = hb.load_geotiff_chunk_by_cr_size(path, correct_fine_block_list)
                            a = a * (1 - (np.random.random(a.shape)/100))
                            spatial_layers_3d[c] = a
                        else:
                            spatial_layers_3d[c] = hb.load_geotiff_chunk_by_cr_size(path, correct_fine_block_list)

                else:
                    raise NameError('unspecified type')

            if p.base_data_dir in p.lulc_simplified_paths[p.key_base_year]:
                observed_lulc_array = hb.load_geotiff_chunk_by_cr_size(p.lulc_simplified_paths[p.key_base_year], p.global_fine_blocks_list, output_path=None).astype(np.int64)
            else:
                observed_lulc_array = hb.load_geotiff_chunk_by_cr_size(p.lulc_simplified_paths[p.key_base_year], p.fine_blocks_list, output_path=None).astype(np.int64)


            valid_mask_array = np.where((observed_lulc_array != p.lulc_ndv), 1, 0).astype(np.int64)


            # Set how much change for each class needs to be allocated.
            projected_coarse_change_3d = np.zeros((len(changing_class_indices_array), coarse_n_r, coarse_n_c)).astype(np.float64)
            # projected_coarse_change_dir = os.path.join(p.coarse_simplified_ha_difference_from_previous_year_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(p.year))

            # HACK IUCN
            # TODOOO: Add this functionality in the scenarios csv? Maybe have it be a string to function?
            # if 'restoration' in p.counterfactual_label:
            #     projected_coarse_change_dir = os.path.join(p.coarse_simplified_projected_ha_difference_from_previous_year_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(p.year))


            filename_end = '_' + str(p.year) + '_' + str(p.previous_year) + '_ha_diff_'  + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '.tif'
            # filename_end = '_' + str(p.year) + '_' + str(p.previous_year) + '_ha_diff_'  + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + p.aggregation_method_string + '.tif'
            projected_coarse_change_paths = [os.path.join(p.projected_coarse_change_dir, i + filename_end) for i in p.changing_class_labels]


            for c, path in enumerate(projected_coarse_change_paths):
            # for c, path in enumerate(list(p.projected_coarse_change_files[p.current_scenario_pairing_label][p.current_year][p.current_policy_scenario_label].values())):

                # Scaling is unnecessary if you use stricly pyramidal zones... but i'm not sure i want to lose this yet e.g. for intersecting zones and country boundaries.
                scale_coarse_results = 0
                if scale_coarse_results:
                    unscaled = hb.as_array(path)
                    p.proportion_valid_fine_per_coarse_cell = hb.calc_proportion_of_coarse_res_with_valid_fine_res(unscaled, valid_mask_array).astype(np.float64)
                    scaled_proportion_to_allocate = p.proportion_valid_fine_per_coarse_cell * unscaled
                    scaled_proportion_to_allocate_path = os.path.join(p.cur_dir, os.path.split(path)[1])
                    hb.save_array_as_geotiff(scaled_proportion_to_allocate, scaled_proportion_to_allocate_path, p.fine_match_path, data_type=6)
                    projected_coarse_change_3d[c] = scaled_proportion_to_allocate.astype(np.float64)
                else:
                    if p.write_projected_coarse_change_chunks:
                        current_output_path = os.path.join(p.cur_dir, os.path.split(path)[1])
                        # altered_path =
                        projected_coarse_change_3d[c] = hb.load_geotiff_chunk_by_cr_size(path, p.coarse_blocks_list, output_path=current_output_path).astype(np.float64)
                    else:
                        projected_coarse_change_3d[c] = hb.load_geotiff_chunk_by_cr_size(path, p.coarse_blocks_list).astype(np.float64)

            # Note questionable choice here that the actual calibration parameters must be the last n-classes of columns
            p.seals_class_names = spatial_regressors_df.columns.values[-len(changing_class_indices_array):]
            spatial_regressor_trained_coefficients = spatial_regressors_df[p.seals_class_names].values.astype(np.float64).T
            generation_best_parameters = np.copy(spatial_regressor_trained_coefficients)

            p.call_string = ''

            # L.setLevel(logging.DEBUG)
            lulc_baseline_array = lulc_baseline_array.astype(np.int64)
            hb.debug('projected_coarse_change_3d', type(projected_coarse_change_3d), projected_coarse_change_3d.dtype)
            hb.debug('lulc_baseline_array', type(lulc_baseline_array), lulc_baseline_array.dtype, lulc_baseline_array)
            hb.debug('spatial_layers_3d', type(spatial_layers_3d), spatial_layers_3d.dtype, spatial_layers_3d)
            hb.debug('generation_best_parameters', type(generation_best_parameters), generation_best_parameters.dtype, generation_best_parameters)
            hb.debug('spatial_layer_function_types_1d', type(spatial_layer_function_types_1d), spatial_layer_function_types_1d.dtype, spatial_layer_function_types_1d)
            hb.debug('valid_mask_array', type(valid_mask_array), valid_mask_array.dtype, valid_mask_array)
            hb.debug('p.changing_class_indices_array', type(changing_class_indices_array), changing_class_indices_array.dtype, changing_class_indices_array)
            hb.debug('observed_lulc_array', type(observed_lulc_array), observed_lulc_array.dtype, observed_lulc_array)
            hb.debug('hectares_per_grid_cell', type(hectares_per_grid_cell), hectares_per_grid_cell.dtype, hectares_per_grid_cell)
            hb.debug('p.cur_dir', type(p.cur_dir), p.cur_dir)
            hb.debug('p.loss_function_sigma', type(p.loss_function_sigma), p.loss_function_sigma)
            hb.debug('p.call_string', type(p.call_string), p.call_string)

            previous_cumulative_change_happened_path = os.path.join(previous_year_dir, 'cumulative_change_happened.tif')
            if hb.path_exists(previous_cumulative_change_happened_path):
                previous_cumulative_change_happened_array = hb.as_array(previous_cumulative_change_happened_path)
                valid_mask_array = np.where(previous_cumulative_change_happened_array == 1, 0, valid_mask_array)
            else:
                previous_cumulative_change_happened_array = np.zeros_like(valid_mask_array)
            # previous_change_year_path = os.path.join(previous_year_dir, 'change_year.tif')
            # if hb.path_exists(previous_change_year_path):
            #     previous_change_year_array = hb.as_array(previous_change_year_path)

            allow_contracting = np.int64(p.allow_contracting)
            # Strange choice, but the allocation function both calibrates AND RUNS the final projection using the calibration. If instead you want to
            # Run on precalibrated parameters, you have to reload coarse_change_3d.
            overall_similarity_score, lulc_projected_array, overall_similarity_plot, class_similarity_scores, class_similarity_plots, output_change_arrays, change_happened = \
                calibrate(projected_coarse_change_3d,
                        lulc_baseline_array,
                        spatial_layers_3d,
                        generation_best_parameters,
                        spatial_layer_function_types_1d,
                        valid_mask_array,
                        changing_class_indices_array,
                        p.changing_class_labels,
                        observed_lulc_array,
                        hectares_per_grid_cell,
                        p.cur_dir,
                        p.cython_reporting_level,
                        allow_contracting,
                        p.loss_function_sigma,
                        output_match_path=tile_match_path,
                        call_string=p.call_string)

            # LEARNING POINT: GDAL silently fails to write if you have a file path too long. This happened below. Fix by enabling long-filepaths in OS.

            # Write generated arrays to disk
            generated_gt = hb.generate_geotransform_of_chunk_from_cr_size_and_larger_path(p.fine_blocks_list, p.base_year_lulc_path)
            generated_projection = hb.common_projection_wkts['wgs84']
            lulc_projected_array = lulc_projected_array.astype(np.int8)
            hb.save_array_as_geotiff(lulc_projected_array, lulc_projected_path, tile_match_path, projection_override=generated_projection, ndv=255, data_type=1, compress=True, verbose=False)

            # lulc_baseline_path = os.path.join(p.cur_dir, 'lulc_' + p.lulc_simplification_label + '_baseline_' + p.model_label + '_' + str(p.year) + '.tif')
            # # lulc_baseline_path = os.path.join(p.cur_dir, 'lulc_' + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(p.year) + '.tif')
            # if p.output_writing_level >= 1:
            #     if not hb.path_exists(lulc_baseline_path):
            #         hb.save_array_as_geotiff(lulc_baseline_array, lulc_baseline_path, tile_match_path, projection_override=generated_projection, ndv=255, data_type=1, compress=True, verbose=False)

            if p.output_writing_level >= 1:
                change_year_path = os.path.join(p.cur_dir, 'change_year.tif')
                if not hb.path_exists(change_year_path):
                    change_year_array = np.zeros(lulc_projected_array.shape, dtype=np.int64)
                    for c, class_id in enumerate(p.changing_class_indices):
                        change_year_array += np.where(output_change_arrays[c] == 1, p.year, 0)
                    hb.save_array_as_geotiff(change_year_array, change_year_path, tile_match_path, projection_override=generated_projection, ndv=-9999, data_type=5, compress=True, verbose=False)

            if p.output_writing_level >= 1:
                change_happened_path = os.path.join(p.cur_dir, 'change_happened.tif')
                if not hb.path_exists(change_happened_path):
                    hb.save_array_as_geotiff(change_happened, change_happened_path, tile_match_path, projection_override=generated_projection, ndv=-9999, data_type=5, compress=True, verbose=False)

            if p.output_writing_level >= 5:
                for i, label in enumerate(p.changing_class_labels):
                    hb.save_array_as_geotiff(output_change_arrays[i], os.path.join(p.cur_dir, 'allocations_for_class_' + p.changing_class_labels[i] + '.tif'), tile_match_path)

                    # naive_upscale = hb.upscale_array(output_change_arrays[i], upscale_factor=resolution, upscale_method='mode')
                    # hb.show(output_change_arrays[i], output_path=hb.ruri(os.path.join(output_dir, 'allocations_for_class_' + str(i) + '.png')), vmin=0, vmax=1, title='allocations for class ' + str(i))
            # if p.output_writing_level >= 4:
            #     hb.save_array_as_geotiff(projected_lulc, hb.suri(os.path.join(output_dir, 'projected_lulc.tif'), call_string), output_match_path)


            # This is the one other required written file because it is used in the next iteration.
            cumulative_change_happened_path = os.path.join(p.cur_dir, 'cumulative_change_happened.tif')
            if not hb.path_exists(cumulative_change_happened_path):
                updated_cumulative_change_happened = np.where(change_happened == 1, 1, previous_cumulative_change_happened_array)
                hb.save_array_as_geotiff(updated_cumulative_change_happened, cumulative_change_happened_path, tile_match_path, projection_override=generated_projection, ndv=-9999, data_type=5, compress=True, verbose=False)

                if p.output_writing_level >= 5:
                    validation_dir = os.path.join(p.cur_dir, 'validation')
                    hb.create_directories(validation_dir)
                    from hazelbean.calculation_core import \
                        aspect_ratio_array_functions

                    # import upscale_retaining_sum
                    for c, class_label in enumerate(p.changing_class_labels):
                        projected_recoarsening_path = os.path.join(validation_dir, class_label + '_allocated_prop.tif')
                        hb.create_directories(projected_recoarsening_path)
                        if not hb.path_exists(projected_recoarsening_path):
                            target_value = p.changing_class_indices[c]

                            upscale_factor = int(p.coarse_resolution / p.fine_resolution)

                            was_class = np.where(lulc_baseline_array == target_value, 1, 0).astype(np.float64)
                            is_class = np.where(lulc_projected_array == target_value, 1, 0).astype(np.float64)

                            was_class_coarse = aspect_ratio_array_functions.upscale_retaining_sum(was_class, upscale_factor)
                            is_class_coarse = aspect_ratio_array_functions.upscale_retaining_sum(is_class, upscale_factor)
                            hectares_per_grid_cell_upscaled = aspect_ratio_array_functions.upscale_using_mean(hectares_per_grid_cell, upscale_factor)
                            net = is_class_coarse - was_class_coarse
                            net_ha = net * hectares_per_grid_cell_upscaled


                            hb.save_array_as_geotiff(net_ha, projected_recoarsening_path, block_ha_per_cell_coarse_path, projection_override=generated_projection, ndv=-9999, data_type=5, compress=True, verbose=False)



                # seals_utils.calc_observed_lulc_change_for_two_lulc_paths(previous_year_path, lulc_projected_path, block_ha_per_cell_coarse_path, p.changing_class_indices, validation_dir)




def stitched_lulc_simplified_scenarios(p):
    # Stitch the simplified LULC tiles back together
    """Stitch together the lulc_projected.tif files in each of the zones (or in the case of magpie, also in zones_adjusted).
    Also write on top of a global base map if selected so that areas not downscaled (like oceans) have the correct LULC from
    the base map. (E.g., we don't downscale the Falkland islands becuase SSPs don't have any change there. We don't want to delete
    the Falklands either, though.)"""

    if p.run_this:


        vrt_paths_to_remove = []
        for index, row in p.scenarios_df.iterrows():
            seals_utils.assign_df_row_to_object_attributes(p, row)
            hb.log('Stitching for scenario ' + str(index) + ' of ' + str(len(p.scenarios_df)))

            current_force_to_global_bb = 0
            
            if p.scenario_type != 'baseline':
                for year in p.years:

                    include_string = p.lulc_simplification_label + '_'
                    include_string = 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(year) + '.tif'
                    target_dir = os.path.join(p.allocations_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year))


                    p.layers_to_stitch = hb.list_filtered_paths_recursively(target_dir, include_strings=include_string, include_extensions='.tif', depth=None)


                    stitched_output_name = 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(year)


                    p.lulc_projected_stitched_path = os.path.join(p.cur_dir, stitched_output_name + '.tif')

                    if not hb.path_exists(p.lulc_projected_stitched_path):
                        if len(p.layers_to_stitch) > 0:
                            hb.log('Stitching for year ' + str(year))
                            hb.log('Stitching ' + str(len(p.layers_to_stitch)) + ' layers, first 1 of which was: ' + str(p.layers_to_stitch[:1]))
                            vrt_path = os.path.join(p.cur_dir, stitched_output_name + '.vrt')

                            # First, make a stitched VRT to get the bb of the generated tiles. Is fastish because write_vrt_to_tif=False
                            hb.create_gdal_vrt(
                                p.layers_to_stitch,
                                vrt_path,
                                srcnodata=None,
                                dstnodata=255,
                            )
                            vrt_paths_to_remove.append(vrt_path)
                            p.bb_of_tiles = hb.get_bounding_box(vrt_path)

                            hb.log('Calculated bb of tiles to be ' + str(p.bb_of_tiles))
                            global_bb = hb.get_bounding_box(p.base_year_lulc_path)

                            if p.bb_of_tiles != global_bb and not p.force_to_global_bb:
                                current_force_to_global_bb = 0
                            else:
                                current_force_to_global_bb = 1
                            p.L.warning('When stamping lulc, found that the set of tiles was not global, current_force_to_global_bb = ' + str(current_force_to_global_bb))

                            hb.log('Stamping generated lulcs with extent_shift_match_path of base_year_lulc_path ' + str(p.base_year_lulc_path))
                            ndv = hb.get_datatype_from_uri(p.base_year_lulc_path)

                            # The only difference between these is if it forces the written raster to have a global extent
                            # applied. Default vrt processing will fill those with NDV for now.


                            if not current_force_to_global_bb:
                                # We stil need something to stamp onto so that non-allocated locations still have values,
                                # but this needs to be clipped to size.
                                # p.local_output_base_map_path = os.path.join(p.cur_dir, 'lulc_' + p.baseline_reference_label + '_' + str(p.key_base_year) + '.tif')

                                baseline_model_label = p.scenarios_df.loc[p.scenarios_df['scenario_label'] == p.baseline_reference_label, 'model_label'].values[0]
                                baseline_filename = 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + str(p.key_base_year) + '.tif'
                                stub_path = os.path.join('lulc', p.lulc_src_label, p.lulc_simplification_label, baseline_filename)
                                # p.local_output_base_map_path = os.path.join(p.cur_dir, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + baseline_model_label + '_' + str(p.key_base_year) + '.tif')
                                local_output_base_map_path = p.get_path(stub_path, prepend_possible_dirs=p.fine_processed_inputs_dir)



                                lu_classes_ha_csv_path = os.path.join(p.cur_dir, p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + str(p.key_base_year) + '_lu_classes_ha.csv')
                                # lulc_dir = os.path.join(p.fine_processed_inputs_dir, 'lulc', p.lulc_src_label, p.lulc_simplification_label)
                                lulc_stub_path = os.path.join('lulc', p.lulc_src_label, p.lulc_simplification_label)
                                lulc_filename = 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + str(p.key_base_year) + '.tif'
                                lulc_path = p.get_path('lulc', p.lulc_src_label, p.lulc_simplification_label, lulc_filename, prepend_possible_dirs=p.fine_processed_inputs_dir)


                                p.layers_to_stitch.insert(0, local_output_base_map_path)
                                hb.stitch_rasters_using_vrt(
                                    p.layers_to_stitch,
                                    p.lulc_projected_stitched_path,
                                    srcnodata=None,
                                    dstnodata=255,
                                )

                                # Clean up by removing the vrt file
                                vrt_path = hb.replace_ext(p.lulc_projected_stitched_path, '.vrt')
                                hb.remove_path(vrt_path)


                            else:
                                # NOTE By inserting this in the front of the list, it makes sure it is BEHIND the newly generated tiles. It also ensures the BB is fully global.
                                global_lulc_simplified_base_year_path = p.get_path('lulc', p.lulc_src_label, p.lulc_simplification_label, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + str(p.key_base_year) + '.tif')
                                p.layers_to_stitch.insert(0, global_lulc_simplified_base_year_path)
                                L.info('Added to stitch ' + p.lulc_simplified_paths[p.key_base_year])
                                # hb.add_class_counts_file_to_raster(p.base_year_lulc_path)
                                # hb.create_gdal_virtual_raster_using_file(p.layers_to_stitch, p.lulc_projected_stitched_path, write_vrt_to_tif=True, bands='all',
                                #              vrt_extent_shift_match_path=p.base_year_lulc_path, extent_shift_match_path=p.base_year_lulc_path,
                                #              remove_generator_files=True,
                                #              srcnodata=None, dstnodata=255, compress=True, output_pixel_size=None, s_srs=None, t_srs=None, resampling_method='near',
                                #              output_datatype='Byte')

                                hb.stitch_rasters_using_vrt(
                                    p.layers_to_stitch,
                                    p.lulc_projected_stitched_path,
                                    srcnodata=None,
                                    dstnodata=255,
                                )

                                # baseline_model_label = p.scenarios_df.loc[p.scenarios_df['scenario_label'] == p.baseline_reference_label, 'model_label'].values[0]
                                # p.local_output_base_map_path = os.path.join(p.cur_dir, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + baseline_model_label + '_' + str(p.key_base_year) + '.tif')
                                # if not hb.path_exists(p.local_output_base_map_path):
                                #     # for base_year in p.base_year:
                                #     hb.clip_raster_by_bb(p.lulc_simplified_paths[p.key_base_year], p.bb_of_tiles, p.local_output_base_map_path)
                    else:
                        hb.log('Skipping stitching ' + p.lulc_projected_stitched_path + ' because it already exists.')
                    
                    
                    # POSSIBLE STARTING POINT: I have no idea why, but the areas in the NORTH outside of the aereg but inside the bb have change, but the areas IN the aezreg don't have change.
                    if p.clip_to_aoi and p.aoi != 'global' and hb.path_exists(p.aoi_path):
                        hb.timer('start clip')
                        clipped_path = hb.suri(p.lulc_projected_stitched_path, 'clipped')
                        if not hb.path_exists(clipped_path):
                            hb.clip_raster_by_vector(p.lulc_projected_stitched_path, clipped_path, p.aoi_path)

                            # Generally you don't want to clip the file because it will mess up the bounding box.
                            # hb.displace_file(clipped_path, p.lulc_projected_stitched_path, displaced_path=None, delete_original=True)

                            hb.timer('end clip')

        for file_path in vrt_paths_to_remove:
            hb.remove_path(file_path)
        
        # Even with 256gb memory, running 30 parallel pogers at once hits a ZSTD-related compression memory error.
        if current_force_to_global_bb:
            hb.make_paths_pogs_in_parallel(p.cur_dir, exclude_strings='2025', max_workers=16, dont_actually_do_it=False, verbose=True, verbose_pog_check=True)

def luh_seals_baseline_adjustment(p):
    # SHORTCUT, this should have been put in GTAP project but i'm racing.
    if p.run_this:

        # Set the coarse_match
        coarse_match_path = p.aoi_ha_per_cell_coarse_path

        ## --------------- Clip the ids by the BB. ----------------
        # Note sorta unintuitively named load_geotiff_chunk_by_bb actually is just used
        # to save the clipped file, which happens becuase we set an output_path.
        ee_r50_aez18_ids_path = p.get_path("gtap_invest", "region_boundaries", "ee_r50_aez18_ids.tif")
        bb_ee_r50_aez18_ids_path = os.path.join(p.cur_dir, 'bb_ee_r50_aez18_ids_10sec.tif')
        if not hb.path_exists(bb_ee_r50_aez18_ids_path):
            hb.load_geotiff_chunk_by_bb(ee_r50_aez18_ids_path, p.bb, output_path=bb_ee_r50_aez18_ids_path)

        # DEVELOPMENT: Still need to document this.
        # On the outside of each performance-enhancing path_exists call, there are to-be-generated files above but also conditional loads
        # below that utilize string vs gpkg differentiation.
        # vector_boundaries_path = p.get_path("gtap_invest", "region_boundaries", "ee_r50_aez18_correspondence.gpkg")
        # vector_boundaries = vector_boundaries_path

        ## --------------- Create coarse zone ids ----------------
        bb_ee_r50_aez18_ids_15min_path = os.path.join(p.cur_dir, 'bb_ee_r50_aez18_ids_15min.tif')
        if not hb.path_exists(bb_ee_r50_aez18_ids_15min_path):

                hb.resample_to_match_pyramid(bb_ee_r50_aez18_ids_path,
                        coarse_match_path,
                        bb_ee_r50_aez18_ids_15min_path,
                        resample_method='near',
                        output_data_type=5,
                        src_ndv=-9999.,
                        ndv=-9999.,
                        s_srs_wkt=None,
                        compress=True,
                        ensure_fits=False,
                        gtiff_creation_options=hb.globals.DEFAULT_GTIFF_CREATION_OPTIONS,
                        calc_raster_stats=False,
                        add_overviews=False,
                        pixel_size_override=None,
                        remove_intermediate_files=False,
                        verbose=False)


        ## --------------- Create lu_classes_num_cells ----------------
        ### NOTE: I did this two different, unexpected ways. I first did the by-class cell count on the binary files. But then
        # I remembered I could have just used the stats_to_retreive = 'enumeration' option in the zonal statistics function
        # which I then used via the multiply_raster_path funciton to get the class_ha
        combined_class_num_cells_path = os.path.join(p.cur_dir, p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + str(p.key_base_year) + '_lu_classes_num_cells.csv')
        paths_to_merge = []

        if not 0 and hb.path_exists(combined_class_num_cells_path):

            for class_c, class_label in enumerate(p.all_class_labels):

                # Calc zonal statistics of given the coarse aez_reg ids from the fine LULC data
                target_dir = os.path.join(p.fine_processed_inputs_dir, 'lulc', p.lulc_src_label, p.lulc_simplification_label, 'binaries', str(p.key_base_year))
                target_filename = 'binary_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + str(p.key_base_year) + '_' + class_label + '.tif'
                lulc_binary_path = os.path.join(target_dir, target_filename)
                # "C:\Users\jajohns\Files\seals\projects\test_iucn_30by30\intermediate\fine_processed_inputs\lulc\esa\seals7\binaries\2017\binary_esa_seals7_2017_other.tif"
                current_num_cells_path = os.path.join(p.cur_dir, p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + str(p.key_base_year) + '_' + class_label + '_num_cells.csv')
                paths_to_merge.append(current_num_cells_path)
                if not hb.path_exists(current_num_cells_path):
                    hb.zonal_statistics(
                                lulc_binary_path,
                                zones_vector_path=None, #zones_vector_path=vector_boundaries_path,
                                id_column_label='ee_r50_aez18_id',
                                zone_ids_raster_path=bb_ee_r50_aez18_ids_path,
                                stats_to_retrieve='sums',
                                enumeration_classes=None,
                                enumeration_labels=None,
                                multiply_raster_path=p.aoi_ha_per_cell_coarse_path,
                                output_column_prefix=None,
                                vector_columns_to_keep='all',
                                csv_output_path=current_num_cells_path,
                                vector_output_path=None,
                                zones_ndv = -9999,
                                zones_raster_data_type=5,
                                unique_zone_ids=None, # CAUTION on changing this one. Cython code is optimized by assuming a continuous set of integers of the right bit size that covers all value possibilities and zero and the NDV.
                                id_min = 0,
                                id_max = 7000,
                                assert_projections_same=False,
                                values_ndv=-9999,
                                max_enumerate_value=20000,
                                use_pygeoprocessing_version=False,
                                verbose=False,
                    )

            # Merge all of the class specifi n_cells csvs into one
            hb.df_merge_list_of_csv_paths(paths_to_merge,
                                        output_csv_path=combined_class_num_cells_path,
                                        on='id',
                                        column_suffix='ignore',
                                        verbose=False)

            # Now rename and rewrite binary_esa_seals7_2017_c
            df = pd.read_csv(combined_class_num_cells_path)
            columns = {i: i.replace('_sums', '').replace('binary_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + str(p.key_base_year) + '_', '') for i in df.columns}
            df.rename(columns=columns, inplace=True)
            df = df[['id'] + p.all_class_labels]

            # Fill nans with 0
            df.fillna(0, inplace=True)

            df.to_csv(combined_class_num_cells_path, index=False)

            # Remove temp files:
            for path in paths_to_merge:
                hb.path_remove(path)


        ## --------------- Create lu_classes_ha ----------------
        lu_classes_ha_csv_path = os.path.join(p.cur_dir, p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + str(p.key_base_year) + '_lu_classes_ha.csv')
        # lulc_dir = os.path.join(p.fine_processed_inputs_dir, 'lulc', p.lulc_src_label, p.lulc_simplification_label)
        lulc_stub_path = os.path.join('lulc', p.lulc_src_label, p.lulc_simplification_label)
        lulc_filename = 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + str(p.key_base_year) + '.tif'
        lulc_path = p.get_path('lulc', p.lulc_src_label, p.lulc_simplification_label, lulc_filename, prepend_possible_dirs=p.fine_processed_inputs_dir)

        if not hb.path_exists(lu_classes_ha_csv_path):
            hb.zonal_statistics(
                        lulc_path,
                        zones_vector_path=None,
                        id_column_label='ee_r50_aez18_id',
                        zone_ids_raster_path=bb_ee_r50_aez18_ids_path,
                        stats_to_retrieve='enumeration',
                        enumeration_classes=p.all_class_indices,
                        enumeration_labels=p.all_class_labels,
                        multiply_raster_path=p.aoi_ha_per_cell_fine_path,
                        output_column_prefix='',
                        vector_columns_to_keep='all',
                        csv_output_path=lu_classes_ha_csv_path,
                        vector_output_path=None,
                        zones_ndv = -9999,
                        zones_raster_data_type=5,
                        unique_zone_ids=None, # CAUTION on changing this one. Cython code is optimized by assuming a continuous set of integers of the right bit size that covers all value possibilities and zero and the NDV.
                        id_min = 0,
                        id_max = 7000,
                        assert_projections_same=False,
                        values_ndv=-9999,
                        max_enumerate_value=20000,
                        use_pygeoprocessing_version=False,
                        verbose=False,
            )

            # Reorder, clean and resave
            df = pd.read_csv(lu_classes_ha_csv_path)
            if 'id' not in df.columns:
                df['id'] = df['id_created']
            df = df[['id'] + p.all_class_labels]
            df.fillna(0, inplace=True)
            df.to_csv(lu_classes_ha_csv_path, index=False)


        ## --------------- Create zonal sum of LUH pprojection ----------------
        coarse_class_ha_path = os.path.join(p.cur_dir, p.coarse_src_label + '_' + p.coarse_simplification_label + '_' + str(p.key_base_year) + '_luh_classes_ha.csv')
        if not hb.path_exists(coarse_class_ha_path):
            paths_to_merge = []
            for class_c, class_label in enumerate(p.coarse_correspondence_class_labels):
                current_dir = os.path.join(p.coarse_simplified_ha_dir, 'baseline', p.model_label, str(p.key_base_year))
                current_filename = class_label + '_ha_baseline_' + p.model_label + '_' + str(p.key_base_year) + '.tif'
                current_luh_path = os.path.join(current_dir, current_filename)
                current_luh_csv_path = os.path.join(p.cur_dir, class_label + '_luh_ha_' + p.exogenous_label + '_' + p.model_label + '_' + str(p.key_base_year) + '.csv')

                paths_to_merge.append(current_luh_csv_path)
                hb.zonal_statistics(
                            current_luh_path,
                            zones_vector_path=None, #zones_vector_path=vector_boundaries_path,
                            id_column_label='ee_r50_aez18_id',
                            zone_ids_raster_path=bb_ee_r50_aez18_ids_15min_path,
                            stats_to_retrieve='sums',
                            enumeration_classes=None,
                            enumeration_labels=None,
                            multiply_raster_path=p.aoi_ha_per_cell_coarse_path,
                            output_column_prefix=None,
                            vector_columns_to_keep='all',
                            csv_output_path=current_luh_csv_path,
                            vector_output_path=None,
                            zones_ndv = -9999,
                            zones_raster_data_type=5,
                            unique_zone_ids=None, # CAUTION on changing this one. Cython code is optimized by assuming a continuous set of integers of the right bit size that covers all value possibilities and zero and the NDV.
                            id_min = 0,
                            id_max = 7000,
                            assert_projections_same=False,
                            values_ndv=-9999,
                            max_enumerate_value=20000,
                            use_pygeoprocessing_version=False,
                            verbose=False,
                        )

                # Drop the 0 column
                df = pd.read_csv(current_luh_csv_path)
                df = df.drop('0', axis=1)
                df.to_csv(current_luh_csv_path, index=False)


            # Merge all of the class specifi n_cells csvs into one
            hb.df_merge_list_of_csv_paths(paths_to_merge,
                                        output_csv_path=coarse_class_ha_path,
                                        on='id',
                                        column_suffix='ignore',
                                        verbose=False,)

            # Now rename and rewrite binary_esa_seals7_2017_[class]
            df = pd.read_csv(coarse_class_ha_path)
            columns = {i: i.replace('_ha_baseline_' + p.model_label + '_' + str(p.key_base_year) + '_sums', '') for i in df.columns}
            df.rename(columns=columns, inplace=True)
            df = df[['id'] + p.coarse_correspondence_class_labels]

            # Fill nans with 0
            df.fillna(0, inplace=True)

            df.to_csv(coarse_class_ha_path, index=False)

            # Remove temp files:
            for path in paths_to_merge:
                hb.path_remove(path)

        ## --------------- Create difference between Coarse and Fine projections ----------------
        coarse_minus_fine_ha_path = os.path.join(p.cur_dir, 'coarse_minus_fine_ha.csv')
        if not hb.path_exists(coarse_minus_fine_ha_path):
            coarse_class_ha = pd.read_csv(coarse_class_ha_path)               
            lu_classes_ha = pd.read_csv(lu_classes_ha_csv_path)                
            
            output_df = hb.df_merge(coarse_class_ha, lu_classes_ha, on='id', how='outer', verbose=False, supress_warnings=True)
            output_df.fillna(0, inplace=True)


            for class_c, class_label in enumerate(p.changing_class_labels):

                output_df[class_label] = output_df[class_label + '_x'] - output_df[class_label + '_y']

            output_df = output_df[['id'] + p.changing_class_labels]

            # Write to file
            output_df.to_csv(coarse_minus_fine_ha_path, index=False)



        # ## ----------------- Create a shifter  of the difference between coarse and fine by zone, which will eventually be used to make the coarse have the same totals as the fine
        # baseline_correction_shifter_paths = [os.path.join(p.cur_dir, 'baseline_correction_shifter_' + i + '.tif') for i in p.changing_class_labels]
        # if not hb.path_all_exist(baseline_correction_shifter_paths):
        #     df = pd.read_csv(coarse_minus_fine_ha_path)
        #     id_values = df['id'].values

        #     combined_class_num_cells = pd.read_csv(combined_class_num_cells_path)

        #     for class_c, class_label in enumerate(p.changing_class_labels):
        #         n_cells = combined_class_num_cells[class_label].values
        #         values = df[class_label].values

        #         hb.log('Shifting for ', class_label)
        #         hb.log('values', values)
        #         hb.log('n_cells', n_cells)

        #         shift_list = np.where((values != 0) & (n_cells != 0), values / n_cells, 0)
        #         hb.log('shift_list', shift_list)
        #         # shift_list = [values[i] / n_cells[i] for i in range(len(id_values))]
        #         reclassify_rules = {id_values[i]: shift_list[i] for i in range(len(id_values))}
        #         hb.reclassify_raster_hb(bb_ee_r50_aez18_ids_15min_path, reclassify_rules, baseline_correction_shifter_paths[class_c], 6, -9999)

        # ## --------------- Shift the coarse raster by the shifter above ----------------
        # shifted_coarse_paths = [os.path.join(p.cur_dir, p.coarse_src_label + '_corrected_' + i + '.tif') for i in p.changing_class_labels]
        # if not hb.path_all_exist(shifted_coarse_paths):
        #     def do_the_shift(a, b):
        #         return a - b


        #     for class_c, class_label in enumerate(p.changing_class_labels):
        #         hb.log('Shifting class ' + class_label + ' by the shifter')
        #         current_input_list = [os.path.join(p.coarse_simplified_ha_dir, 'baseline', p.model_label, str(p.key_base_year), class_label + '_ha_baseline_' + p.model_label + '_' + str(p.key_base_year) + '.tif'), baseline_correction_shifter_paths[class_c]]
        #         hb.raster_calculator_flex(current_input_list, do_the_shift, shifted_coarse_paths[class_c], datatype=5, ndv=-9999., gtiff_creation_options=None, add_overviews=False)




        # ## --------------- Diagnose any weird things about the totals between fine and coarse ---------------

        # coarse_fine_diagnostic_report_path = os.path.join(p.cur_dir, 'coarse_fine_diagnostic_report.csv')
        # if not 0 and hb.path_exists(coarse_fine_diagnostic_report_path):
        #     hb.log('Creating diagnostic report for fine and coarse totals for scenarios ' + str(p.scenario_label) + ' ' + str(p.years))
        #     coarse_class_ha = pd.read_csv(coarse_class_ha_path)
        #     lu_classes_ha = pd.read_csv(lu_classes_ha_csv_path)
        #     coarse_minus_fine_ha = pd.read_csv(coarse_minus_fine_ha_path)


        #     overall_difference = 0
        #     total_fine_ha = 0
        #     for class_c, class_label in enumerate(p.changing_class_labels):


        #         coarse_sum = coarse_class_ha[class_label].sum()
        #         fine_sum = lu_classes_ha[class_label].sum()
        #         implied_diff = coarse_sum - fine_sum
        #         diff_sum = coarse_minus_fine_ha[class_label].sum()

        #         overall_difference += implied_diff
        #         total_fine_ha += fine_sum

        #         if abs(implied_diff - diff_sum) > .001:
        #             hb.log('Class: ' + class_label + ' had different sums!!!' + ' Coarse sum: ' + str(coarse_sum) + ' Fine sum: ' + str(fine_sum) + ' Implied diff: ' + str(implied_diff) + ' Diff sum: ' + str(diff_sum))

        #         percent_difference = (coarse_sum - fine_sum) / fine_sum * 100

        #         hb.log('Class: ' + class_label + ' Coarse sum: ' + str(coarse_sum) + ' Fine sum: ' + str(fine_sum) + ' Diff sum: ' + str(diff_sum)+ ' Percent diff: ' + str(percent_difference))

        #     hb.log('Overall difference: ' + str(overall_difference) + ' Total fine ha: ' + str(total_fine_ha) + ' Percent diff: ' + str(overall_difference / total_fine_ha * 100))

def iucn_30by30_scenarios(p):
    pass

def restoration(p):
    pass

def protection_by_aezreg_to_meet_30by30(p):
    # Creates a CSV by AEZReg of the protection needed to meet 30by30

    # Key output here will be used to generate CSVs of the protection needed by zone.
    p.protection_by_aezreg_to_meet_30by30_path = os.path.join(p.cur_dir, 'protection_by_aezreg_to_meet_30by30.csv')

    if p.run_this:

        # Make coarse res (15 min manuually here) zone ids, This should be put into base data
        ee_r50_aez18_ids_path = p.get_path('gtap_invest', 'region_boundaries', 'ee_r50_aez18_ids.tif')
        ee_r50_aez18_ids_15min_path = os.path.join(p.cur_dir, 'ee_r50_aez18_ids_15min.tif')
        # ee_r50_aez18_ids_15min_path = p.get_path('ee_r50_aez18_ids_15min.tif')

        # resample to 15min
        if not hb.path_exists(ee_r50_aez18_ids_15min_path):
            hb.resample_to_match_pyramid(ee_r50_aez18_ids_path,
                    p.aoi_ha_per_cell_coarse_path,
                    ee_r50_aez18_ids_15min_path,
                    resample_method='near',
                    output_data_type=5,
                    src_ndv=-9999.,
                    ndv=-9999.,
                    s_srs_wkt=None,
                    compress=True,
                    ensure_fits=False,
                    gtiff_creation_options=hb.globals.DEFAULT_GTIFF_CREATION_OPTIONS,
                    calc_raster_stats=False,
                    add_overviews=False,
                    pixel_size_override=None,
                    remove_intermediate_files=False,
                    verbose=False)


        # Key input here takes from ECN data

        # existing hectares protected
        protected_areas_all_baseline_300sec_path = p.get_path('gtap_invest', 'projects', 'economic_case_for_nature', 'protected_areas_all_baseline_hectares_300sec.tif')

        # Resample existing hectares to coarse res
        protected_areas_all_baseline_15min_path = os.path.join(p.cur_dir, 'protected_areas_all_baseline_15min.tif')
        if not hb.path_exists(protected_areas_all_baseline_15min_path):
            hb.resample_to_match_pyramid(protected_areas_all_baseline_300sec_path,
                    ee_r50_aez18_ids_15min_path,
                    protected_areas_all_baseline_15min_path,
                    resample_method='bilinear',
                    output_data_type=6,
                    )


        # new hectares to protect
        ha_to_protect_300sec_path = p.get_path('gtap_invest', 'projects', 'economic_case_for_nature', 'ha_to_protect.tif')

        # Resample new hectares to coarse res
        ha_to_protect_15min_path = os.path.join(p.cur_dir, 'ha_to_protect_15min.tif')
        if not hb.path_exists(ha_to_protect_15min_path):
            hb.resample_to_match_pyramid(ha_to_protect_300sec_path,
                    ee_r50_aez18_ids_15min_path,
                    ha_to_protect_15min_path,
                    resample_method='bilinear',
                    output_data_type=6,
                    )

        # Calculate zonal sums of existing protected ha
        protected_areas_all_baseline_sums_path = os.path.join(p.cur_dir, 'protected_areas_all_baseline_sums.csv')
        if not hb.path_exists(protected_areas_all_baseline_sums_path):
            hb.zonal_statistics(
                protected_areas_all_baseline_15min_path,
                zone_ids_raster_path=ee_r50_aez18_ids_15min_path,
                stats_to_retrieve='sums',
                csv_output_path=protected_areas_all_baseline_sums_path,
            )

            # HACK to drop the extra
            df = pd.read_csv(protected_areas_all_baseline_sums_path)
            df = df[['id'] + [i for i in df.columns if 'sums' in str(i) or 'counts' in str(i) or 'enumerat' in str(i)]]
            df.to_csv(protected_areas_all_baseline_sums_path)


        # Calculate zonal sums of new ha to protect
        ha_to_protect_sums_path = os.path.join(p.cur_dir, 'ha_to_protect_sums.csv')
        if not hb.path_exists(ha_to_protect_sums_path):
            hb.zonal_statistics(
                ha_to_protect_15min_path,
                zone_ids_raster_path=ee_r50_aez18_ids_15min_path,
                stats_to_retrieve='sums',
                csv_output_path=ha_to_protect_sums_path,
            )


            # HACK to drop the extra
            df = pd.read_csv(ha_to_protect_sums_path)
            df = df[['id'] + [i for i in df.columns if 'sums' in str(i) or 'counts' in str(i) or 'enumerat' in str(i)]]
            df.to_csv(ha_to_protect_sums_path)


        # Merge Zonal sums  and export as csv
        if not hb.path_exists(p.protection_by_aezreg_to_meet_30by30_path):
            df1 = pd.read_csv(protected_areas_all_baseline_sums_path)
            df2 = pd.read_csv(ha_to_protect_sums_path)
            combined_df = hb.df_merge(df1, df2, on='id', how='outer', supress_warnings=True, verbose=False)
            combined_df.fillna(0, inplace=True)

            combined_df['sum'] = combined_df['ha_to_protect_15min_sums'] + combined_df['protected_areas_all_baseline_15min_sums']

            rename_dict = {}
            rename_dict['ha_to_protect_15min_sums'] = 'ha_to_protect'
            rename_dict['protected_areas_all_baseline_15min_sums'] = 'ha_already_protected'
            rename_dict['0_x'] = 'id_pre'
            combined_df.rename(columns=rename_dict, inplace=True)
            if 'id_pre' in combined_df.columns:
                combined_df = combined_df[['id_pre', 'ha_to_protect', 'ha_already_protected', 'sum']]
                combined_df.rename(columns={'id_pre': 'id'}, inplace=True)
            else:
                combined_df = combined_df[['id', 'ha_to_protect', 'ha_already_protected', 'sum']]


            combined_df.to_csv(p.protection_by_aezreg_to_meet_30by30_path, index=False)




def ag_value(p):

    # Resamples ag value from 5min to 10sec.

    # Key outpu: High res ag value
    p.ag_value_10sec_path = p.get_path('crops', 'earthstat', 'ag_value_10sec.tif')
    if p.run_this:
        ag_value = p.get_path("crops", "ag_value_2000.tif")
        # resample to 10sec

        if not hb.path_exists(p.ag_value_10sec_path):
            hb.resample_to_match_pyramid(ag_value,
                    p.aoi_ha_per_cell_fine_path,
                    p.ag_value_10sec_path,
                    resample_method='bilinear',
                    output_data_type=6,
                    )


def coarse_simplified_projected_ha_difference_from_previous_year(p):
    # Create a coarse resolution change map from existing coarse map and shift vector

    ee_r50_aez18_ids_15min_path = os.path.join(p.protection_by_aezreg_to_meet_30by30_dir, 'ee_r50_aez18_ids_15min.tif')

    # NOTE: SEALS Requires that if there is a regional_change, it is in a dir (linked to a task) named regional_change_dir
    p.regional_change_dir = p.cur_dir

    if p.run_this:

        # Calculate number of cells in each AEZ-Reg
        n_cells_per_zone_path = os.path.join(p.cur_dir, 'n_cells_per_zone.csv')
        if not hb.path_exists(n_cells_per_zone_path):
            reclassify_rules = hb.enumerate_raster_path(ee_r50_aez18_ids_15min_path)
            final_dict = {'id': 'n_cells_per_zone'}
            final_dict.update(reclassify_rules)
            hb.python_object_to_csv(final_dict, n_cells_per_zone_path)



        # Iterate through all the scenarios
        for index, row in p.scenarios_df.iterrows():
            seals_utils.assign_df_row_to_object_attributes(p, row)

            if p.scenario_type != 'baseline' and p.regional_projections_input_path:

                # Read protection_by_aezreg_to_meet_30by30_path (this was generated based on ECN protected areas)
                protection_by_aezreg_to_meet_30by30 = pd.read_csv(p.protection_by_aezreg_to_meet_30by30_path)

                # Read fine_lu_classes_ha_csv_path from luh_seals_baseline_adjustment_dir
                fine_lu_classes_ha_csv_path = os.path.join(p.luh_seals_baseline_adjustment_dir, p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + str(p.key_base_year) + '_lu_classes_ha.csv')
                fine_lu_classes_ha = pd.read_csv(fine_lu_classes_ha_csv_path)
                n_cells_df = pd.read_csv(n_cells_per_zone_path)

                # Merge the above
                merged = pd.merge(fine_lu_classes_ha, protection_by_aezreg_to_meet_30by30, on='id', how='outer')
                merged2 = pd.merge(merged, n_cells_df, on='id', how='outer')
                merged2.fillna(0, inplace=True)

                for year_c, year in enumerate(p.years):

                    # IUCN SPECIFIC STEP
                    # Calculate what the natural vegetation would have been based on the proportion of total aezreg area in that type of natural land
                    merged2['grassland_prop'] =  merged2['grassland']  /  (merged2['grassland'] + merged2['forest'] + merged2['othernat'])
                    merged2['forest_prop'] =  merged2['forest']  /  (merged2['grassland'] + merged2['forest'] + merged2['othernat'])
                    merged2['othernat_prop'] =  merged2['othernat']  /  (merged2['grassland'] + merged2['forest'] + merged2['othernat'])


                    for class_c, class_label in enumerate(p.changing_class_labels):

                        # Get the correct previous year
                        if year_c == 0:
                            previous_year = p.key_base_year

                            previous_coarse_ha_dir = os.path.join(p.coarse_simplified_ha_dir, 'baseline', p.model_label, str(previous_year))
                            previous_coarse_ha_path = os.path.join(previous_coarse_ha_dir, class_label + '_ha_baseline_' + p.model_label + '_' + str(p.key_base_year) + '.tif')

                            previous_coarse_ha_diff_dir = os.path.join(p.coarse_simplified_ha_difference_from_previous_year_dir, 'baseline', p.model_label, str(previous_year))
                            previous_coarse_ha_diff_path = os.path.join(previous_coarse_ha_diff_dir, class_label + '_ha_diff_' + p.model_label + '_' + str(p.key_base_year) + '.tif')
                        else:
                            previous_year = p.years[year_c - 1]
                            previous_coarse_ha_dir = os.path.join(p.cur_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(previous_year))
                            previous_coarse_ha_filename = class_label + '_ha_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(previous_year) + '.tif'
                            previous_coarse_ha_path = os.path.join(previous_coarse_ha_dir, previous_coarse_ha_filename)

                            previous_coarse_ha_diff_dir = os.path.join(p.cur_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(previous_year))
                            previous_coarse_ha_diff_filename = class_label + '_ha_diff_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(previous_year) + '.tif'
                            previous_coarse_ha_diff_path = os.path.join(previous_coarse_ha_dir, previous_coarse_ha_filename)

                        current_coarse_ha_shift_filename = class_label + '_' + str(year) + '_' + str(previous_year) + '_ha_diff_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '.tif'
                        current_coarse_ha_diff_dir = os.path.join(p.cur_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year))
                        current_coarse_ha_shift_path = os.path.join(current_coarse_ha_diff_dir, current_coarse_ha_shift_filename)

                        # If the land is in a natural class (hacked by having the word prop), then shift it by the amount of new natural land required
                        # but multiplies by the potential natural vegetation proportion (IUCN SPEIFIC)
                        if class_label + '_prop' in merged2.columns:
                            if not hb.path_exists(current_coarse_ha_shift_path):

                                # Shift the input coarse projection via a reclassification
                                zones = merged2['id'].values.astype(np.int32)
                                values = merged2['ha_to_protect'] * merged2[class_label + '_prop'].astype(np.float32)
                                n_cells = merged2['n_cells_per_zone'].values.astype(np.float32)

                                # Build the reclassification rule
                                # NOTE: The Values here is the total amoutn needed for the AEZ, so we allocate evenly by dividing among the number of
                                # cells within the AEZ. This could have been further targetted but we did not.
                                reclassify_rules = {zones[i]: values[i] / n_cells[i] for i in range(len(zones))}

                                # TODO Build TEST that ensures this doesn't have wonky pregen

                                reclassify_rules_final = {}
                                for i in range(-9999, 9999):
                                    if i not in reclassify_rules:
                                        reclassify_rules_final[i] = np.float32(0)
                                    else:
                                        reclassify_rules_final[i] = reclassify_rules[i].astype(np.float32)

                                # LEARNING POINT, due to the f'ed logic in the classification algirithms, must have a lowest value in the dict.
                                # reclassify_rules[np.int32(-9999)] = np.float32(0)
                                hb.reclassify_raster_hb(ee_r50_aez18_ids_15min_path, reclassify_rules_final, current_coarse_ha_shift_path, 6, output_ndv=-9999)

                        # If the land is not natural, then make a new coarse projection raster that is all zeros
                        else:
                            if not hb.path_exists(current_coarse_ha_shift_path):
                                zones = merged2['id'].values.astype(np.int32)
                                reclassify_rules = {zones[i]: np.float32(0) for i in range(len(zones))}

                                reclassify_rules_final = {}
                                for i in range(-9999, 9999):
                                    if i not in reclassify_rules:
                                        reclassify_rules_final[i] = np.float32(0)
                                    else:
                                        reclassify_rules_final[i] = reclassify_rules[i]
                                hb.reclassify_raster_hb(ee_r50_aez18_ids_15min_path, reclassify_rules_final, current_coarse_ha_shift_path, 6, output_ndv=-9999)




                        current_coarse_ha_diff_filename = class_label + '_' + str(year) + '_' + str(previous_year) + '_ha_diff_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '.tif'
                        current_coarse_ha_diff_dir = os.path.join(p.cur_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year))
                        current_coarse_ha_diff_path = os.path.join(current_coarse_ha_diff_dir, current_coarse_ha_diff_filename)

                        shifted_path = os.path.join(p.cur_dir, 'shifted_' + class_label + '.tif')
                        if not hb.path_exists(current_coarse_ha_diff_path):
                            def do_the_shift(a, b):
                                    return a + b
                            current_input_list = [previous_coarse_ha_path, current_coarse_ha_diff_path]
                            hb.raster_calculator_flex(current_input_list, do_the_shift, current_coarse_ha_diff_path, datatype=6, ndv=-9999., gtiff_creation_options=None, add_overviews=False)

                # If both regional and coarse inputs are defined, combine them
                if p.scenario_type != 'baseline' and hb.path_exists(p.get_path(p.coarse_projections_input_path)):
                    for year_c, year in enumerate(p.years):

                        if year_c == 0:
                            previous_year = p.key_base_year

                            # previous_coarse_ha_dir = os.path.join(p.coarse_simplified_ha_dir, 'baseline', p.model_label, str(previous_year))
                            # previous_coarse_ha_path = os.path.join(previous_coarse_ha_dir, class_label + '_ha_baseline_' + p.model_label + '_' + str(p.key_base_year) + '.tif')

                            # previous_coarse_ha_diff_dir = os.path.join(p.coarse_simplified_ha_difference_from_previous_year_dir, 'baseline', p.model_label, str(previous_year))
                            # previous_coarse_ha_diff_path = os.path.join(previous_coarse_ha_diff_dir, class_label + '_ha_diff_' + p.model_label + '_' + str(p.key_base_year) + '.tif')
                        else:
                            previous_year = p.years[year_c - 1]
                            # previous_coarse_ha_dir = os.path.join(p.cur_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(previous_year))
                            # previous_coarse_ha_filename = class_label + '_ha_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(previous_year) + '.tif'
                            # previous_coarse_ha_path = os.path.join(previous_coarse_ha_dir, previous_coarse_ha_filename)

                            # previous_coarse_ha_diff_dir = os.path.join(p.cur_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(previous_year))
                            # previous_coarse_ha_diff_filename = class_label + '_ha_diff_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(previous_year) + '.tif'
                            # previous_coarse_ha_diff_path = os.path.join(previous_coarse_ha_dir, previous_coarse_ha_filename)


                        for class_c, class_label in enumerate(p.changing_class_labels):
                            current_coarse_ha_diff_filename = class_label + '_' + str(year) + '_' + str(previous_year) + '_ha_diff_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '.tif'
                            current_coarse_ha_diff_dir = os.path.join(p.cur_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year))
                            current_coarse_ha_diff_path = os.path.join(current_coarse_ha_diff_dir, current_coarse_ha_diff_filename)
                            current_coarse_input_path = os.path.join(p.coarse_simplified_ha_difference_from_previous_year_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year), current_coarse_ha_diff_filename)
                            # first rename the shifted
                            displaced_path = hb.rsuri(current_coarse_ha_diff_path, 'pre_coarse')
                            hb.path_rename(current_coarse_ha_diff_path, displaced_path)

                            # add them
                            def do_the_shift(a, b):
                                    return a + b
                            current_input_list = [current_coarse_input_path, displaced_path]
                            hb.raster_calculator_flex(current_input_list, do_the_shift, current_coarse_ha_diff_path, datatype=6, ndv=-9999., gtiff_creation_options=None, add_overviews=False)


def stitched_lulc_esa_scenarios(p):
    # optional
    # Reclassify the stitched simplified to original classification.

    def fill_where_not_changed(changed, baseline, esa):
        return np.where(changed == baseline, esa, changed)

    if p.run_this:


        # Note difference between simplified and full esa here.
        baseline_simplified_lulc_label = p.lulc_simplification_label + '_' + 'gtap1_baseline_' + str(p.key_base_year)

        baseline_esa_lulc_label = 'lulc_esa_gtap1_baseline_' + str(p.key_base_year)
        p.base_year_esa_lulc_path = os.path.join(p.cur_dir, baseline_esa_lulc_label + '.tif')

        if os.path.exists(os.path.join(p.stitched_lulc_simplified_scenarios_dir, baseline_simplified_lulc_label + '.tif')):
            p.aligned_simplified_output_base_map_path = os.path.join(p.stitched_lulc_simplified_scenarios_dir, baseline_simplified_lulc_label + '.tif')
        else:
            p.aligned_simplified_output_base_map_path = p.base_year_lulc_path

        # if not hb.path_exists(p.base_year_esa_lulc_path):
        #     hb.copy_shutil_flex(p.base_year_lulc_path, p.base_year_esa_lulc_path)

        if p.is_magpie_run and 0:
            for baseline_label in p.baseline_labels:
                for year in p.key_base_years:
                    simplified_include_string = p.lulc_simplification_label + '_' + baseline_label + '_' + str(year) + '_adjusted'
                    esa_include_string = 'lulc_esa_gtap1_' + baseline_label + '_' + str(year) + '_adjusted'

                    p.simplified_projected_stitched_path = os.path.join(p.stitched_lulc_simplified_scenarios_dir, simplified_include_string + '.tif')

                    stitched_bb = hb.get_bounding_box(p.simplified_projected_stitched_path)
                    global_bb = hb.get_bounding_box(p.base_year_lulc_path)

                    hb.log('Stitched_bb: ' + str(stitched_bb))
                    hb.log('global_bb: ' + str(global_bb))

                    if stitched_bb != global_bb:
                        baseline_simplified_lulc_label = 'lulc_esa_gtap1_baseline_' + str(p.key_base_year)

                        p.aligned_esa_output_base_map_path = os.path.join(p.cur_dir, baseline_simplified_lulc_label + '.tif')
                        if not hb.path_exists(p.aligned_esa_output_base_map_path):
                            hb.clip_raster_by_bb(p.base_year_lulc_path, stitched_bb, p.aligned_esa_output_base_map_path)
                    else:
                        p.aligned_esa_output_base_map_path = p.base_year_lulc_path

                    base_raster_path_band_const_list = [
                        (p.simplified_projected_stitched_path, 1),
                        (p.aligned_simplified_output_base_map_path, 1),
                        (p.aligned_esa_output_base_map_path, 1),
                    ]
                    target_raster_pre_path = hb.temp('.tif', simplified_include_string + '_prereclass', True, p.cur_dir)
                    # target_raster_pre_path = os.path.join(p.cur_dir, esa_include_string + '_pre.tif')
                    p.lulc_projected_stitched_path = os.path.join(p.cur_dir, esa_include_string + '.tif')

                    datatype_target = 1
                    nodata_target = 255

                    # TODOO Massive optimization here would be to just have the reclass happen in the fill_where_not_changed function..., and or make it parallel.
                    if not hb.path_exists(target_raster_pre_path) and not hb.path_exists(p.lulc_projected_stitched_path):
                        hb.log('Starting raster calculator with ' + str(target_raster_pre_path) + ' and ' + str(base_raster_path_band_const_list))
                        hb.raster_calculator_hb(
                            base_raster_path_band_const_list, fill_where_not_changed, target_raster_pre_path,
                            datatype_target, nodata_target,
                            gtiff_creation_options=hb.DEFAULT_GTIFF_CREATION_OPTIONS,
                            calc_raster_stats=False,
                            largest_block=hb.LARGEST_ITERBLOCK)

                    rules_extended_with_existing_esa_classes = dict(hb.seals_simplified_to_esa_rules)
                    rules_extended_with_existing_esa_classes.update({i: i for i in hb.esacci_extended_classes})
                    rules_extended_with_existing_esa_classes.update({255: 255})
                    if not hb.path_exists(p.lulc_projected_stitched_path):
                        # TODOO I broke the previous functionality where it could either replace with zeros or replace with default when a value wasn't in the dictionary.
                        hb.log('Starting reclassify_raster_hb with ' + str(target_raster_pre_path) + ' and ' + str(p.lulc_projected_stitched_path))

                        hb.reclassify_raster_hb(target_raster_pre_path, rules_extended_with_existing_esa_classes, p.lulc_projected_stitched_path, output_data_type=1)

                    if p.write_global_lulc_overviews_and_tifs:
                        if p.aoi == 'global':
                            hb.make_path_global_pyramid(p.lulc_projected_stitched_path)
                        # else:
                        #     hb.make_path_spatially_clean(p.lulc_projected_stitched_path)

        for scenario_pairing_label in p.scenario_pairing_labels:
            for year in p.scenario_years:
                for policy_scenario_label in p.policy_scenario_labels:
                    simplified_include_string = p.lulc_simplification_label + '_'  + scenario_pairing_label + '_' + str(year) + '_' + policy_scenario_label
                    esa_include_string = 'lulc_esa_gtap1_' + scenario_pairing_label + '_' + str(year) + '_' + policy_scenario_label

                    p.simplified_projected_stitched_path = os.path.join(p.stitched_lulc_simplified_scenarios_dir, simplified_include_string + '.tif')
                    p.lulc_projected_stitched_path = os.path.join(p.stitched_lulc_simplified_scenarios_dir, esa_include_string + '.tif')

                    stitched_bb = hb.get_bounding_box(p.simplified_projected_stitched_path)
                    global_bb = hb.get_bounding_box(p.base_year_lulc_path)

                    hb.log('Stitched_bb: ' + str(stitched_bb))
                    hb.log('global_bb: ' + str(global_bb))

                    if stitched_bb != global_bb:
                        baseline_esa_gtap1_lulc_label = 'lulc_esa_gtap1_baseline_' + str(p.key_base_year)

                        p.aligned_esa_output_base_map_path = os.path.join(p.cur_dir, baseline_esa_gtap1_lulc_label + '.tif')
                        if not hb.path_exists(p.aligned_esa_output_base_map_path):
                            hb.clip_raster_by_bb(p.base_year_lulc_path, stitched_bb, p.aligned_esa_output_base_map_path)
                    else:
                        p.aligned_esa_output_base_map_path = p.base_year_lulc_path

                    base_raster_path_band_const_list = [
                        (p.simplified_projected_stitched_path, 1),
                        (p.aligned_simplified_output_base_map_path, 1),
                        (p.aligned_esa_output_base_map_path, 1),
                    ]

                    target_raster_pre_path = hb.temp('.tif', esa_include_string + '_prereclass', True, p.cur_dir)

                    p.lulc_projected_stitched_path = os.path.join(p.cur_dir, esa_include_string + '.tif')

                    datatype_target = 1
                    nodata_target = 255

                    # TODOO Massive optimization here would be to just have the reclass happen in the fill_where_not_changed function..., and or make it parallel.
                    if not hb.path_exists(target_raster_pre_path) and not hb.path_exists(p.lulc_projected_stitched_path):
                        hb.log('Starting raster calculator with ' + str(target_raster_pre_path) + ' and ' + str(base_raster_path_band_const_list))

                        hb.raster_calculator_hb(
                            base_raster_path_band_const_list, fill_where_not_changed, target_raster_pre_path,
                            datatype_target, nodata_target,
                            gtiff_creation_options=hb.DEFAULT_GTIFF_CREATION_OPTIONS,
                            calc_raster_stats=False,
                            largest_block=hb.LARGEST_ITERBLOCK)

                    rules_extended_with_existing_esa_classes = dict(hb.seals_simplified_to_esa_rules)
                    rules_extended_with_existing_esa_classes.update({i: i for i in hb.esacci_extended_classes})
                    rules_extended_with_existing_esa_classes.update({255: 255})
                    if not hb.path_exists(p.lulc_projected_stitched_path):
                        # TODOO I broke the previous functionality where it could either replace with zeros or replace with default when a value wasn't in the dictionary.
                        hb.log('Starting reclassify_raster_hb with ' + str(target_raster_pre_path) + ' and ' + str(p.lulc_projected_stitched_path))

                        hb.reclassify_raster_hb(target_raster_pre_path, rules_extended_with_existing_esa_classes, p.lulc_projected_stitched_path, output_data_type=1)

                    if p.write_global_lulc_overviews_and_tifs:
                        if p.aoi == 'global':
                            hb.make_path_global_pyramid(p.lulc_projected_stitched_path)
                        # else:
                        #     hb.make_path_spatially_clean(p.lulc_projected_stitched_path)
