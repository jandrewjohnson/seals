"""MAJOR Reversion back to 0_2_8 because the WB project really needed to be chunk based rather than vector based."""

import collections
from collections import OrderedDict
import logging
import os
import copy
import warnings
import numpy as np
import hazelbean as hb
import scipy.ndimage
from osgeo import gdal

import multiprocessing
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import hazelbean.pyramids
import seals_utils
import pandas as pd
import time
import math

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

L = hb.get_logger()

# TODOO Make cython recompilation utilize project flow.
# Recompile cython file if needed.
recompile_cython = 0
if recompile_cython:
    # NOTE, successful recompilation assumes a strict definitino of where the project run_script is relative to the src dir.
    old_cwd = os.getcwd()
    os.chdir('../../../seals/seals_dev/seals')
    cython_command = "python compile_cython_functions.py build_ext -i clean"
    returned = os.system(cython_command)
    if returned:
        raise NameError('Cythonization failed.')
    os.chdir(old_cwd)
import seals_cython_functions as seals_cython_functions
from seals_cython_functions import calibrate as calibrate
from seals_cython_functions import calibrate_exclusive
from seals_cython_functions import calibrate_from_change_matrix

def initialize_paths(p):
    p.combined_block_lists_paths = None # This will be smartly determined in either calibration or allocation

    p.write_global_lulc_seals7_scenarios_overview_and_tifs = True
    p.write_global_lulc_esa_scenarios_overview_and_tifs = True

    p.countries_iso3_path = os.path.join(p.base_data_dir, 'pyramids', 'countries_iso3.gpkg'
                                                                      '')
    # To easily convert between per-ha and per-cell terms, these very accurate ha_per_cell maps are defined.
    p.ha_per_cell_10sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_10sec.tif")
    p.ha_per_cell_300sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_300sec.tif")
    p.ha_per_cell_900sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_900sec.tif")
    p.ha_per_cell_1800sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_1800sec.tif")
    p.ha_per_cell_3600sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_3600sec.tif")

    p.ha_per_cell_paths = {}
    p.ha_per_cell_paths[10.0] = p.ha_per_cell_10sec_path
    p.ha_per_cell_paths[300.0] = p.ha_per_cell_300sec_path
    p.ha_per_cell_paths[900.0] = p.ha_per_cell_900sec_path
    p.ha_per_cell_paths[1800.0] = p.ha_per_cell_1800sec_path
    p.ha_per_cell_paths[3600.0] = p.ha_per_cell_3600sec_path

    # The ha per cell paths also can be used when writing new tifs as the match path.
    p.match_10sec_path = p.ha_per_cell_10sec_path
    p.match_300sec_path = p.ha_per_cell_300sec_path
    p.match_900sec_path = p.ha_per_cell_900sec_path
    p.match_1800sec_path = p.ha_per_cell_1800sec_path
    p.match_3600sec_path = p.ha_per_cell_3600sec_path

    p.match_paths = {}
    p.match_paths[10.0] = p.match_10sec_path
    p.match_paths[300.0] = p.match_300sec_path
    p.match_paths[900.0] = p.match_900sec_path
    p.match_paths[1800.0] = p.match_1800sec_path
    p.match_paths[3600.0] = p.match_3600sec_path

    p.match_float_paths = p.match_paths.copy()


    p.ha_per_cell_column_10sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_column_10sec.tif")
    p.ha_per_cell_column_300sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_column_300sec.tif")
    p.ha_per_cell_column_900sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_column_900sec.tif")
    p.ha_per_cell_column_1800sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_column_1800sec.tif")
    p.ha_per_cell_column_3600sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_column_3600sec.tif")

    p.ha_per_cell_column_paths = {}
    p.ha_per_cell_column_paths[10.0] = p.ha_per_cell_column_10sec_path
    p.ha_per_cell_column_paths[300.0] = p.ha_per_cell_column_300sec_path
    p.ha_per_cell_column_paths[900.0] = p.ha_per_cell_column_900sec_path
    p.ha_per_cell_column_paths[1800.0] = p.ha_per_cell_column_1800sec_path
    p.ha_per_cell_column_paths[3600.0] = p.ha_per_cell_column_3600sec_path

    p.luh_data_dir = os.path.join(p.base_data_dir, 'luh2', 'raw_data')

    p.luh_scenario_states_paths = {}
    p.luh_scenario_states_paths['rcp26_ssp1'] = os.path.join(p.luh_data_dir, 'rcp26_ssp1', r"multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-IMAGE-ssp126-2-1-f_gn_2015-2100.nc")
    p.luh_scenario_states_paths['rcp34_ssp4'] = os.path.join(p.luh_data_dir, 'rcp34_ssp4', r"multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-GCAM-ssp434-2-1-f_gn_2015-2100.nc")
    p.luh_scenario_states_paths['rcp45_ssp2'] = os.path.join(p.luh_data_dir, 'rcp45_ssp2', r"multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-MESSAGE-ssp245-2-1-f_gn_2015-2100.nc")
    p.luh_scenario_states_paths['rcp60_ssp4'] = os.path.join(p.luh_data_dir, 'rcp60_ssp4', r"multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-GCAM-ssp460-2-1-f_gn_2015-2100.nc")
    p.luh_scenario_states_paths['rcp70_ssp3'] = os.path.join(p.luh_data_dir, 'rcp70_ssp3', r"multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-AIM-ssp370-2-1-f_gn_2015-2100.nc")
    p.luh_scenario_states_paths['rcp85_ssp5'] = os.path.join(p.luh_data_dir, 'rcp85_ssp5', r"multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-MAGPIE-ssp585-2-1-f_gn_2015-2100.nc")
    p.luh_scenario_states_paths['historical'] = os.path.join(p.luh_data_dir, 'historical', r"states.nc")

def extract_magpie_style_nc(p):

    if p.run_this:

        # Extract for base year(s)
        for baseline_label in p.baseline_coarse_state_paths:

            for baseline_year in p.baseline_coarse_state_paths[baseline_label]:

                # This is some project input file specific stuff; in the latest release, 2015 = 0, 2050 = 1
                time_index = baseline_year - 2015
                scenario_dir = os.path.join(p.cur_dir, baseline_label, str(baseline_year))
                hb.create_directories(scenario_dir)
                scenario_nc_path = p.baseline_coarse_state_paths[baseline_label][baseline_year]
                p.L.info('Extracting for time_index', time_index)
                hb.extract_global_netcdf_to_geotiff(scenario_nc_path, scenario_dir, output_files_prefix=baseline_label, time_indices_to_extract=time_index, flip_array_vertically=True, verbose=True)

                # Rename files to not have long cell.land_0.5_share_to_seals....



        # Extract for scenarios future year
        for luh_scenario_label in p.luh_scenario_labels:
            for year in p.scenario_years:

                # This is some project input file specific stuff; in the latest release, 2015 = 0, 2050 = 1
                time_index = year - 2049
                for policy_scenario_label in p.policy_scenario_labels:
                    scenario_dir = os.path.join(p.cur_dir, luh_scenario_label, str(year), policy_scenario_label)
                    hb.create_directories(scenario_dir)
                    scenario_nc_path = p.scenario_coarse_state_paths[luh_scenario_label][year][policy_scenario_label]
                    hb.extract_global_netcdf_to_geotiff(scenario_nc_path, scenario_dir, output_files_prefix=policy_scenario_label, time_indices_to_extract=time_index, flip_array_vertically=True, verbose=True)


def prepare_global_lulc(p):
    """For the purposes of calibration, create change-matrices for each coarse grid-cell based on two observed ESA lulc maps.
    Does something similar to prepare_lulc"""

    if p.is_magpie_run:
        p.global_coarse_dummy_path = os.path.join(p.extract_magpie_style_nc_dir, 'baseline', str(p.base_year), 'baseline_crop_' + str(p.base_year) + '.tif')
    elif p.is_gtap1_run:
        p.global_coarse_dummy_path = os.path.join(p.base_data_dir, r'luh2\processed_data\states_and_management\RCP26_SSP1\2015\states\c3ann.tif')


    t1 = hb.ArrayFrame(p.training_start_year_seals7_lulc_path)
    t2 = hb.ArrayFrame(p.training_end_year_seals7_lulc_path)

    if p.run_this:
        p.coarse_ha_per_cell = hb.ArrayFrame(p.coarse_ha_per_cell_path)
        p.coarse_match = hb.ArrayFrame(p.coarse_ha_per_cell_path)

        fine_cells_per_coarse_cell = round((p.coarse_ha_per_cell.cell_size / t1.cell_size) ** 2)
        aspect_ratio = t1.num_cols / p.coarse_match.num_cols

        output_arrays = np.zeros((len(p.class_indices), p.coarse_match.shape[0], p.coarse_match.shape[1]))
        numpy_output_path = os.path.join(p.cur_dir, 'change_matrices.npy')
        if not hb.path_exists(numpy_output_path):
            for r in range(p.coarse_match.num_rows):
                p.L.info('Processing observed change row', r)
                for c in range(p.coarse_match.num_cols):
                    t1_subarray = t1.data[int(r * aspect_ratio): int((r + 1) * aspect_ratio), int(c * aspect_ratio): int((c + 1) * aspect_ratio)]
                    t2_subarray = t2.data[int(r * aspect_ratio): int((r + 1) * aspect_ratio), int(c * aspect_ratio): int((c + 1) * aspect_ratio)]
                    # ha_per_cell_subarray = p.coarse_ha_per_cell.data[int(r * aspect_ratio): int((r + 1) * aspect_ratio), int(c * aspect_ratio): int((c + 1) * aspect_ratio)]

                    ha_per_coarse_cell_this_subarray = p.coarse_ha_per_cell.data[r, c]
                    change_matrix, counters = hb.calc_change_matrix_of_two_int_arrays(t1_subarray.astype(np.int), t2_subarray.astype(np.int), p.class_indices)
                    # Potentially unused relic from prepare_lulc
                    full_change_matrix = np.zeros((len(p.class_indices), len(p.class_indices)))
                    vector = seals_utils.calc_change_vector_of_change_matrix(change_matrix)

                    ha_per_cell_this_subarray = p.coarse_ha_per_cell.data[r, c] / fine_cells_per_coarse_cell

                    if vector:
                        for i in p.class_indices:
                            output_arrays[i - 1, r, c] = vector[i - 1] * ha_per_cell_this_subarray
                    else:
                        output_arrays[i, r, c] = 0.0

            for c, class_label in enumerate(p.class_labels):
                output_path = os.path.join(p.cur_dir, class_label + '_observed_change.tif')
                hb.save_array_as_geotiff(output_arrays[c], output_path, p.coarse_match.path)
            hb.save_array_as_npy(output_arrays, numpy_output_path)

        # Stores all of the classes in a 3d array ready for validation exercises below.
        change_3d = hb.load_npy_as_array(numpy_output_path)

        # Sometimes you don't want the change but need the actual state maps (ala luh) implied by a given ESA map.
        # Here calculates a cython function that downscales a fine_categorical to a stack of coarse_continuous 3d
        # Test that this is equivilent to change_3d
        p.base_year_seals7_lulc_path = hb.global_esa_seals7_lulc_paths_by_year[p.base_years[0]]

        p.observed_lulc_paths_to_calculate_states = [hb.global_esa_seals7_lulc_paths_by_year[2000], hb.global_esa_seals7_lulc_paths_by_year[2010], p.base_year_seals7_lulc_path]

        p.years_to_calculate_states = [2000, 2010] + p.base_years
        p.observed_state_paths = {}
        for year in p.years_to_calculate_states:
            p.observed_state_paths[year] = {}
            for class_label in p.class_labels:
                p.observed_state_paths[year][class_label] = os.path.join(p.cur_dir, hb.file_root(hb.global_esa_seals7_lulc_paths_by_year[year]) + '_state_' + str(class_label) + '_observed.tif')

        global_bb = hb.get_bounding_box(p.observed_lulc_paths_to_calculate_states[0])
        # TODOO Here incorporate test-mode bb.
        # stitched_bb = hb.get_bounding_box()

        for c, year in enumerate(p.years_to_calculate_states):
            if not hb.path_exists(p.observed_state_paths[year][p.class_label[0]], verbose=True):
            # if not all([hb.path_exists(i) for i in p.observed_state_paths[year]]):

                fine_path = p.observed_lulc_paths_to_calculate_states[c]
                p.L.info('Calculating coarse_state_stack from ' + fine_path)
                coarse_match_path = p.global_coarse_dummy_path
                output_dir = p.cur_dir

                fine_input_array = hb.load_geotiff_chunk_by_bb(fine_path, global_bb, datatype=5)
                coarse_match_array = hb.load_geotiff_chunk_by_bb(coarse_match_path, global_bb, datatype=6)

                chunk_edge_length = int(fine_input_array.shape[0] / coarse_match_array.shape[0])

                max_value_to_summarize = 8
                values_to_summarize = np.asarray(p.class_indices, dtype=np.int32)

                coarse_state_3d = hb.calculate_coarse_state_stack_from_fine_classified(fine_input_array,
                                                                      coarse_match_array,
                                                                      values_to_summarize,
                                                                      max_value_to_summarize)

                c = 0
                for k, v in p.observed_state_paths[year].items():
                    # Convert a count of states to a proportion of grid-cell
                    a = coarse_state_3d[c].astype(np.float32) / np.float32(chunk_edge_length ** 2)
                    hb.save_array_as_geotiff(a, v, coarse_match_path, data_type=6)
                    c += 1


# TODOO Note that there are some functions, like luh2_difference_from_base_year, that should have been in seals_main, but are in gtap_invest_main. Move them when you reorganize gtap_shifting to be vector_shifting 3-layer vs 2-layer
def convert_magpie_style_coarse_totals_to_seals7(p):
    magpie_to_seals7_correspondence = {}
    magpie_to_seals7_correspondence['urban'] = 'urban'
    magpie_to_seals7_correspondence['crop'] = 'cropland'
    magpie_to_seals7_correspondence['forestry'] = 'forest'
    magpie_to_seals7_correspondence['past'] = 'grassland'
    magpie_to_seals7_correspondence['forest'] = 'forest'
    magpie_to_seals7_correspondence['primforest'] = 'forest'
    magpie_to_seals7_correspondence['primother'] = 'nonforestnatural'
    magpie_to_seals7_correspondence['secdforest'] = 'forest'
    magpie_to_seals7_correspondence['secother'] = 'nonforestnatural'
    magpie_to_seals7_correspondence['other'] = 'nonforestnatural'

    hb.seals_simplified_to_esacci_correspondence

    if p.run_this:

        ha_per_cell_column = hb.as_array(hb.pyramid_ha_per_cell_column[p.coarse_arcseconds])

        # Optional step: make a new base-year map that is consistent with the magpie results, which may not be the same
        if p.adjust_baseline_to_match_magpie_2015:

            for baseline_label in p.baseline_coarse_state_paths:
                for base_year in p.base_years:
                    baseline_extract_dir = os.path.join(p.extract_magpie_style_nc_dir, baseline_label, str(base_year))

                    output_dir = os.path.join(p.cur_dir, baseline_label, str(base_year))
                    hb.create_directories(output_dir)

                    for class_index in p.class_indices:
                        esa_base_year_implied_path = os.path.join(p.prepare_global_lulc_dir, 'lulc_esa_simplified_' + str(base_year) + '_state_' + str(class_index) + '_observed.tif')
                        magpie_base_year_implied_path = os.path.join(baseline_extract_dir, 'baseline_' + p.shortened_class_labels[class_index-1] + '_' + str(base_year) + '.tif')
                        a = hb.as_array(magpie_base_year_implied_path) - hb.as_array(esa_base_year_implied_path)
                        output_path = os.path.join(output_dir, 'magpie_' + p.shortened_class_labels[class_index-1] + '_' + str(base_year) + '_ha_difference_from_esa.tif')
                        hb.save_array_as_geotiff(a * ha_per_cell_column, output_path, esa_base_year_implied_path)

        # Note, difference shows pretty big discrepency between magpie2015 and esa2015. If this isn't a problem, now just need to do 2 generations of seals.
        for luh_scenario_label in p.luh_scenario_labels:
            for year in p.scenario_years:
                for policy_scenario_label in p.policy_scenario_labels:

                    p.magpie_long_label = 'cell.land_0.5_share_to_seals'
                    # long_label = 'SSP2_test_cell.land_0.5_primother_share'
                    p.magpie_short_label = 'magpie'

                    # Hack, just pulling the first baseyear from the list because i need to rethink how I'm doing dynamic overall and this was a shortcut.
                    baseline_label = 'baseline'
                    baseline_extract_dir = os.path.join(p.extract_magpie_style_nc_dir,  baseline_label, str(p.base_years[0]))

                    # extract_dir = os.path.join(p.extract_magpie_style_nc_dir, luh_scenario_label, str(p.base_year), policy_scenario_label)
                    output_dir = os.path.join(p.cur_dir, luh_scenario_label, str(year), policy_scenario_label)
                    hb.create_directories(output_dir)

                    for class_index in p.class_indices:
                        # esa_base_year_implied_path = os.path.join(p.prepare_global_lulc_dir, 'lulc_esa_simplified_' + str(base_year) + '_state_' + str(class_index) + '_observed.tif')

                        scenario_year_implied_path = os.path.join(p.extract_magpie_style_nc_dir, luh_scenario_label, str(year), policy_scenario_label, policy_scenario_label + '_' + p.shortened_class_labels[class_index - 1] + '_' + str(year) + '.tif')
                        if p.adjust_baseline_to_match_magpie_2015:
                            # base_year_implied_path = os.path.join(output_dir, 'magpie_' + p.shortened_class_labels[class_index - 1] + '_' + str(base_year) + '_ha_difference_from_esa.tif')
                            base_year_implied_path = os.path.join(p.prepare_global_lulc_dir, 'lulc_esa_simplified_' + str(p.base_years[0]) + '_state_' + str(class_index) + '_observed.tif')
                        else:
                            base_year_implied_path = os.path.join(baseline_extract_dir, 'baseline_' + p.shortened_class_labels[class_index - 1] + '_' + str(p.base_years[0]) + '.tif')

                        a = hb.as_array(scenario_year_implied_path) - hb.as_array(base_year_implied_path)
                        output_path = os.path.join(output_dir, 'magpie_' + p.shortened_class_labels[class_index-1] + '_' + str(year) + '_' + str(p.base_years[0]) + '_ha_difference.tif')
                        # output_path = hb.ruri(output_path)
                        p.L.info('Calcualting differene in hectarage of ' + str(scenario_year_implied_path) + ' ' + str(base_year_implied_path) + ', writing to ' + str(output_path))
                        # hb.save_array_as_geotiff(a, output_path, esa_base_year_implied_path)

                        hb.save_array_as_geotiff(a * ha_per_cell_column, output_path, scenario_year_implied_path)

def calibration_generated_inputs(p):
    """DEPRECATED IN FAVOR OF regressors_starting_values, possibly can delete. Create an xls with starting-guess parameters for the SEALS calibration run. This identifies where the
    input data (and generated base data) is stored, parsed to the class-simpliicaiton scheme used."""

    p.coefficients_training_starting_value_path = os.path.join(p.cur_dir, 'coefficients_training_starting_value.xlsx')
    if p.run_this:

        column_headers = ['spatial_regressor_name', 'data_location', 'type']
        column_headers.extend(['class_' + str(i) for i in p.class_labels])

        df_input_2d_list = []

        # HACK, TODOO this should be tied to if it's a magpie run, or better yet, a more robust specification of the reclassification for the simplification.
        # lulc_alternate_reclassification_string = '_mosaic_is_natural'
        lulc_alternate_reclassification_string = ''

        # Write the default starting coefficients

        # Set Multiplicative (constraint) coefficients
        # For barren and water binaries set it to zero or 1 for others.
        for c, label in enumerate(p.regression_input_class_labels):
            row = [label + '_presence_constraint', os.path.join(
                p.base_data_dir, 'lulc', 'esa', 'seals7', 'binaries' + lulc_alternate_reclassification_string, str(p.training_start_year), 'lulc_esa_seals7_' + str(p.training_start_year) + '_class_' + str(p.regression_input_class_indices[c]) + '_binary.tif'),
                   'multiplicative'] + \
                  [0 if i == p.regression_input_class_indices[c] or p.regression_input_class_indices[c] in [6, 7] else 1 for i in p.class_indices]
            df_input_2d_list.append(row)

        # Set additive coefficients
        # for class binaries
        for c, label in enumerate(p.regression_input_class_labels):
            row = [label + '_presence', os.path.join(
                p.base_data_dir, 'lulc', 'esa', 'seals7', 'binaries' + lulc_alternate_reclassification_string, str(p.training_start_year), 'lulc_esa_seals7_' + str(p.training_start_year) + '_class_' + str(p.regression_input_class_indices[c]) + '_binary.tif'),
                   'additive'] + [0 if i == p.regression_input_class_indices[c] else 0 for i in p.class_indices]
            df_input_2d_list.append(row)

        # for class convolutions of sigma 1
        for c, label in enumerate(p.regression_input_class_labels):
            row = [label + '_gaussian_1', os.path.join(
                p.base_data_dir, 'lulc', 'esa', 'seals7', 'convolutions' + lulc_alternate_reclassification_string, str(p.training_start_year), 'class_' + str(p.regression_input_class_indices[c]) + '_gaussian_' + str(1) + '_convolution.tif'),
                   'additive'] + [0 if i == p.regression_input_class_indices[c] else 0 for i in p.class_indices]
            df_input_2d_list.append(row)

        # for class convolutions of sigma 5, set to zero except for diagonal (self edge expansion)
        for c, label in enumerate(p.regression_input_class_labels):
            row = [label + '_gaussian_5', os.path.join(
                p.base_data_dir, 'lulc', 'esa', 'seals7', 'convolutions' + lulc_alternate_reclassification_string,  str(p.training_start_year), 'class_' + str(p.regression_input_class_indices[c]) + '_gaussian_' + str(5) + '_convolution.tif'),
                   'additive'] + [1 if i == p.regression_input_class_indices[c] else 0 for i in p.class_indices]
            df_input_2d_list.append(row)

        # for all static variables, set to zero, except for as a hack one of them so that the it is edefined everyone.
        for static_regressor_label, path in p.static_regressor_paths.items():
            row = [static_regressor_label, path,
                   'additive'] + [1 if static_regressor_label == 'soil_organic_content' else 0 for i in p.class_indices]
            df_input_2d_list.append(row)

        df = pd.DataFrame(df_input_2d_list, columns=column_headers)
        df.set_index('spatial_regressor_name', inplace=True)

        df.to_excel(p.coefficients_training_starting_value_path)



def calibration(p):

    # Need to set projected_coarse_change_dir differently for magpie vs gtap runs because they have different tasks that generate their coarse change files.
    if p.is_magpie_run:
        p.projected_coarse_change_dir = os.path.join(p.convert_magpie_style_coarse_totals_to_seals7_dir, p.luh_scenario_labels[0], str(p.base_years[0]))
    elif p.is_gtap1_run:
        p.projected_coarse_change_dir = os.path.join(p.gtap_results_joined_with_luh_change_dir, p.current_luh_scenario_label, str(p.current_year), p.current_policy_scenario_label)
    elif p.is_calibration_run:
        p.projected_coarse_change_dir = os.path.join(p.seals7_difference_from_base_year_dir, p.luh_scenario_labels[0], str(p.scenario_years[0]))
    elif p.is_standard_seals_run:
        p.projected_coarse_change_dir = p.input_dir
    else:
        raise NameError('Unhandled.')

    # Generate lists of which zones change and thus need to be rerun
    if p.combined_block_lists_paths is None:
        p.combined_block_lists_paths = {
            'fine_blocks_list': os.path.join(p.cur_dir, 'fine_blocks_list.csv'),
            'coarse_blocks_list': os.path.join(p.cur_dir, 'coarse_blocks_list.csv'),
            'global_fine_blocks_list': os.path.join(p.cur_dir, 'global_fine_blocks_list.csv'),
            'global_coarse_blocks_list': os.path.join(p.cur_dir, 'global_coarse_blocks_list.csv'),
            'global_processing_blocks_list': os.path.join(p.cur_dir, 'global_processing_blocks_list.csv'),
        }

    try:
        if all(hb.path_exists(i) for i in combined_block_lists_paths):
            blocks_lists_already_exist = True
        else:
            blocks_lists_already_exist = False
    except:
        blocks_lists_already_exist = False

    if blocks_lists_already_exist:
        fine_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'fine_blocks_list.csv'), '2d_list'))
        coarse_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'coarse_blocks_list.csv'), '2d_list'))
        global_fine_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'global_fine_blocks_list.csv'), '2d_list'))
        global_coarse_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'global_coarse_blocks_list.csv'), '2d_list'))
        global_processing_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'global_processing_blocks_list.csv'), '2d_list'))

    else:
        if p.aoi == 'global':

            fine_blocks_list = hb.get_global_block_list_from_resolution(p.processing_block_size, p.fine_resolution)
            coarse_blocks_list = hb.get_global_block_list_from_resolution(p.processing_block_size, p.coarse_resolution)
            global_fine_blocks_list = fine_blocks_list
            global_coarse_blocks_list = coarse_blocks_list
            global_processing_blocks_list = hb.get_global_block_list_from_resolution_and_bb(p.processing_block_size,
                                                                                            p.processing_block_size, p.bb)

        else:
            fine_blocks_list = hb.get_subglobal_block_list_from_resolution_and_bb(p.processing_block_size, p.fine_resolution, p.bb)
            coarse_blocks_list = hb.get_subglobal_block_list_from_resolution_and_bb(p.processing_block_size, p.coarse_resolution, p.bb)
            global_fine_blocks_list = hb.get_global_block_list_from_resolution_and_bb(p.processing_block_size, p.fine_resolution, p.bb)
            global_coarse_blocks_list = hb.get_global_block_list_from_resolution_and_bb(p.processing_block_size, p.coarse_resolution, p.bb)
            global_processing_blocks_list = hb.get_global_block_list_from_resolution_and_bb(p.processing_block_size, p.processing_block_size, p.bb)
            # global_processing_blocks_list = hb.get_global_block_list_from_resolution(p.processing_block_size, p.coarse_resolution)
        p.L.info('Length of iterator before pruning in task calibration:', len(fine_blocks_list))

    if p.subset_of_blocks_to_run is not None:
        old_fine_blocks_list = fine_blocks_list
        old_coarse_blocks_list = coarse_blocks_list

        fine_blocks_list = []
        coarse_blocks_list = []

        for i in p.subset_of_blocks_to_run:
            fine_blocks_list.append(old_fine_blocks_list[i])
            coarse_blocks_list.append(old_coarse_blocks_list[i])

    p.L.info('Length of iterator after considering subset_of_blocks_to_run:', len(fine_blocks_list))

    combined_block_lists_dict = {
        'fine_blocks_list': fine_blocks_list,
        'coarse_blocks_list': coarse_blocks_list,
        'global_fine_blocks_list': global_fine_blocks_list,
        'global_coarse_blocks_list': global_coarse_blocks_list,
        'global_processing_blocks_list': global_processing_blocks_list,
    }

    if not all([hb.path_exists(i) for i in p.combined_block_lists_paths.values()]):

        # Pare down the number of blocks to run based on if there is change in the projected_coarse_change
        old_fine_blocks_list = fine_blocks_list
        old_coarse_blocks_list = coarse_blocks_list
        old_global_fine_blocks_list = global_fine_blocks_list
        old_global_coarse_blocks_list = global_coarse_blocks_list
        old_global_processing_blocks_list = global_processing_blocks_list
        fine_blocks_list = []
        coarse_blocks_list = []
        global_fine_blocks_list = []
        global_coarse_blocks_list = []
        global_processing_blocks_list = []
        L.info('Checking existing blocks for change in the LUH data and excluding if no change.')
        for c, block in enumerate(old_coarse_blocks_list):
            progress_percent = float(c) / float(len(old_coarse_blocks_list)) * 100.0
            print('Percent finished: ' + str(progress_percent), end='\r', flush=False)
            skip = []
            # TODOO  Make this only look at the correct data. NEVERY USE FILE EXISTNEC for figuring out anything other than if it needs to be recreated, not how it should be accessed later).
            current_coarse_change_rasters = []
            for class_label in p.class_labels:
                gen_path = os.path.join(p.projected_coarse_change_dir, class_label + '_' + str(p.scenario_years[0]) + '_' + str(p.base_years[0]) + '_ha_difference.tif')
                current_coarse_change_rasters.append(gen_path)
            for path in current_coarse_change_rasters:
                # for path in hb.list_filtered_paths_nonrecursively(p.projected_coarse_change_dir, include_extensions='.tif'):
                block = old_coarse_blocks_list[c]
                a = hb.load_geotiff_chunk_by_cr_size(path, block)
                changed = np.where((a != 0) & (a != -9999.) & (~np.isnan(a)), 1, 0)
                # hb.show(a)
                # hb.show(changed)
                p.L.debug('Checking to see if there is change in ', path)
                if np.nansum(changed) == 0:
                    p.L.debug('Skipping because no change in coarse projections:', path)
                    skip.append(True)
                else:
                    skip.append(False)

            if not all(skip):
                fine_blocks_list.append(old_fine_blocks_list[c])
                coarse_blocks_list.append(old_coarse_blocks_list[c])
                global_fine_blocks_list.append(old_global_fine_blocks_list[c])
                global_coarse_blocks_list.append(old_global_coarse_blocks_list[c])
                global_processing_blocks_list.append(old_global_processing_blocks_list[c])

        # Write the blockslists to csvs to avoid future reprocessing (actually is quite slow (2 mins) when 64000 tiles).
        for block_name, block_list in combined_block_lists_dict.items():
            hb.python_object_to_csv(block_list, os.path.join(p.cur_dir, block_name + '.csv'), '2d_list')

    else:
        fine_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'fine_blocks_list.csv'), '2d_list'))
        coarse_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'coarse_blocks_list.csv'), '2d_list'))
        global_fine_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'global_fine_blocks_list.csv'), '2d_list'))
        global_coarse_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'global_coarse_blocks_list.csv'), '2d_list'))
        global_processing_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'global_processing_blocks_list.csv'), '2d_list'))

    p.L.info('Length of iterator after removing non-changing zones:', len(fine_blocks_list))


    # Remove from iterator lists that have already been computed.
    old_fine_blocks_list = fine_blocks_list
    old_coarse_blocks_list = coarse_blocks_list
    old_global_fine_blocks_list = global_fine_blocks_list
    old_global_coarse_blocks_list = global_coarse_blocks_list
    old_global_processing_blocks_list = global_processing_blocks_list
    fine_blocks_list = []
    coarse_blocks_list = []
    global_fine_blocks_list = []
    global_coarse_blocks_list = []
    global_processing_blocks_list = []

    for c, fine_block in enumerate(old_fine_blocks_list):
        tile_dir = str(fine_block[4]) + '_' + str(fine_block[5])
        expected_path = os.path.join(p.cur_dir, tile_dir, 'calibration_zones', 'trained_coefficients_zone_' + tile_dir + '.xlsx')
        if not hb.path_exists(expected_path):
            fine_blocks_list.append(old_fine_blocks_list[c])
            coarse_blocks_list.append(old_coarse_blocks_list[c])
            global_fine_blocks_list.append(old_global_fine_blocks_list[c])
            global_coarse_blocks_list.append(old_global_coarse_blocks_list[c])
            global_processing_blocks_list.append(old_global_processing_blocks_list[c])

    p.L.info('Length of iterator after removing finished zones:', len(fine_blocks_list))

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


                # for k, v in zone_calibration_block_lookup_dict.items():
                #     print(k, v, zones_raster[v])


                with open(p.processing_zones_to_calibration_chunk_path, "w") as f:
                    for k, line in zone_calibration_block_lookup_dict.items():
                        # print(k, line)
                        f.write(str(k) + ',' + str(line[0]) + '_' + str(line[1]) + '\n')

    p.iterator_replacements = {}
    p.iterator_replacements['fine_blocks_list'] = fine_blocks_list
    p.iterator_replacements['coarse_blocks_list'] = coarse_blocks_list
    p.iterator_replacements['global_fine_blocks_list'] = global_fine_blocks_list
    p.iterator_replacements['global_coarse_blocks_list'] = global_coarse_blocks_list
    p.iterator_replacements['global_processing_blocks_list'] = global_processing_blocks_list

    # Trickier replacement that will redefine the parent dir for each task so that it also WRITES in the correct output location
    p.iterator_replacements['cur_dir_parent_dir'] = [p.intermediate_dir + '/calibration/' + str(i[4]) + '_' + str(i[5]) for i in fine_blocks_list]

    p.iterator_replacements['current_luh_scenario_label'] = ['rcp45_ssp2' for i in fine_blocks_list]
    p.iterator_replacements['current_year'] = [p.base_years[0] for i in fine_blocks_list]
    p.iterator_replacements['current_policy_scenario_label'] = ['calibration' for i in fine_blocks_list]


def calibration_prepare_lulc(passed_p=None):
    if passed_p is None:
        global p
    else:
        p = passed_p

    p.zone_esa_seals7_lulc_base_year_path = os.path.join(p.cur_dir, 'zone_esa_seals7_lulc_' + str(p.base_year) + '.tif')

    p.chunk_coarse_ha_per_cell_path = os.path.join(p.cur_dir, 'chunk_coarse_ha_per_cell.tif')

    p.lulc_class_types_path = r"C:\OneDrive\Projects\cge\seals\projects\ipbes\input\lulc_class_types.csv"

    # Problem here: Change vector method needs to be expanded to Change matrix, full from-to relationships
    # but when doing from-to, that only works when doing observed time-period validation. What would be the assumption for going into
    # the future? Possibly attempt to match prior change matrices, but only as a slight increase in probability? Secondly, why is my
    # search algorithm not itself finding the from-to relationships just by minimizing difference? Basically, need to take seriously deallocation.

    full_change_matrix_no_diagonal_path = os.path.join(p.cur_dir, 'full_change_matrix_no_diagonal.tif')

    if p.run_this:


        # Clip ha_per_cell and use it as the match
        chunk_coarse_ha_per_cell = hb.load_geotiff_chunk_by_cr_size(p.aoi_ha_per_cell_coarse_path, p.coarse_blocks_list, output_path=p.chunk_coarse_ha_per_cell_path)
        # hb.clip_raster_by_vector(p.coarse_ha_per_cell_path, p.coarse_ha_per_cell_path, p.aoi_path, all_touched=True, ensure_fits=True)
        p.chunk_coarse_ha_per_cell = hb.ArrayFrame(p.chunk_coarse_ha_per_cell_path)
        p.chunk_coarse_match = hb.ArrayFrame(p.chunk_coarse_ha_per_cell_path)

        # Get the processing zone ID
        current_processing_zone_path = os.path.join(p.cur_dir, 'processing_zone.tif')
        # processing_zone = hb.load_geotiff_chunk_by_cr_size(p.processing_zones_raster_path, p.global_processing_blocks_list, output_path=current_processing_zone_path)

        # calibration_zone =

        p.lulc_base_year_chunk_10sec_path = os.path.join(p.cur_dir, 'lulc_esa_seals7_' + str(p.base_year) + '.tif')
        p.lulc_training_start_year_chunk_10sec_path = os.path.join(p.cur_dir, 'lulc_esa_seals7_' + str(p.training_start_year) + '.tif')

        # Clip ha_per_cell and use it as the match
        hb.load_geotiff_chunk_by_cr_size(p.lulc_simplified_paths['lulc_esa_seals7_' + str(p.base_year)], p.fine_blocks_list, output_path=p.lulc_base_year_chunk_10sec_path)
        hb.load_geotiff_chunk_by_cr_size(p.lulc_simplified_paths['lulc_esa_seals7_' + str(p.training_start_year)], p.fine_blocks_list, output_path=p.lulc_training_start_year_chunk_10sec_path)
        p.fine_match = hb.ArrayFrame(p.lulc_base_year_chunk_10sec_path)

        p.coarse_ha_per_cell = hb.ArrayFrame(p.coarse_ha_per_cell_path)
        # p.coarse_match = hb.ArrayFrame(p.coarse_ha_per_cell_path)

        fine_cells_per_coarse_cell = round((p.chunk_coarse_ha_per_cell.cell_size / p.fine_match.cell_size) ** 2)
        aspect_ratio = int(p.fine_match.num_cols / p.chunk_coarse_match.num_cols)

        p.lulc_training_start_year_chunk = hb.ArrayFrame(p.lulc_training_start_year_chunk_10sec_path)

        # TODOOO later on, figure out why this failed.
        # if aspect_ratio != p.fine_match.num_cols / p.coarse_match.num_cols:
        #     p.L.info('aspect ratio ' + str(aspect_ratio) + ' not same as non-inted version ' + str(p.fine_match.num_cols / p.coarse_match.num_cols) + '. This could indicate non pyramidal data.')

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
                    # ha_per_cell_subarray = chunk_coarse_ha_per_cell.data[int(r * aspect_ratio): int((r + 1) * aspect_ratio), int(c * aspect_ratio): int((c + 1) * aspect_ratio)]

                    ha_per_coarse_cell_this_subarray = p.chunk_coarse_ha_per_cell.data[r, c]

                    change_matrix, counters = hb.calc_change_matrix_of_two_int_arrays(t1_subarray.astype(np.int), t2_subarray.astype(np.int), p.class_indices)

                    vector = seals_utils.calc_change_vector_of_change_matrix(change_matrix)

                    ha_per_cell_this_subarray = p.chunk_coarse_ha_per_cell.data[r, c] / fine_cells_per_coarse_cell

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
                print('current_net_change_array_path', current_net_change_array_path)
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
                for year in p.base_years:
                    p.projected_coarse_change_files_adjustment_run[scenario_label][year] = {}
                    for policy_scenario_label in ['baseline']:
                        p.projected_coarse_change_files_adjustment_run[scenario_label][year][policy_scenario_label] = {}
                        for class_label in current_class_labels:
                            implied_magpie_path = os.path.join(p.projected_coarse_change_dir, magpie_short_label + '_' + class_label + '_' + str(p.base_year) + '_ha_difference_from_esa.tif')
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
                                implied_luh_path = os.path.join(p.seals7_difference_from_base_year_dir, luh_scenario_label, str(year),
                                                                class_label + '_' + str(year) + '_' + str(p.base_year) + '_difference.tif')
                                p.projected_coarse_change_files[luh_scenario_label][year][policy_scenario_label][class_label] = implied_luh_path
                        elif p.is_magpie_run:
                            implied_magpie_path = os.path.join(p.projected_coarse_change_dir, magpie_short_label + '_' + class_label + '_' + str(year) + '_' + str(p.base_year) + '_ha_difference.tif')
                            p.projected_coarse_change_files[luh_scenario_label][year][policy_scenario_label][class_label] = implied_magpie_path
                        else:
                            implied_ssp_path = os.path.join(p.projected_coarse_change_dir, class_label + '_' + str(year) + '_' + str(p.base_year) + '_ha_difference.tif')
                            p.projected_coarse_change_files[luh_scenario_label][year][policy_scenario_label][class_label] = implied_ssp_path

        # TODOO FIX THIS HACK. How should I deal with adjustment runs... would this fit into the overall dynamic scenario iteration?
        if 'esa_luh_baseline_lulc_adjustment' in p.cur_dir_parent_dir:
            p.projected_coarse_change_files = p.projected_coarse_change_files_adjustment_run
            p.is_first_pass = True
        else:
            p.is_first_pass = False

        p.L.info('prepare_lulc looked for projected_coarse_change_files and found ', p.projected_coarse_change_files)
        if p.write_projected_coarse_change_chunks:
            for luh_scenario_label, v in p.projected_coarse_change_files.items():
                for year_label, vv in v.items():
                    for policy_scenario_label, vvv in vv.items():
                        for class_label, coarse_change_path in vvv.items():
                            current_net_change_array_path = os.path.join(p.cur_dir, class_label + '_projected_change.tif')

                            hb.load_geotiff_chunk_by_cr_size(coarse_change_path, p.coarse_blocks_list, output_path=current_net_change_array_path)

                            # STOPPED HERE, waiting for discussion from patrick:
                            # baseline adjustment is working, but the old change figs do'nt seem to look right. esp.
                            # C:\Files\Research\cge\gtap_invest\projects\test_magpie_seals\intermediate\convert_magpie_style_coarse_totals_to_seals7\rcp45_ssp2\2050\SSP2_BiodivPol_ClimPol_NCPpol_LPJmL5\magpie_forest_2050_2015_ha_difference.tif
                            # Also, need to hear from Patrick on just using change figs



def calibration_change_matrix(passed_p=None):
    if passed_p is None:
        global p
    else:
        p = passed_p


    p.calibration_full_change_matrix_no_diagonal_path = os.path.join(p.cur_dir, 'calibration_full_change_matrix_no_diagonal.tif')
    p.calibration_full_change_matrix_path = os.path.join(p.cur_dir, 'calibration_full_change_matrix.tif')

    if p.run_this:
        # Clip ha_per_cell and use it as the match
        chunk_coarse_ha_per_cell = hb.load_geotiff_chunk_by_cr_size(p.aoi_ha_per_cell_coarse_path, p.coarse_blocks_list, output_path=p.chunk_coarse_ha_per_cell_path)
        # hb.clip_raster_by_vector(p.coarse_ha_per_cell_path, p.coarse_ha_per_cell_path, p.aoi_path, all_touched=True, ensure_fits=True)
        p.chunk_coarse_ha_per_cell = hb.ArrayFrame(p.chunk_coarse_ha_per_cell_path)
        p.chunk_coarse_match = hb.ArrayFrame(p.chunk_coarse_ha_per_cell_path)

        p.lulc_base_year_chunk_10sec_path = os.path.join(p.cur_dir, 'lulc_esa_seals7_' + str(p.base_year) + '.tif')
        p.lulc_training_start_year_chunk_10sec_path = os.path.join(p.cur_dir, 'lulc_esa_seals7_' + str(p.training_start_year) + '.tif')

        # Clip ha_per_cell and use it as the match
        hb.load_geotiff_chunk_by_cr_size(p.lulc_simplified_paths['lulc_esa_seals7_' + str(p.base_year)], p.fine_blocks_list, output_path=p.lulc_base_year_chunk_10sec_path)

        hb.load_geotiff_chunk_by_cr_size(p.lulc_simplified_paths['lulc_esa_seals7_' + str(p.training_start_year)], p.fine_blocks_list, output_path=p.lulc_training_start_year_chunk_10sec_path)
        p.lulc_training_start_year_chunk = hb.ArrayFrame(p.lulc_training_start_year_chunk_10sec_path)

        p.lulc_base_year_chunk = hb.ArrayFrame(p.lulc_base_year_chunk_10sec_path)
        p.fine_match = hb.ArrayFrame(p.lulc_base_year_chunk_10sec_path)

        p.coarse_ha_per_cell = hb.ArrayFrame(p.coarse_ha_per_cell_path)
        # p.coarse_match = hb.ArrayFrame(p.coarse_ha_per_cell_path)

        fine_cells_per_coarse_cell = round((p.chunk_coarse_ha_per_cell.cell_size / p.fine_match.cell_size) ** 2)
        aspect_ratio = int(p.fine_match.num_cols / p.chunk_coarse_match.num_cols)


        net_change_output_arrays = np.zeros((len(p.class_indices), p.chunk_coarse_match.shape[0], p.chunk_coarse_match.shape[1]))
        full_change_matrix = np.zeros((len(p.class_indices * p.chunk_coarse_match.n_rows), len(p.class_indices) * p.chunk_coarse_match.n_cols))
        full_change_matrix_no_diagonal = np.zeros((len(p.class_indices * p.chunk_coarse_match.n_rows), len(p.class_indices) * p.chunk_coarse_match.n_cols))
        for r in range(p.chunk_coarse_match.num_rows):
            for c in range(p.chunk_coarse_match.num_cols):

                t1_subarray = p.lulc_training_start_year_chunk.data[int(r * aspect_ratio): int((r + 1) * aspect_ratio), int(c * aspect_ratio): int((c + 1) * aspect_ratio)]
                t2_subarray = p.lulc_base_year_chunk.data[int(r * aspect_ratio): int((r + 1) * aspect_ratio), int(c * aspect_ratio): int((c + 1) * aspect_ratio)]
                # ha_per_cell_subarray = chunk_coarse_ha_per_cell.data[int(r * aspect_ratio): int((r + 1) * aspect_ratio), int(c * aspect_ratio): int((c + 1) * aspect_ratio)]

                ha_per_coarse_cell_this_subarray = p.chunk_coarse_ha_per_cell.data[r, c]

                change_matrix, counters = hb.calc_change_matrix_of_two_int_arrays(t1_subarray.astype(np.int), t2_subarray.astype(np.int), p.class_indices)

                vector = seals_utils.calc_change_vector_of_change_matrix(change_matrix)

                verbose = False
                if verbose:
                    p.L.info('change_matrix in coarse chunk ' + str(r) + ', ' + str(c) + '\n' + str(change_matrix) + '\nwith change vector ' + str(vector))


                ha_per_cell_this_subarray = p.chunk_coarse_ha_per_cell.data[r, c] / fine_cells_per_coarse_cell

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

        if p.output_writing_level >= 3:
            write_change_matrix_rasters = 1
        else:
            write_change_matrix_rasters = 0

        # Note that this at least the full change matrix is needed because this is a raw input into the calibration.
        hb.save_array_as_geotiff(full_change_matrix, p.calibration_full_change_matrix_path, p.chunk_coarse_match.path, n_rows=full_change_matrix.shape[1], n_cols=full_change_matrix.shape[1])
        if write_change_matrix_rasters:
            hb.save_array_as_geotiff(full_change_matrix_no_diagonal, p.calibration_full_change_matrix_no_diagonal_path, p.chunk_coarse_match.path, n_rows=full_change_matrix_no_diagonal.shape[1], n_cols=full_change_matrix_no_diagonal.shape[1])

        # TODOO I disabled this section because I was getting invalid vmin, probably from a block with no change
        p.plot_change_matrices = 0
        if p.plot_change_matrices:
            from matplotlib import colors as colors
            fig, ax = plt.subplots()
            fig.set_size_inches(10, 8)

            # Plot the heatmap

            vmin = np.min(full_change_matrix_no_diagonal)
            vmax = np.max(full_change_matrix_no_diagonal)
            im = ax.imshow(full_change_matrix_no_diagonal, cmap='YlGnBu', norm=colors.LogNorm(vmin=vmin + 1, vmax=vmax))

            # Create colorbar
            import matplotlib.ticker as ticker

            cbar = ax.figure.colorbar(im, ax=ax, format=ticker.FuncFormatter(lambda x, p : int(x)))
            cbar.set_label('Number of cells changed from class ROW to class COL', size=10)

            # Set ticks...
            ax.set_xticks(np.arange(full_change_matrix_no_diagonal.shape[1]))
            ax.set_yticks(np.arange(full_change_matrix_no_diagonal.shape[0]))

            # Create labels for each coarse zone indexed by i and j
            row_labels = []
            col_labels = []
            for i in range(n_classes * p.chunk_coarse_match.n_rows):
                class_id = i % n_classes
                coarse_grid_cell_counter = int(i / n_classes)
                row_labels.append(str(class_id))
                col_labels.append(str(class_id))

            trans = ax.get_xaxis_transform()  # x in data untis, y in axes fraction

            for i in range(p.chunk_coarse_match.n_rows):
                ann = ax.annotate('Zone i=' + str(i + 1), xy=(-3.5, (p.chunk_coarse_match.n_rows - i) / p.chunk_coarse_match.n_rows - .5 / p.chunk_coarse_match.n_rows), xycoords=trans)
                ann = ax.annotate('Zone j=' + str(i + 1), xy=(i * (p.chunk_coarse_match.n_rows + 1) + .25 * p.chunk_coarse_match.n_rows, 1.05), xycoords=trans)  #

            ax.set_xticklabels(col_labels)
            ax.set_yticklabels(row_labels)

            # Let the horizontal axes labeling appear on top.
            ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

            plt.setp(ax.get_xticklabels(), ha="center", rotation_mode="anchor")

            # Turn spines off and create white grid.
            for edge, spine in ax.spines.items():
                spine.set_visible(False)

            ax.set_xticks(np.arange(full_change_matrix_no_diagonal.shape[1] + 1) - .5, minor=True)
            ax.set_yticks(np.arange(full_change_matrix_no_diagonal.shape[0] + 1) - .5, minor=True)
            ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
            ax.tick_params(which="minor", bottom=False, left=False)

            full_change_matrix_no_diagonal_png_path = os.path.join(p.cur_dir, 'fcmnd.png')
            # texts = annotate_heatmap(im, valfmt="{x:.1f} t")

            major_gridline = False
            for i in range(n_classes * p.chunk_coarse_match.n_rows + 1):
                try:
                    if i % n_classes == 0:
                        major_gridline = i
                    else:
                        major_gridline = False
                except:
                    major_gridline = 0

                if major_gridline is not False:
                    xloc = major_gridline - .5
                    yloc = major_gridline - .5
                    ax.axvline(x=xloc, color='grey')
                    ax.axhline(y=yloc, color='grey')

            plt.savefig(p.calibration_full_change_matrix_no_diagonal_path.replace('.tif', '.png'))

            # vmax = np.max(full_change_matrix_no_diagonal)
            # full_change_matrix_no_diagonal_png_path = os.path.join(p.cur_dir, 'full_change_matrix_no_diagona_auto.png')

            # Not really necessary but decent exampe of auto plot.
            # hb.full_show_array(full_change_matrix_no_diagonal, output_path=full_change_matrix_no_diagonal_png_path, cbar_label='Number of changes from class R to class C per tile', title='Change matrix mosaic',
            #                    num_cbar_ticks=2, vmin=0, vmid=vmax / 10.0, vmax=vmax, color_scheme='ylgnbu')

        p.chunk_coarse_ha_per_cell = None

def calibration_zones(passed_p=None):
    if passed_p is None:
        global p
    else:
        p = passed_p

    final_coefficients_path = os.path.join(p.cur_dir, 'trained_coefficients_zone_' + os.path.split(os.path.split(p.cur_dir)[0])[1] + '.xlsx')

    if p.run_this and not hb.path_exists(final_coefficients_path):
        starting_coefficients_path = p.local_data_regressors_starting_values_path
        # starting_coefficients_path = os.path.join(p.input_dir, 'spatial_regressor_starting_coefficients.xlsx')
        # current_pretrained_coefficients_path = p.pretrained_coefficients_path_dict[p.current_luh_scenario_label][p.current_year][p.current_policy_scenario_label]
        # current_pretrained_coefficients_path = p.pretrained_coefficients_path_dict[p.current_policy_scenario_label]
        # current_pretrained_coefficients_path = hb.get_existing_path_from_nested_sources(p.pretrained_coefficients_path_dict[p.current_policy_scenario_label], p, verbose=True)
        p.spatial_regressors_df = pd.read_excel(starting_coefficients_path)

        p.spatial_layer_names = p.spatial_regressors_df['spatial_regressor_name'].values
        p.spatial_layer_paths = p.spatial_regressors_df['data_location'].values
        p.spatial_layer_types = p.spatial_regressors_df['type'].values

        # TODOO For now, this doesnt allow for more than just a single parameter per layer, but next phase would extend this so that it was class_1_1, class_1_2, class_2_1, class_2_2, etc.
        p.seals_class_names = p.spatial_regressors_df.columns.values[3:]

        p.lulc_baseline_path = os.path.join(p.cur_dir, 'lulc_baseline.tif')
        p.zone_esa_seals7_lulc_base_year_path = os.path.join(p.cur_dir, 'zone_esa_seals7_lulc_base_year.tif')
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
        spatial_regressor_starting_coefficients_read = pd.read_excel(starting_coefficients_path, index_col=0)
        # spatial_regressor_starting_coefficients_read = pd.read_excel(os.path.join(p.input_dir, 'spatial_regressor_starting_coefficients.xlsx'), index_col=0)
        spatial_regressor_starting_coefficients = spatial_regressor_starting_coefficients_read[p.seals_class_names].values.astype(np.float64).T

        # TODOO I have inconsistent usage of p.lulc_simplification_label. Use this throughout.
        observed_lulc_array = hb.load_geotiff_chunk_by_cr_size(p.lulc_simplified_paths['lulc_esa_'+ p.lulc_simplification_label + '_' + str(p.training_end_year)], p.fine_blocks_list).astype(np.int64)
        p.lulc_ndv = hb.get_ndv_from_path(p.lulc_simplified_paths['lulc_esa_'+ p.lulc_simplification_label + '_' + str(p.training_end_year)])
        valid_mask_array = np.where((observed_lulc_array != p.lulc_ndv), 1, 0).astype(np.int64)

        p.observed_current_coarse_change_input_paths = hb.list_filtered_paths_nonrecursively(p.calibration_prepare_lulc_dir, include_strings='observed', include_extensions='.tif')


        coarse_n_c, coarse_n_r = int(p.coarse_blocks_list[2]), int(p.coarse_blocks_list[3])
        n_c, n_r = int(p.fine_blocks_list[2]), int(p.fine_blocks_list[3])

        spatial_layers_3d = np.zeros((len(p.spatial_layer_paths), n_r, n_c)).astype(np.float64)


        normalize_inputs = False
        for c, path in enumerate(p.spatial_layer_paths):
            p.L.debug('Loading spatial layer at path ' + path)
            current_bb = hb.get_bounding_box(path)
            if current_bb == hb.global_bounding_box:
                correct_fine_block_list = p.global_fine_blocks_list
                correct_coarse_block_list = p.global_coarse_blocks_list
            else:
                correct_fine_block_list = p.fine_blocks_list
                correct_coarse_block_list = p.coarse_blocks_list
            if p.spatial_layer_types[c] == 'additive' or p.spatial_layer_types[c] == 'multiplicative':
                if normalize_inputs is True:
                    spatial_layers_3d[c] = hb.normalize_array(hb.load_geotiff_chunk_by_cr_size(path, correct_fine_block_list))
                else:
                    spatial_layers_3d[c] = hb.load_geotiff_chunk_by_cr_size(path, correct_fine_block_list)
            elif p.spatial_layer_types[c][0:8] == 'gaussian':
                # updated_path = os.path.join(p.calibration_zones_dir, 'class_' + p.spatial_layer_names[c].split('_')[1] + '_gaussian_' + p.spatial_layer_names[c].split('_')[3] + '_convolution.tif')
                # L.debug('updated_path', updated_path)
                L.debug('path', path)
                if normalize_inputs is True:
                    spatial_layers_3d[c] = hb.normalize_array(hb.as_array(path))
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






        # p.projected_current_coarse_change_input_paths = []
        #
        # # p.projected_coarse_change_files[luh_scenario_label][year][policy_scenario_label][class_label]
        # for class_label in p.class_labels:
        #     p.projected_current_coarse_change_input_paths.append(os.path.join(p.prepare_lulc_dir, str(i) + '_projected_change.tif'))
        #



        # MAYBE STILL USEFUL?! Doesn't yet work with DataRef method
        plot_observed_coarse_change_3d = 1
        if plot_observed_coarse_change_3d:
            observed_coarse_change_3d = np.zeros((len(p.class_labels), coarse_n_r, coarse_n_c)).astype(np.float64)
            for c, class_label in enumerate(p.class_labels):
                path = os.path.join(p.cur_dir, '../calibration_prepare_lulc', class_label + '_observed_change.tif')
                observed_coarse_change_3d[c] = hb.as_array(path).astype(np.float64)
            seals_utils.plot_coarse_change_3d(p.cur_dir, observed_coarse_change_3d)


        for generation_id in range(p.num_generations):
            p.L.info('Starting generation ' + str(generation_id) + ' for location ' + str(p.fine_blocks_list))
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
            # p.change_class_labels_list = [int(i.split('_')[1]) for i in p.spatial_regressors_df.columns[3:]]
            p.change_class_labels = np.asarray(p.class_indices, dtype=np.int64)  # For Cythonization, load these as the "labels", which is used for writing.

            lulc_baseline_array = hb.load_geotiff_chunk_by_cr_size(p.lulc_simplified_paths['lulc_esa_seals7_' + str(p.training_start_year)], p.fine_blocks_list, datatype=np.int64, output_path=p.lulc_baseline_path).astype(np.int64)
            # lulc_baseline_array = hb.load_geotiff_chunk_by_cr_size(p.training_start_year_seals7_lulc_path, p.fine_blocks_list, datatype=np.int64, output_path=p.lulc_baseline_path).astype(np.int64)

            # Load spatial_regressors

            # # Consider adding a function like this to avoid all the loading bloat.
            # seals_utils.load_zone_fast_inputs(p)

            p.spatial_layer_names = p.spatial_regressors_df['spatial_regressor_name'].values
            p.spatial_layer_paths = p.spatial_regressors_df['data_location'].values
            p.spatial_layer_types = p.spatial_regressors_df['type'].values


            spatial_layer_types_to_codes = {'multiplicative': 1,
                                            'additive': 2,
                                            }
            # QUIRCK, adjacency is really just additive with preprocessing.
            # LEARNING POINT, Here I could have used the set-concatenate function of |=
            spatial_layer_types_to_codes.update({'gaussian_' + str(sigma): 2 for sigma in p.kernel_halflives})



            spatial_layer_function_types_1d = np.asarray([spatial_layer_types_to_codes[i] for i in p.spatial_layer_types], np.int64)

            # p.spatial_layer_chunk_paths = []
            # for c, path in enumerate(p.spatial_layer_paths):
            #     if p.spatial_regressors_df['type'].values[c] == 'gaussian_parametric_1':
            #         _, class_id, _, sigma = p.spatial_regressors_df['spatial_regressor_name'].values[c].split('_')
            #         filename = os.path.split(path)[1]
            #         spatial_chunk_path = os.path.join(p.cur_dir, os.path.split(p.spatial_regressors_df['data_location'].values[c])[1])
            #         # spatial_chunk_path = os.path.join(p.cur_dir, p.spatial_regressors_df['spatial_regressor_name'].values[c] + '.tif')
            #         if not os.path.exists(spatial_chunk_path):
            #             # hb.load_geotiff_chunk_by_cr_size(path, p.fine_blocks_list)
            #             hb.load_geotiff_chunk_by_cr_size(path, p.fine_blocks_list, output_path=spatial_chunk_path)
            #         p.spatial_layer_chunk_paths.append(spatial_chunk_path)
            #
            # spatial_layer_chunk_counter = 0
            # for c, class_label in enumerate(p.spatial_regressors_df['spatial_regressor_name'].values):
            #     if p.spatial_regressors_df['type'].values[c] == 'gaussian_parametric_1':
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
            for spatial_layer_id, spatial_layer_label in enumerate(p.spatial_layer_names):
                for change_class_id, change_class_label in enumerate(p.change_class_labels):
                    if p.spatial_layer_types[spatial_layer_id] == 'additive' or p.spatial_layer_types[spatial_layer_id][0:8] == 'gaussian':
                        # QUIRK, notice that we copy the generation starting parameters EACH TIME we update a parameter. This is so that we have a fresh, unmodified set for the single thing we change.
                        # Increment it down
                        p.generation_parameters[generation_id][try_id] = np.copy(p.generation_parameters[generation_id][0])
                        new_coefficient = p.generation_parameters[generation_id][0][change_class_id, spatial_layer_id] - additive_coefficients_modulo
                        p.generation_parameters[generation_id][try_id][change_class_id, spatial_layer_id] = new_coefficient
                        p.generation_parameter_notations[generation_id][try_id] = {'new_coefficient': new_coefficient, 'old_coefficient': new_coefficient + additive_coefficients_modulo, 'spatial_layer_label': spatial_layer_label, 'spatial_layer_id': spatial_layer_id, 'change_class_id': change_class_id, 'change_class_label': change_class_label, 'spatial_layer_type': p.spatial_layer_types[spatial_layer_id]}

                        try_id += 1

                        # Increment it up
                        p.generation_parameters[generation_id][try_id] = np.copy(p.generation_parameters[generation_id][0])
                        new_coefficient = p.generation_parameters[generation_id][0][change_class_id, spatial_layer_id] + additive_coefficients_modulo
                        p.generation_parameters[generation_id][try_id][change_class_id, spatial_layer_id] = new_coefficient

                        p.generation_parameter_notations[generation_id][try_id] = {'new_coefficient': new_coefficient, 'old_coefficient': new_coefficient - additive_coefficients_modulo, 'spatial_layer_label': spatial_layer_label, 'spatial_layer_id': spatial_layer_id, 'change_class_id': change_class_id, 'change_class_label': change_class_label, 'spatial_layer_type': p.spatial_layer_types[spatial_layer_id]}

                        try_id += 1
                    elif p.spatial_layer_types[spatial_layer_id] == 'multiplicative':
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
            p.L.debug('Starting to run allocation iteratively for individual parameter changes. ')
            for k, spatial_layer_coefficients_2d in p.generation_parameters[generation_id].items():
                p.call_string = str(k) + '_' + str(generation_id)

                p.L.debug('coarse_change_matrix_4d', coarse_change_matrix_4d.shape, coarse_change_matrix_4d.dtype)
                p.L.debug('lulc_baseline_array', lulc_baseline_array.shape, lulc_baseline_array.dtype)
                p.L.debug('spatial_layers_3d', spatial_layers_3d.shape, spatial_layers_3d.dtype)
                p.L.debug('spatial_layer_coefficients_2d', spatial_layer_coefficients_2d.shape, spatial_layer_coefficients_2d.dtype)
                p.L.debug('spatial_layer_function_types_1d', spatial_layer_function_types_1d.shape, spatial_layer_function_types_1d.dtype)
                p.L.debug('valid_mask_array', valid_mask_array.shape, valid_mask_array.dtype)
                p.L.debug('change_class_labels', p.change_class_labels.shape, p.change_class_labels.dtype)
                p.L.debug('observed_lulc_array', observed_lulc_array.shape, observed_lulc_array.dtype)
                p.L.debug('hectares_per_grid_cell', hectares_per_grid_cell.shape, hectares_per_grid_cell.dtype)
                p.L.debug('cur_dir', p.cur_dir)
                p.L.debug('calibration_reporting_level', p.calibration_reporting_level)
                p.L.debug('call_string', p.call_string)

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

                p.L.debug('  Sweep found score ' + str(weighted_score) + ' for ' + str(p.generation_parameter_notations[generation_id][k]['spatial_layer_label']) +
                       ' with coeff ' + str(p.generation_parameter_notations[generation_id][k]['new_coefficient']) + ' on class '
                       + str(p.generation_parameter_notations[generation_id][k]['change_class_label']) + ' for try ' + str(k) + ' on generation ' + str(generation_id))
                try_scores[k] = weighted_score

            # Iterate through all score-improving changes from best to worst, seeing if they further improve the score in conjunction.
            p.L.debug('Iterating through all scores based on intial sweep value, testing to see if they make improvements in combination.')
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

                p.L.debug('  Score iterate found score ' + str(weighted_score) + ' for ' + str(p.generation_parameter_notations[generation_id][k]['spatial_layer_label']) +
                       ' with coeff ' + str(p.generation_parameter_notations[generation_id][k]['new_coefficient']) + ' on class '
                       + str(p.generation_parameter_notations[generation_id][k]['change_class_label']) + ' for try ' + str(k) + ' on generation ' + str(generation_id))

                if weighted_score > best_score:
                    kept_spatial_layer_coefficients_2d = copy.deepcopy(current_spatial_layer_coefficients_2d)

                    best_score = weighted_score

                    p.L.debug('    Score improved by adding in ' + str(p.generation_parameter_notations[generation_id][k]['spatial_layer_label']) +
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
                            p.L.debug('      Found improvement in permutations by further scaling ' +
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
                            p.L.debug('      SECOND PASS Found improvement in permutations by further scaling ' +
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


            output_df_2 = copy.deepcopy(p.spatial_regressors_df)
            for c, class_name in enumerate(p.seals_class_names):
                output_df_2[class_name] = generation_best_parameters[c, 0:]

            output_df_2.to_excel(os.path.join(p.cur_dir, 'trained_coefficients_gen' + str(generation_id) + '.xlsx'), index=False)

        # Write final coefficients
        output_df_2.to_excel(final_coefficients_path, index=False)

        # Write final lulc
        p.lulc_projected_path = os.path.join(p.cur_dir, 'lulc_projected.tif')
        hb.save_array_as_geotiff(lulc_projected_array, p.lulc_projected_path, p.lulc_baseline_path, ndv=-9999., data_type=1)

        # p.generation_best_parameters = copy.deepcopy(output_df_2)


        # # Otherwise load them

        # SHOULD I HAVE the non-calibrate assume it will find coeffs PER ZONE?
        # else:
        #     # spatial_regressor_trained_coefficients_read = pd.read_excel(p.pretrained_coefficients_path, index_col=0)
        #     current_pretrained_coefficients_path = p.pretrained_coefficients_path_dict[p.current_luh_scenario_label][p.current_year][p.current_policy_scenario_label]
        #
        #     # current_pretrained_coefficients_path = p.pretrained_coefficients_path_dict[p.current_policy_scenario_label]
        #     # current_pretrained_coefficients_path = hb.get_existing_path_from_nested_sources(p.pretrained_coefficients_path_dict[p.current_policy_scenario_label], p, verbose=True)
        #
        #     spatial_regressor_trained_coefficients_read = pd.read_excel(current_pretrained_coefficients_path, index_col=0)
        #     spatial_regressor_trained_coefficients = spatial_regressor_trained_coefficients_read[p.seals_class_names].values.astype(np.float64).T
        #     generation_best_parameters = np.copy(spatial_regressor_trained_coefficients)
        #     p.call_string = ''
    else:
        pass
        # NOTE, this doesnt persist because it is an iterated child.
        # p.generation_best_parameters = pd.read_excel(final_coefficients_path)
        # output_df_2.to_excel(final_coefficients_path, index=False)
def calibration_plots(passed_p=None):
    if passed_p is None:
        global p
    else:
        p = passed_p


    for c, class_label in enumerate(p.class_labels):
        baseline_array = hb.as_array(os.path.join(p.cur_dir, '../calibration_prepare_lulc', 'lulc_esa_seals7_' + str(p.training_start_year) + '.tif'))
        observed_array = hb.as_array(os.path.join(p.cur_dir, '../calibration_prepare_lulc', 'lulc_esa_seals7_' + str(p.base_year) + '.tif'))
        # os.path.join(p.cur_dir, '../calibration_zones', 'lulc_seals7_projected.tif')

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
            seals_utils.show_lulc_class_change_difference(baseline_array, observed_array, projected_array, lulc_class, similarity_array, change_array, annotation_text, output_path)


def esa_luh_baseline_lulc_adjustment(p):
    """Magpie2015 and ESA2015 are not the same. Allocate the changes of magpie2015 - esa2015coarse onto esa2015fine."""

    p.iterator_replacements = collections.OrderedDict()
    p.iterator_replacements['current_luh_scenario_label'] = []
    p.iterator_replacements['current_year'] = []
    p.iterator_replacements['current_policy_scenario_label'] = []
    p.iterator_replacements['cur_dir_parent_dir'] = []


    if p.run_this:
        if p.is_magpie_run:
            for baseline_label in p.baseline_labels:
                for year in p.base_years:
                    scenario_string = baseline_label + '_' + str(year)
                    p.iterator_replacements['current_luh_scenario_label'].append(baseline_label)
                    p.iterator_replacements['current_year'].append(year)
                    p.iterator_replacements['current_policy_scenario_label'].append(baseline_label)
                    p.iterator_replacements['cur_dir_parent_dir'].append(os.path.join(p.cur_dir, baseline_label, str(year)))

def zones_adjusted(p):
    """WTF IS THIS UNUSED?!? """
    fine_blocks_list = hb.get_global_block_list_from_resolution(p.processing_block_size, p.fine_resolution)

    p.L.info('Defining zones based on block size and resolutions for coarse_blocks_list:', p.processing_block_size, p.coarse_resolution)

    p.L.info('Length of iterator before pruning in task zones_adjusted:', len(fine_blocks_list))
    coarse_blocks_list = hb.get_global_block_list_from_resolution(p.processing_block_size, p.coarse_resolution)

    if p.subset_of_blocks_to_run is not None:
        old_fine_blocks_list = fine_blocks_list
        old_coarse_blocks_list = coarse_blocks_list

        fine_blocks_list = []
        coarse_blocks_list = []

        for i in p.subset_of_blocks_to_run:
            fine_blocks_list.append(old_fine_blocks_list[i])
            coarse_blocks_list.append(old_coarse_blocks_list[i])

    # Need to set projected_coarse_change_dir differently for magpie vs gtap runs because they have different tasks that generate their coarse change files.
    if p.is_magpie_run:
        p.projected_coarse_change_dir = os.path.join(p.convert_magpie_style_coarse_totals_to_seals7_dir, p.current_luh_scenario_label, str(p.current_year))
    elif p.is_gtap1_run:
        p.projected_coarse_change_dir = os.path.join(p.gtap_results_joined_with_luh_change_dir, p.current_luh_scenario_label, str(p.current_year), p.current_policy_scenario_label)
    elif p.is_calibration_run:
        p.projected_coarse_change_dir = os.path.join(p.seals7_difference_from_base_year_dir, p.current_luh_scenario_label, str(p.current_year), 'difference_2050_2015', 'esa_seals_simplified', 'rcp45_ssp2')
        # p.projected_coarse_change_dir = p.input_dir
    elif p.is_standard_seals_run:
        p.projected_coarse_change_dir = p.input_dir
    else:
        raise NameError('Unhandled.')

    p.L.info('Length of iterator after considering subset_of_blocks_to_run:', len(fine_blocks_list))
    # Pare down the number of blocks to run based on if there is change in the projected_coarse_change
    old_fine_blocks_list = fine_blocks_list
    old_coarse_blocks_list = coarse_blocks_list
    old_global_fine_blocks_list = global_fine_blocks_list
    old_global_coarse_blocks_list = global_coarse_blocks_list
    fine_blocks_list = []
    coarse_blocks_list = []
    global_fine_blocks_list = []
    global_coarse_blocks_list = []
    # if current_bb == hb.global_bounding_box:
    #     correct_fine_block_list = p.global_fine_blocks_list
    #     correct_coarse_block_list = p.global_coarse_blocks_list
    # else:
    #     correct_fine_block_list = p.fine_blocks_list
    #     correct_coarse_block_list = p.coarse_blocks_list
    for c, block in enumerate(old_coarse_blocks_list):
        skip = []
        for path in hb.list_filtered_paths_nonrecursively(p.projected_coarse_change_dir, include_extensions='.tif'):
            a = hb.load_geotiff_chunk_by_cr_size(path, block)
            changed = np.where((a != 0) & (a != -9999.) & (~np.isnan(a)), 1, 0)
            # hb.show(a)
            # hb.show(changed)
            p.L.debug('Checking to see if there is change in ', path)
            if np.nansum(changed) == 0:
                p.L.debug('Skipping because no change in coarse projections:', path)
                skip.append(True)
            else:
                skip.append(False)

        if not all(skip):
            fine_blocks_list.append(old_fine_blocks_list[c])
            coarse_blocks_list.append(old_coarse_blocks_list[c])
            global_fine_blocks_list.append(old_global_fine_blocks_list[c])
            global_coarse_blocks_list.append(old_global_coarse_blocks_list[c])

    p.L.info('Length of iterator after removing non-changing zones:', len(fine_blocks_list))

    p.iterator_replacements = collections.OrderedDict()
    p.iterator_replacements['fine_blocks_list'] = fine_blocks_list
    p.iterator_replacements['coarse_blocks_list'] = coarse_blocks_list
    p.iterator_replacements['global_fine_blocks_list'] = global_fine_blocks_list
    p.iterator_replacements['global_coarse_blocks_list'] = global_coarse_blocks_list
    # Trickier replacement that will redefine the parent dir for each task so that it also WRITES in the correct output location
    p.iterator_replacements['cur_dir_parent_dir'] = [p.cur_dir_parent_dir + '/zones_adjusted/' + str(i[4]) + '_' + str(i[5]) for i in fine_blocks_list]
    # p.iterator_replacements['cur_dir_parent_dir'] = [p.intermediate_dir + '/policy_scenario_allocations/' + p.current_policy_scenario_label + '/zones/' + str(i[4]) + '_' + str(i[5]) for i in fine_blocks_list]



def scenarios(p):
    """Create task to group downscaling of different scenarios."""

    # START HERE ValueError: Input Rasters are not the same dimensions. The following raster are not identical {(129600, 64800), (259070, 129535)} from map to esa
    # Calculated bb of tiles to be [-180.0, -55.99999999998508, 125.99999999996737, 84.0]
    # Stitching 12500 layers, first 1 of which was: ['..\\..\\projects\\seals_jaj_workstation2022\\intermediate\\scenarios\\rcp45_ssp2\\2050\\no_policy\\allocation_zones\\0_106\\allocation\\lulc_seals7_projected.tif']

    p.iterator_replacements = collections.OrderedDict()
    p.iterator_replacements['current_luh_scenario_label'] = []
    p.iterator_replacements['current_year'] = []
    p.iterator_replacements['current_policy_scenario_label'] = []
    p.iterator_replacements['cur_dir_parent_dir'] = []

    for luh_scenario_label in p.luh_scenario_labels:
        for year in p.scenario_years:
            for policy_scenario_label in p.policy_scenario_labels:
                scenario_string = luh_scenario_label + '_' + str(year) + '_' + policy_scenario_label
                p.iterator_replacements['current_luh_scenario_label'].append(luh_scenario_label)
                p.iterator_replacements['current_year'].append(year)
                p.iterator_replacements['current_policy_scenario_label'].append(policy_scenario_label)
                p.iterator_replacements['cur_dir_parent_dir'].append(os.path.join(p.cur_dir, luh_scenario_label, str(year), policy_scenario_label))



def allocation_zones(p):

    # Need to set projected_coarse_change_dir differently for magpie vs gtap runs because they have different tasks that generate their coarse change files.
    if p.is_magpie_run:
        p.projected_coarse_change_dir = os.path.join(p.convert_magpie_style_coarse_totals_to_seals7_dir, p.current_luh_scenario_label, str(p.current_year), p.current_policy_scenario_label)
    elif p.is_gtap1_run:
        p.projected_coarse_change_dir = os.path.join(p.gtap_results_joined_with_luh_change_dir, p.current_luh_scenario_label, str(p.current_year), p.current_policy_scenario_label)
    elif p.is_calibration_run:
        p.projected_coarse_change_dir = os.path.join(p.seals7_difference_from_base_year_dir, p.current_luh_scenario_label, str(p.current_year))
        # p.projected_coarse_change_dir = os.path.join(p.base_data_dir, 'luh2', 'processed_data', 'difference_2050_2015', 'esa_seals_simplified', p.current_luh_scenario_label)
        #C:\Files\Research\base_data\luh2\processed_data\difference_2050_2015\esa_seals_simplified\RCP45_SSP2
    else:
        raise NameError('Unhandled.')

    # Generate lists of which zones change and thus need to be rerun
    if p.combined_block_lists_paths is None:
        p.combined_block_lists_paths = {
            'fine_blocks_list': os.path.join(p.cur_dir, 'fine_blocks_list.csv'),
            'coarse_blocks_list': os.path.join(p.cur_dir, 'coarse_blocks_list.csv'),
            'global_fine_blocks_list': os.path.join(p.cur_dir, 'global_fine_blocks_list.csv'),
            'global_coarse_blocks_list': os.path.join(p.cur_dir, 'global_coarse_blocks_list.csv'),
            'global_processing_blocks_list': os.path.join(p.cur_dir, 'global_processing_blocks_list.csv'),
        }

    try:
        if all(hb.path_exists(i) for i in combined_block_lists_paths):
            blocks_lists_already_exist = True
        else:
            blocks_lists_already_exist = False
    except:
        blocks_lists_already_exist = False

    if blocks_lists_already_exist:
        fine_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'fine_blocks_list.csv'), '2d_list'))
        coarse_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'coarse_blocks_list.csv'), '2d_list'))
        global_fine_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'global_fine_blocks_list.csv'), '2d_list'))
        global_coarse_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'global_coarse_blocks_list.csv'), '2d_list'))
        global_processing_blocks_list = list(hb.file_to_python_object(os.path.join(p.cur_dir, 'global_processing_blocks_list.csv'), '2d_list'))

    else:
        if p.aoi == 'global':

            fine_blocks_list = hb.get_global_block_list_from_resolution(p.processing_block_size, p.fine_resolution)
            coarse_blocks_list = hb.get_global_block_list_from_resolution(p.processing_block_size, p.coarse_resolution)
            global_fine_blocks_list = fine_blocks_list
            global_coarse_blocks_list = coarse_blocks_list
            global_processing_blocks_list = hb.get_global_block_list_from_resolution_and_bb(p.processing_block_size, p.processing_block_size, p.bb)
        else:
            fine_blocks_list = hb.get_subglobal_block_list_from_resolution_and_bb(p.processing_block_size, p.fine_resolution, p.bb)
            coarse_blocks_list = hb.get_subglobal_block_list_from_resolution_and_bb(p.processing_block_size, p.coarse_resolution, p.bb)
            global_fine_blocks_list = hb.get_global_block_list_from_resolution_and_bb(p.processing_block_size, p.fine_resolution, p.bb)
            global_coarse_blocks_list = hb.get_global_block_list_from_resolution_and_bb(p.processing_block_size, p.coarse_resolution, p.bb)
            global_processing_blocks_list = hb.get_global_block_list_from_resolution_and_bb(p.processing_block_size, p.processing_block_size, p.bb)

        p.L.info('Length of iterator before pruning in task calibration:', len(fine_blocks_list))

        if p.subset_of_blocks_to_run is not None:
            old_fine_blocks_list = fine_blocks_list
            old_coarse_blocks_list = coarse_blocks_list

            fine_blocks_list = []
            coarse_blocks_list = []

            for i in p.subset_of_blocks_to_run:
                fine_blocks_list.append(old_fine_blocks_list[i])
                coarse_blocks_list.append(old_coarse_blocks_list[i])

    p.L.info('Length of iterator after considering subset_of_blocks_to_run:', len(fine_blocks_list))

    combined_block_lists_dict = {
        'fine_blocks_list': fine_blocks_list,
        'coarse_blocks_list': coarse_blocks_list,
        'global_fine_blocks_list': global_fine_blocks_list,
        'global_coarse_blocks_list': global_coarse_blocks_list,
        'global_processing_blocks_list': global_processing_blocks_list,
    }

    if not all([hb.path_exists(i) for i in p.combined_block_lists_paths.values()]):

        # Pare down the number of blocks to run based on if there is change in the projected_coarse_change
        old_fine_blocks_list = fine_blocks_list
        old_coarse_blocks_list = coarse_blocks_list
        old_global_fine_blocks_list = global_fine_blocks_list
        old_global_coarse_blocks_list = global_coarse_blocks_list
        old_global_processing_blocks_list = global_processing_blocks_list
        fine_blocks_list = []
        coarse_blocks_list = []
        global_fine_blocks_list = []
        global_coarse_blocks_list = []
        global_processing_blocks_list = []

        L.info('Checking existing blocks for change in the LUH data and excluding if no change.')
        for c, block in enumerate(old_coarse_blocks_list):
            progress_percent = float(c) / float(len(old_coarse_blocks_list)) * 100.0
            print('Percent finished: ' + str(progress_percent), end='\r', flush=False)
            skip = []
            # TODOO  Make this only look at the correct data. NEVERY USE FILE EXISTNEC for figuring out anything other than if it needs to be recreated, not how it should be accessed later).
            current_coarse_change_rasters = []
            for class_label in p.class_labels:
                gen_path = os.path.join(p.projected_coarse_change_dir, class_label + '_' + str(p.scenario_years[0]) + '_' + str(p.base_years[0]) + '_ha_difference.tif')
                current_coarse_change_rasters.append(gen_path)
            for path in current_coarse_change_rasters:
                # for path in hb.list_filtered_paths_nonrecursively(p.projected_coarse_change_dir, include_extensions='.tif'):
                block = old_coarse_blocks_list[c]
                a = hb.load_geotiff_chunk_by_cr_size(path, block)
                changed = np.where((a != 0) & (a != -9999.) & (~np.isnan(a)), 1, 0)
                # hb.show(a)
                # hb.show(changed)
                p.L.debug('Checking to see if there is change in ', path)
                if np.nansum(changed) == 0:
                    p.L.debug('Skipping because no change in coarse projections:', path)
                    skip.append(True)
                else:
                    skip.append(False)
                    skip.append(False)

            if not all(skip):
                fine_blocks_list.append(old_fine_blocks_list[c])
                coarse_blocks_list.append(old_coarse_blocks_list[c])
                global_fine_blocks_list.append(old_global_fine_blocks_list[c])
                global_coarse_blocks_list.append(old_global_coarse_blocks_list[c])
                global_processing_blocks_list.append(old_global_processing_blocks_list[c])

        # Write the blockslists to csvs to avoid future reprocessing (actually is quite slow (2 mins) when 64000 tiles).
        for block_name, block_list in combined_block_lists_dict.items():
            hb.python_object_to_csv(block_list, os.path.join(p.cur_dir, block_name + '.csv'), '2d_list')

    else:
        # NOTE: This could be generalized so that it just uses the project level variable for block list paths: p.combined_block_lists_paths
        fine_blocks_list = list(hb.file_to_python_object(os.path.join(p.calibration_dir, 'fine_blocks_list.csv'), '2d_list'))
        coarse_blocks_list = list(hb.file_to_python_object(os.path.join(p.calibration_dir, 'coarse_blocks_list.csv'), '2d_list'))
        global_fine_blocks_list = list(hb.file_to_python_object(os.path.join(p.calibration_dir, 'global_fine_blocks_list.csv'), '2d_list'))
        global_coarse_blocks_list = list(hb.file_to_python_object(os.path.join(p.calibration_dir, 'global_coarse_blocks_list.csv'), '2d_list'))
        global_processing_blocks_list = list(hb.file_to_python_object(os.path.join(p.calibration_dir, 'global_processing_blocks_list.csv'), '2d_list'))

    p.L.info('Length of iterator after removing non-changing zones:', len(fine_blocks_list))

    # Remove from iterator lists that have already been computed.
    old_fine_blocks_list = fine_blocks_list
    old_coarse_blocks_list = coarse_blocks_list
    old_global_fine_blocks_list = global_fine_blocks_list
    old_global_coarse_blocks_list = global_coarse_blocks_list
    old_global_processing_blocks_list = global_processing_blocks_list
    fine_blocks_list = []
    coarse_blocks_list = []
    global_fine_blocks_list = []
    global_coarse_blocks_list = []
    global_processing_blocks_list = []

    for c, fine_block in enumerate(old_fine_blocks_list):
        tile_dir = str(fine_block[4]) + '_' + str(fine_block[5])
        expected_path = os.path.join(p.cur_dir, tile_dir, 'allocation', 'lulc_seals7_projected.tif')
        if not hb.path_exists(expected_path):
            fine_blocks_list.append(old_fine_blocks_list[c])
            coarse_blocks_list.append(old_coarse_blocks_list[c])
            global_fine_blocks_list.append(old_global_fine_blocks_list[c])
            global_coarse_blocks_list.append(old_global_coarse_blocks_list[c])
            global_processing_blocks_list.append(old_global_processing_blocks_list[c])

    p.L.info('Length of iterator after removing finished zones:', len(fine_blocks_list))

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

                # for k, v in zone_calibration_block_lookup_dict.items():
                #     print(k, v, zones_raster[v])

                with open(p.processing_zones_to_calibration_chunk_path, "w") as f:
                    for k, line in zone_calibration_block_lookup_dict.items():
                        # print(k, line)
                        f.write(str(k) + ',' + str(line[0]) + '_' + str(line[1]) + '\n')

    p.iterator_replacements = collections.OrderedDict()
    p.iterator_replacements['fine_blocks_list'] = fine_blocks_list
    p.iterator_replacements['coarse_blocks_list'] = coarse_blocks_list
    p.iterator_replacements['global_fine_blocks_list'] = global_fine_blocks_list
    p.iterator_replacements['global_coarse_blocks_list'] = global_coarse_blocks_list
    p.iterator_replacements['global_processing_blocks_list'] = global_processing_blocks_list

    # Trickier replacement that will redefine the parent dir for each task so that it also WRITES in the correct output location
    p.iterator_replacements['cur_dir_parent_dir'] = [p.cur_dir_parent_dir+ '/allocation_zones/' + str(i[4]) + '_' + str(i[5]) for i in fine_blocks_list]
    # p.iterator_replacements['cur_dir_parent_dir'] = [p.intermediate_dir + '/scenarios/' + p.current_policy_scenario_label + '/zones/' + str(i[4]) + '_' + str(i[5]) for i in fine_blocks_list]

def prepare_lulc(passed_p=None):
    if passed_p is None:
        global p
    else:
        p = passed_p

    p.zone_esa_seals7_lulc_base_year_path = os.path.join(p.cur_dir, 'zone_esa_seals7_lulc_' + str(p.base_year) + '.tif')

    p.chunk_coarse_ha_per_cell_path = os.path.join(p.cur_dir, 'chunk_coarse_ha_per_cell.tif')

    p.lulc_class_types_path = r"C:\OneDrive\Projects\cge\seals\projects\ipbes\input\lulc_class_types.csv"

    # Problem here: Change vector method needs to be expanded to Change matrix, full from-to relationships
    # but when doing from-to, that only works when doing observed time-period validation. What would be the assumption for going into
    # the future? Possibly attempt to match prior change matrices, but only as a slight increase in probability? Secondly, why is my
    # search algorithm not itself finding the from-to relationships just by minimizing difference? Basically, need to take seriously deallocation.

    if p.run_this:
        # Clip ha_per_cell and use it as the match
        chunk_coarse_ha_per_cell = hb.load_geotiff_chunk_by_cr_size(p.aoi_ha_per_cell_coarse_path, p.coarse_blocks_list, output_path=p.chunk_coarse_ha_per_cell_path)

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
                for year in p.base_years:
                    p.projected_coarse_change_files_adjustment_run[scenario_label][year] = {}
                    for policy_scenario_label in ['baseline']:
                        p.projected_coarse_change_files_adjustment_run[scenario_label][year][policy_scenario_label] = {}
                        for class_label in current_class_labels:
                            implied_magpie_path = os.path.join(p.projected_coarse_change_dir, magpie_short_label + '_' + class_label + '_' + str(p.base_year) + '_ha_difference_from_esa.tif')
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

                            implied_gtap1_path = os.path.join(p.projected_coarse_change_dir, 'gtap1_' + class_label + '_ha_change_15min.tif')

                            if hb.path_exists(implied_gtap1_path):
                                p.projected_coarse_change_files[luh_scenario_label][year][policy_scenario_label][class_label] = implied_gtap1_path
                            else:
                                implied_luh_path = os.path.join(p.seals7_difference_from_base_year_dir, luh_scenario_label, str(year),
                                                                class_label + '_' + str(year) + '_' + str(p.base_year) + '_ha_difference.tif')
                                p.projected_coarse_change_files[luh_scenario_label][year][policy_scenario_label][class_label] = implied_luh_path

                        elif p.is_magpie_run:
                            implied_magpie_path = os.path.join(p.projected_coarse_change_dir, magpie_short_label + '_' + class_label + '_' + str(year) + '_' + str(p.base_year) + '_ha_difference.tif')
                            p.projected_coarse_change_files[luh_scenario_label][year][policy_scenario_label][class_label] = implied_magpie_path
                        else:
                            implied_ssp_path = os.path.join(p.projected_coarse_change_dir, class_label + '_' + str(year) + '_' + str(p.base_year) + '_ha_difference.tif')
                            p.projected_coarse_change_files[luh_scenario_label][year][policy_scenario_label][class_label] = implied_ssp_path

        if 'esa_luh_baseline_lulc_adjustment' in p.cur_dir_parent_dir:
            p.projected_coarse_change_files = p.projected_coarse_change_files_adjustment_run
            p.is_first_pass = True
        else:
            p.is_first_pass = False

        p.L.info('prepare_lulc looked for projected_coarse_change_files and found ', p.projected_coarse_change_files)
        if p.write_projected_coarse_change_chunks:
            for luh_scenario_label, v in p.projected_coarse_change_files.items():
                for year_label, vv in v.items():
                    for policy_scenario_label, vvv in vv.items():
                        for class_label, coarse_change_path in vvv.items():
                            current_net_change_array_path = os.path.join(p.cur_dir, class_label + '_projected_change.tif')

                            hb.load_geotiff_chunk_by_cr_size(coarse_change_path, p.coarse_blocks_list, output_path=current_net_change_array_path)

        # TODOOO longer term: redo allocation routine with specific final change matrix

def allocation_change_matrix(passed_p=None):
    if passed_p is None:
        global p
    else:
        p = passed_p


    p.allocation_full_change_matrix_no_diagonal_path = os.path.join(p.cur_dir, 'allocation_full_change_matrix_no_diagonal.tif')
    p.allocation_full_change_matrix_path = os.path.join(p.cur_dir, 'allocation_full_change_matrix.tif')

    if p.run_this:

        # hb.clip_raster_by_vector(p.coarse_ha_per_cell_path, p.coarse_ha_per_cell_path, p.aoi_path, all_touched=True, ensure_fits=True)
        p.chunk_coarse_ha_per_cell = hb.ArrayFrame(p.chunk_coarse_ha_per_cell_path)
        p.chunk_coarse_match = hb.ArrayFrame(p.chunk_coarse_ha_per_cell_path)

        p.lulc_base_year_chunk_10sec_path = os.path.join(p.cur_dir, 'lulc_esa_seals7_' + str(p.base_year) + '.tif')
        p.lulc_training_start_year_chunk_10sec_path = os.path.join(p.cur_dir, 'lulc_esa_seals7_' + str(p.training_start_year) + '.tif')

        # Clip ha_per_cell and use it as the match

        hb.load_geotiff_chunk_by_cr_size(p.lulc_simplified_paths['lulc_esa_seals7_' + str(p.base_year)], p.fine_blocks_list, output_path=p.lulc_base_year_chunk_10sec_path)


        hb.load_geotiff_chunk_by_cr_size(p.lulc_simplified_paths['lulc_esa_seals7_' + str(p.training_start_year)], p.fine_blocks_list, output_path=p.lulc_training_start_year_chunk_10sec_path)
        p.lulc_training_start_year_chunk = hb.ArrayFrame(p.lulc_training_start_year_chunk_10sec_path)

        p.lulc_base_year_chunk = hb.ArrayFrame(p.lulc_base_year_chunk_10sec_path)
        p.fine_match = hb.ArrayFrame(p.lulc_base_year_chunk_10sec_path)

        p.coarse_ha_per_cell = hb.ArrayFrame(p.coarse_ha_per_cell_path)
        # p.coarse_match = hb.ArrayFrame(p.coarse_ha_per_cell_path)

        fine_cells_per_coarse_cell = round((p.chunk_coarse_ha_per_cell.cell_size / p.fine_match.cell_size) ** 2)
        aspect_ratio = int(p.fine_match.num_cols / p.chunk_coarse_match.num_cols)


        net_change_output_arrays = np.zeros((len(p.class_indices), p.chunk_coarse_match.shape[0], p.chunk_coarse_match.shape[1]))
        full_change_matrix = np.zeros((len(p.class_indices * p.chunk_coarse_match.n_rows), len(p.class_indices) * p.chunk_coarse_match.n_cols))
        full_change_matrix_no_diagonal = np.zeros((len(p.class_indices * p.chunk_coarse_match.n_rows), len(p.class_indices) * p.chunk_coarse_match.n_cols))
        for r in range(p.chunk_coarse_match.num_rows):
            for c in range(p.chunk_coarse_match.num_cols):

                t1_subarray = p.lulc_training_start_year_chunk.data[int(r * aspect_ratio): int((r + 1) * aspect_ratio), int(c * aspect_ratio): int((c + 1) * aspect_ratio)]
                t2_subarray = p.lulc_base_year_chunk.data[int(r * aspect_ratio): int((r + 1) * aspect_ratio), int(c * aspect_ratio): int((c + 1) * aspect_ratio)]
                # ha_per_cell_subarray = chunk_coarse_ha_per_cell.data[int(r * aspect_ratio): int((r + 1) * aspect_ratio), int(c * aspect_ratio): int((c + 1) * aspect_ratio)]

                ha_per_coarse_cell_this_subarray = p.chunk_coarse_ha_per_cell.data[r, c]

                change_matrix, counters = hb.calc_change_matrix_of_two_int_arrays(t1_subarray.astype(np.int), t2_subarray.astype(np.int), p.class_indices)

                vector = seals_utils.calc_change_vector_of_change_matrix(change_matrix)

                verbose = False
                if verbose:
                    p.L.info('change_matrix in coarse chunk ' + str(r) + ', ' + str(c) + '\n' + str(change_matrix) + '\nwith change vector ' + str(vector))


                ha_per_cell_this_subarray = p.chunk_coarse_ha_per_cell.data[r, c] / fine_cells_per_coarse_cell

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

            hb.save_array_as_geotiff(full_change_matrix, p.allocation_full_change_matrix_path, p.chunk_coarse_match.path, n_rows=full_change_matrix.shape[1], n_cols=full_change_matrix.shape[1])
            hb.save_array_as_geotiff(full_change_matrix_no_diagonal, p.allocation_full_change_matrix_no_diagonal_path, p.chunk_coarse_match.path, n_rows=full_change_matrix_no_diagonal.shape[1], n_cols=full_change_matrix_no_diagonal.shape[1])

        p.plot_change_matrices = 0
        if p.plot_change_matrices:
            from matplotlib import colors as colors
            fig, ax = plt.subplots()
            fig.set_size_inches(10, 8)

            # Plot the heatmap
            vmin = np.min(full_change_matrix_no_diagonal)
            vmax = np.max(full_change_matrix_no_diagonal)
            im = ax.imshow(full_change_matrix_no_diagonal, cmap='YlGnBu', norm=colors.LogNorm(vmin=vmin + 1, vmax=vmax))

            # Create colorbar
            import matplotlib.ticker as ticker

            cbar = ax.figure.colorbar(im, ax=ax, format=ticker.FuncFormatter(lambda x, p : int(x)))
            cbar.set_label('Number of cells changed from class ROW to class COL', size=10)

            # Set ticks...
            ax.set_xticks(np.arange(full_change_matrix_no_diagonal.shape[1]))
            ax.set_yticks(np.arange(full_change_matrix_no_diagonal.shape[0]))

            # Create labels for each coarse zone indexed by i and j
            row_labels = []
            col_labels = []
            for i in range(n_classes * p.chunk_coarse_match.n_rows):
                class_id = i % n_classes
                coarse_grid_cell_counter = int(i / n_classes)
                row_labels.append(str(class_id))
                col_labels.append(str(class_id))

            trans = ax.get_xaxis_transform()  # x in data untis, y in axes fraction

            for i in range(p.chunk_coarse_match.n_rows):
                ann = ax.annotate('Zone i=' + str(i + 1), xy=(-3.5, (p.chunk_coarse_match.n_rows - i) / p.chunk_coarse_match.n_rows - .5 / p.chunk_coarse_match.n_rows), xycoords=trans)
                ann = ax.annotate('Zone j=' + str(i + 1), xy=(i * (p.chunk_coarse_match.n_rows + 1) + .25 * p.chunk_coarse_match.n_rows, 1.05), xycoords=trans)  #

            ax.set_xticklabels(col_labels)
            ax.set_yticklabels(row_labels)

            # Let the horizontal axes labeling appear on top.
            ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

            plt.setp(ax.get_xticklabels(), ha="center", rotation_mode="anchor")

            # Turn spines off and create white grid.
            for edge, spine in ax.spines.items():
                spine.set_visible(False)

            ax.set_xticks(np.arange(full_change_matrix_no_diagonal.shape[1] + 1) - .5, minor=True)
            ax.set_yticks(np.arange(full_change_matrix_no_diagonal.shape[0] + 1) - .5, minor=True)
            ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
            ax.tick_params(which="minor", bottom=False, left=False)

            full_change_matrix_no_diagonal_png_path = os.path.join(p.cur_dir, 'fcmnd.png')
            # texts = annotate_heatmap(im, valfmt="{x:.1f} t")

            major_gridline = False
            for i in range(n_classes * p.chunk_coarse_match.n_rows + 1):
                try:
                    if i % n_classes == 0:
                        major_gridline = i
                    else:
                        major_gridline = False
                except:
                    major_gridline = 0

                if major_gridline is not False:
                    xloc = major_gridline - .5
                    yloc = major_gridline - .5
                    ax.axvline(x=xloc, color='grey')
                    ax.axhline(y=yloc, color='grey')

            plt.savefig(full_change_matrix_no_diagonal_png_path)

            vmax = np.max(full_change_matrix_no_diagonal)
            # full_change_matrix_no_diagonal_png_path = os.path.join(p.cur_dir, 'full_change_matrix_no_diagona_auto.png')

            # Not really necessary but decent exampe of auto plot.
            # hb.full_show_array(full_change_matrix_no_diagonal, output_path=full_change_matrix_no_diagonal_png_path, cbar_label='Number of changes from class R to class C per tile', title='Change matrix mosaic',
            #                    num_cbar_ticks=2, vmin=0, vmid=vmax / 10.0, vmax=vmax, color_scheme='ylgnbu')



def allocation(passed_p=None):
    if passed_p is None:
        global p
    else:
        p = passed_p

    start = time.time()

    # Set where CHUNK-specific maps will be saved.
    p.lulc_baseline_path = os.path.join(p.cur_dir, 'lulc_baseline.tif')
    p.zone_esa_seals7_lulc_base_year_path = os.path.join(p.cur_dir, 'zone_esa_seals7_lulc_base_year.tif')
    p.fine_match_path = p.lulc_baseline_path

    p.lulc_ndv = hb.get_ndv_from_path(p.base_year_lulc_path)


    p.loss_function_sigma = np.float64(7.0) # Set how much closeness vs farness matters in assessing accuracy. Sigma = 1 means you need to be REALLY close to count as a food prediction.



    # Load the coefficients as DF from either calibration dir or a prebuilt dir.
    if p.use_calibration_created_coefficients:
        if p.use_calibration_from_zone_centroid_tile:
            # START HERE: calibration_zone_polygons_path points to the correct GTAP37_AEZ18 gpkg. Rasterize it to create coarse-tile to AZREG correspondence then calculate centroids.
            p.calibration_zone_polygons_path
            zone_string = os.path.split(p.cur_dir_parent_dir)[1]
            current_pretrained_coefficients_path = os.path.join(p.calibration_dir, zone_string, 'calibration_zones', 'trained_coefficients_zone_' + zone_string + '.xlsx')
            p.L.info('Setting current_pretrained_coefficients_path to one generated for zone ' + str(zone_string)  + ' at  ' + current_pretrained_coefficients_path)
        else:

            zone_string = os.path.split(p.cur_dir_parent_dir)[1]
            current_pretrained_coefficients_path = os.path.join(p.calibration_dir, zone_string, 'calibration_zones', 'trained_coefficients_zone_' + zone_string + '.xlsx')
            p.L.info('Setting current_pretrained_coefficients_path to one generated in this project, at ' + current_pretrained_coefficients_path)
    else:
        current_pretrained_coefficients_path = p.pretrained_coefficients_path_dict[p.current_luh_scenario_label][p.current_year][p.current_policy_scenario_label]
        p.L.info('Setting current_pretrained_coefficients_path to one specified in run configuration, at ' + current_pretrained_coefficients_path)

    p.spatial_regressors_df = pd.read_excel(current_pretrained_coefficients_path)

    p.change_class_labels = np.asarray(p.class_indices, dtype=np.int64)  # For Cythonization, load these as the "labels", which is used for writing.

    lulc_projected_path = os.path.join(p.cur_dir, 'lulc_seals7_projected.tif')

    if p.skip_created_downscaling_zones:
        skip_this_zone = os.path.exists(lulc_projected_path)
    else:
        skip_this_zone = False

    p.projected_current_coarse_change_input_paths = []
    for class_label in p.class_labels:
        p.projected_current_coarse_change_input_paths.append(os.path.join(p.prepare_lulc_dir, class_label + '_projected_change.tif'))

    if p.run_this and not skip_this_zone:
        lulc_baseline_array = hb.load_geotiff_chunk_by_cr_size(p.lulc_simplified_paths['lulc_esa_seals7_' + str(p.base_year)], p.fine_blocks_list, datatype=np.int64, output_path=p.lulc_baseline_path).astype(np.int64)

        p.spatial_layer_names = p.spatial_regressors_df['spatial_regressor_name'].values
        p.spatial_layer_paths = p.spatial_regressors_df['data_location'].values
        p.spatial_layer_types = p.spatial_regressors_df['type'].values

        # QUIRCK, adjacency is really just additive with preprocessing.
        spatial_layer_types_to_codes = {'multiplicative': 1,
                                        'additive': 2,
                                        }
        spatial_layer_types_to_codes.update({'gaussian_' + str(sigma): 2 for sigma in p.kernel_halflives})

        spatial_layer_function_types_1d = np.asarray([spatial_layer_types_to_codes[i] for i in p.spatial_layer_types], np.int64)

        # # CREATE GAUSSIANS for all variables tagged as that type. Note this is a large performance loss and I use precached global convolutions in all cases to date.
        # p.spatial_layer_chunk_paths = []
        # for c, path in enumerate(p.spatial_layer_paths):
        #     if p.spatial_regressors_df['type'].values[c][0:8] == 'gaussian':
        #         _, class_id, _, sigma = p.spatial_regressors_df['spatial_regressor_name'].values[c].split('_')
        #         filename = os.path.split(path)[1]
        #         spatial_chunk_path = os.path.join(p.cur_dir, os.path.split(p.spatial_regressors_df['data_location'].values[c])[1])
        #         if not os.path.exists(spatial_chunk_path):
        #             hb.load_geotiff_chunk_by_cr_size(path, p.fine_blocks_list, output_path=spatial_chunk_path)
        #         p.spatial_layer_chunk_paths.append(spatial_chunk_path)
        #
        # spatial_layer_chunk_counter = 0
        # for c, class_label in enumerate(p.spatial_regressors_df['spatial_regressor_name'].values):
        #     if p.spatial_regressors_df['type'].values[c][0:8] == 'gaussian':
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
        hectares_per_grid_cell = hb.load_geotiff_chunk_by_cr_size(p.aoi_ha_per_cell_fine_path, p.fine_blocks_list).astype(np.float64)

        # START HERE: Although the final allocation does properly start from the 2015 lulc, the underlying ranking is still based on the 2000 class binaries.
        # Figure out a way to make the spatial_layers_3d smartly update the "new state" variables in a way that is forward looking for year-by-year iteration

        # Build the numpy array for spatial layers.
        spatial_layers_3d = np.zeros((len(p.spatial_layer_paths), n_r, n_c)).astype(np.float64)

        # Chose not to normalize anything.
        normalize_inputs = False

        # Add either the normalized or not normalized array to the spatial_layers_3d
        for c, path in enumerate(p.spatial_layer_paths):
            p.L.debug('Loading spatial layer at path ' + path)
            # TODOO p.L.critical('This replacement needs to be fixed.')
            # path = path.replace('combined_policies', '\worldbank_rtk_feedback_model')
            current_bb = hb.get_bounding_box(path)

            if current_bb == hb.global_bounding_box:
                correct_fine_block_list = p.global_fine_blocks_list
                correct_coarse_block_list = p.global_coarse_blocks_list
            else:
                correct_fine_block_list = p.fine_blocks_list
                correct_coarse_block_list = p.coarse_blocks_list

            if p.spatial_layer_types[c] == 'additive' or p.spatial_layer_types[c] == 'multiplicative':
                if normalize_inputs is True:
                    spatial_layers_3d[c] = hb.normalize_array(hb.load_geotiff_chunk_by_cr_size(path, correct_fine_block_list))
                else:
                    spatial_layers_3d[c] = hb.load_geotiff_chunk_by_cr_size(path, correct_fine_block_list)
            elif p.spatial_layer_types[c][0:8] == 'gaussian':
                # updated_path = os.path.join(p.cur_dir, 'class_' + p.spatial_layer_names[c].split('_')[1] + '_gaussian_' + p.spatial_layer_names[c].split('_')[3] + '_convolution.tif')

                if normalize_inputs is True:
                    spatial_layers_3d[c] = hb.normalize_array(hb.as_array(updated_path))
                else:
                    spatial_layers_3d[c] = hb.load_geotiff_chunk_by_cr_size(path, correct_fine_block_list)
                    # spatial_layers_3d[c] = hb.as_array(updated_path) # NOTE assumes already clipped
            else:
                raise NameError('unspecified type')

        # Load baseline lulc and valid mask
        if p.is_gtap1_run:
            observed_lulc_array = hb.load_geotiff_chunk_by_cr_size(p.lulc_simplified_paths['lulc_esa_seals7_' + str(p.training_start_year)], p.fine_blocks_list, datatype=np.int64, output_path=None).astype(np.int64)
        elif p.is_magpie_run:
            if not p.is_first_pass:

                # HACK. Until I create a full dynamic run framework for seals, these types of errors are going to keep arising.
                if p.adjust_baseline_to_match_magpie_2015:
                    starting_lulc_path = os.path.join(p.cur_dir.replace('rcp45_ssp2\\2050', 'baseline\\2015').replace('zones', 'zones_adjusted').replace('scenarios', 'esa_luh_baseline_lulc_adjustment'), 'lulc_projected.tif')
                    for i in p.magpie_policy_scenario_labels:
                        starting_lulc_path = starting_lulc_path.replace('\\' + i, '')
                else:
                    starting_lulc_path = p.base_year_simplified_lulc_path

                observed_lulc_array = hb.load_geotiff_chunk_by_cr_size(starting_lulc_path, p.fine_blocks_list, datatype=np.int64, output_path=None).astype(np.int64)
                lulc_baseline_array = hb.load_geotiff_chunk_by_cr_size(p.base_year_simplified_lulc_path, p.fine_blocks_list, datatype=np.int64, output_path=p.lulc_baseline_path).astype(np.int64)

            else:
                starting_lulc_path = p.base_year_simplified_lulc_path
                lulc_baseline_array = hb.load_geotiff_chunk_by_cr_size(p.base_year_simplified_lulc_path, p.fine_blocks_list, datatype=np.int64, output_path=p.lulc_baseline_path).astype(np.int64)

                observed_lulc_array = hb.load_geotiff_chunk_by_cr_size(starting_lulc_path, p.fine_blocks_list, datatype=np.int64, output_path=None).astype(np.int64)
        elif p.is_calibration_run:
            observed_lulc_array = hb.load_geotiff_chunk_by_cr_size(p.lulc_simplified_paths['lulc_esa_seals7_' + str(p.training_start_year)], p.fine_blocks_list, datatype=np.int64, output_path=None).astype(np.int64)
        valid_mask_array = np.where((observed_lulc_array != p.lulc_ndv), 1, 0).astype(np.int64)

        # Set how much change for each class needs to be allocated.
        projected_coarse_change_3d = np.zeros((len(p.class_labels), coarse_n_r, coarse_n_c)).astype(np.float64)
        for c, path in enumerate(list(p.projected_coarse_change_files[p.current_luh_scenario_label][p.current_year][p.current_policy_scenario_label].values())):

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
                projected_coarse_change_3d[c] = hb.load_geotiff_chunk_by_cr_size(path, p.coarse_blocks_list).astype(np.float64)

        p.seals_class_names = p.spatial_regressors_df.columns.values[3:]
        spatial_regressor_trained_coefficients = p.spatial_regressors_df[p.seals_class_names].values.astype(np.float64).T
        generation_best_parameters = np.copy(spatial_regressor_trained_coefficients)

        p.call_string = ''

        # L.setLevel(logging.DEBUG)
        lulc_baseline_array = lulc_baseline_array.astype(np.int64)
        p.L.debug('projected_coarse_change_3d', type(projected_coarse_change_3d), projected_coarse_change_3d.dtype)
        p.L.debug('lulc_baseline_array', type(lulc_baseline_array), lulc_baseline_array.dtype, lulc_baseline_array)
        p.L.debug('spatial_layers_3d', type(spatial_layers_3d), spatial_layers_3d.dtype, spatial_layers_3d)
        p.L.debug('generation_best_parameters', type(generation_best_parameters), generation_best_parameters.dtype, generation_best_parameters)
        p.L.debug('spatial_layer_function_types_1d', type(spatial_layer_function_types_1d), spatial_layer_function_types_1d.dtype, spatial_layer_function_types_1d)
        p.L.debug('valid_mask_array', type(valid_mask_array), valid_mask_array.dtype, valid_mask_array)
        p.L.debug('p.change_class_labels', type(p.change_class_labels), p.change_class_labels.dtype, p.change_class_labels)
        p.L.debug('observed_lulc_array', type(observed_lulc_array), observed_lulc_array.dtype, observed_lulc_array)
        p.L.debug('hectares_per_grid_cell', type(hectares_per_grid_cell), hectares_per_grid_cell.dtype, hectares_per_grid_cell)
        p.L.debug('p.cur_dir', type(p.cur_dir), p.cur_dir)
        p.L.debug('p.loss_function_sigma', type(p.loss_function_sigma), p.loss_function_sigma)
        p.L.debug('p.call_string', type(p.call_string), p.call_string)

        # Strange choice, but the allocation function both calibrates AND RUNS the final projection using the calibration. If instead you want to
        # Run on precalibrated parameters, you have to reload coarse_change_3d.
        overall_similarity_score, lulc_projected_array, overall_similarity_plot, class_similarity_scores, class_similarity_plots = \
            calibrate(projected_coarse_change_3d,
                      lulc_baseline_array,
                      spatial_layers_3d,
                      generation_best_parameters,
                      spatial_layer_function_types_1d,
                      valid_mask_array,
                      p.change_class_labels,
                      observed_lulc_array,
                      hectares_per_grid_cell,
                      p.cur_dir,
                      p.reporting_level,
                      p.loss_function_sigma,
                      p.call_string)

        # LEARNING POINT: GDAL silently fails to write if you have a file path too long. This happened below.

        # Write generated arrays to disk
        lulc_baseline_path = os.path.join(p.cur_dir, 'lulc_baseline.tif')
        generated_gt = hb.generate_geotransform_of_chunk_from_cr_size_and_larger_path(p.fine_blocks_list, p.base_year_lulc_path)
        generated_projection = hb.common_projection_wkts['wgs84']
        lulc_projected_array = lulc_projected_array.astype(np.int8)
        hb.save_array_as_geotiff(lulc_projected_array, lulc_projected_path, lulc_baseline_path, projection_override=generated_projection, ndv=255, data_type=1, compress=True, verbose=True)
        # hb.save_array_as_geotiff(lulc_projected_array, lulc_projected_path, lulc_baseline_path, geotransform_override=generated_gt, projection_override=generated_projection, ndv=255, data_type=1, compress=True, verbose=True)

        p.L.info(str(time.time() - start), 'Elapsed for generating lulc_projected_path', lulc_projected_path)
        if p.reporting_level > 11:
            lulc_baseline_path = os.path.join(p.cur_dir, 'lulc_baseline.png')
            hb.show(lulc_baseline_array, output_path=os.path.join(p.cur_dir, 'input_lulc.png'), title='input_lulc', vmin=0, vmax=7, ndv=255, block_plotting=False)

        if p.reporting_level > 4:
        # if p.reporting_level > 4 and p.calibrate:
            pass
            # seals_utils.plot_generation(p, generation_id)


def allocation_exclusive(passed_p=None):
    # IN DEVELOPMENT/UNUSED
    if passed_p is None:
        global p
    else:
        p = passed_p

    start = time.time()

    # Set where CHUNK-specific maps will be saved.
    p.lulc_baseline_path = os.path.join(p.cur_dir, 'lulc_baseline.tif')
    p.zone_esa_seals7_lulc_base_year_path = os.path.join(p.cur_dir, 'zone_esa_seals7_lulc_base_year.tif')
    p.fine_match_path = p.lulc_baseline_path

    p.lulc_ndv = hb.get_ndv_from_path(p.base_year_lulc_path)

    p.loss_function_sigma = np.float64(7.0) # Set how much closeness vs farness matters in assessing accuracy.
    # Sigma = 1 means you need to be REALLY close to count as a good prediction.

    # Load the coefficients as DF from either calibration dir or a prebuilt dir.
    if p.calibrate:
        zone_string = os.path.split(p.cur_dir_parent_dir)[1]
        current_pretrained_coefficients_path = os.path.join(p.calibration_dir, zone_string, 'calibration_zones', 'trained_coefficients_zone_' + zone_string + '.xlsx')
        p.L.info('Setting current_pretrained_coefficients_path to one generated in this project, at ' + current_pretrained_coefficients_path)
    else:
        current_pretrained_coefficients_path = p.pretrained_coefficients_path_dict[p.current_luh_scenario_label][p.current_year][p.current_policy_scenario_label]
        p.L.info('Setting current_pretrained_coefficients_path to one specified in run configuration, at ' + current_pretrained_coefficients_path)

    p.spatial_regressors_df = pd.read_excel(current_pretrained_coefficients_path)

    p.change_class_labels = np.asarray(p.class_indices, dtype=np.int64)  # For Cythonization, load these as the "labels", which is used for writing.

    lulc_projected_path = os.path.join(p.cur_dir, 'lulc_seals7_projected.tif')

    if p.skip_created_downscaling_zones:
        skip_this_zone = os.path.exists(lulc_projected_path)
    else:
        skip_this_zone = False

    p.projected_current_coarse_change_input_paths = []
    for class_label in p.class_labels:
        p.projected_current_coarse_change_input_paths.append(os.path.join(p.prepare_lulc_dir, class_label + '_projected_change.tif'))

    if p.run_this and not skip_this_zone:

        # p.lulc_simplified_paths[os.path.splitext(filename)[0]]
        lulc_baseline_array = hb.load_geotiff_chunk_by_cr_size(p.base_year_simplified_lulc_path, p.fine_blocks_list, datatype=np.int64, output_path=p.lulc_baseline_path).astype(np.int64)

        p.spatial_layer_names = p.spatial_regressors_df['spatial_regressor_name'].values
        p.spatial_layer_paths = p.spatial_regressors_df['data_location'].values
        p.spatial_layer_types = p.spatial_regressors_df['type'].values

        # QUIRCK, adjacency is really just additive with preprocessing.
        spatial_layer_types_to_codes = {'multiplicative': 1,
                                        'additive': 2,
                                        }
        spatial_layer_types_to_codes.update({'gaussian_' + str(sigma): 2 for sigma in p.kernel_halflives})

        spatial_layer_function_types_1d = np.asarray([spatial_layer_types_to_codes[i] for i in p.spatial_layer_types], np.int64)

        # CREATE GAUSSIANS for all variables tagged as that type. Note this is a large performance loss and I use precached global convolutions in all cases to date.
        p.spatial_layer_chunk_paths = []
        for c, path in enumerate(p.spatial_layer_paths):
            if p.spatial_regressors_df['type'].values[c][0:8] == 'gaussian':
                _, class_id, _, sigma = p.spatial_regressors_df['spatial_regressor_name'].values[c].split('_')
                filename = os.path.split(path)[1]
                spatial_chunk_path = os.path.join(p.cur_dir, os.path.split(p.spatial_regressors_df['data_location'].values[c])[1])
                if not os.path.exists(spatial_chunk_path):
                    hb.load_geotiff_chunk_by_cr_size(path, p.fine_blocks_list, output_path=spatial_chunk_path)
                p.spatial_layer_chunk_paths.append(spatial_chunk_path)

        spatial_layer_chunk_counter = 0
        for c, class_label in enumerate(p.spatial_regressors_df['spatial_regressor_name'].values):
            if p.spatial_regressors_df['type'].values[c][0:8] == 'gaussian':
                _, class_id, _, sigma = class_label.split('_')

                kernel_path = os.path.join(p.generated_kernels_dir, 'gaussian_' + str(sigma) + '.tif')
                output_path = os.path.join(p.cur_dir, class_label + '_convolution.tif')

                # NOTE, fft_gaussian has to write to disk
                if not os.path.exists(output_path):
                    seals_utils.fft_gaussian(p.spatial_layer_chunk_paths[spatial_layer_chunk_counter], kernel_path, output_path, -9999.0, True)

                spatial_layer_chunk_counter += 1

        n_c, n_r = int(p.fine_blocks_list[2]), int(p.fine_blocks_list[3])
        coarse_n_c, coarse_n_r = int(p.coarse_blocks_list[2]), int(p.coarse_blocks_list[3])

        # Load things that dont ever change over generations or final run
        hectares_per_grid_cell = hb.load_geotiff_chunk_by_cr_size(p.ha_per_cell_10sec_path, p.fine_blocks_list).astype(np.float64)

        # Build the numpy array for spatial layers.
        spatial_layers_3d = np.zeros((len(p.spatial_layer_paths), n_r, n_c)).astype(np.float64)

        # Chose not to normalize anything.
        normalize_inputs = False

        # Add either the normalized or not normalized array to the spatial_layers_3d
        for c, path in enumerate(p.spatial_layer_paths):
            p.L.debug('Loading spatial layer at path ' + path)
            # TODOO p.L.critical('This replacement needs to be fixed.')
            path = path.replace('combined_policies', '\worldbank_rtk_feedback_model')
            if p.spatial_layer_types[c] == 'additive' or p.spatial_layer_types[c] == 'multiplicative':
                if normalize_inputs is True:
                    spatial_layers_3d[c] = hb.normalize_array(hb.load_geotiff_chunk_by_cr_size(path, p.fine_blocks_list))
                else:
                    spatial_layers_3d[c] = hb.load_geotiff_chunk_by_cr_size(path, p.fine_blocks_list)
            elif p.spatial_layer_types[c][0:8] == 'gaussian':
                updated_path = os.path.join(p.cur_dir, 'class_' + p.spatial_layer_names[c].split('_')[1] + '_gaussian_' + p.spatial_layer_names[c].split('_')[3] + '_convolution.tif')

                if normalize_inputs is True:
                    spatial_layers_3d[c] = hb.normalize_array(hb.as_array(updated_path))
                else:
                    spatial_layers_3d[c] = hb.as_array(updated_path) # NOTE assumes already clipped
            else:
                raise NameError('unspecified type')

        # Load baseline lulc and valid mask
        if p.is_gtap1_run:
            observed_lulc_array = hb.load_geotiff_chunk_by_cr_size(p.base_year_simplified_lulc_path, p.fine_blocks_list, datatype=np.int64, output_path=None).astype(np.int64)
        elif p.is_magpie_run:
            if not p.is_first_pass:

                # HACK. Until I create a full dynamic run framework for seals, these types of errors are going to keep arising.
                if p.adjust_baseline_to_match_magpie_2015:
                    starting_lulc_path = os.path.join(p.cur_dir.replace('rcp45_ssp2\\2050', 'baseline\\2015').replace('zones', 'zones_adjusted').replace('scenarios', 'esa_luh_baseline_lulc_adjustment'), 'lulc_projected.tif')
                    for i in p.magpie_policy_scenario_labels:
                        starting_lulc_path = starting_lulc_path.replace('\\' + i, '')
                else:
                    starting_lulc_path = p.base_year_simplified_lulc_path

                observed_lulc_array = hb.load_geotiff_chunk_by_cr_size(starting_lulc_path, p.fine_blocks_list, datatype=np.int64, output_path=None).astype(np.int64)
                lulc_baseline_array = hb.load_geotiff_chunk_by_cr_size(p.base_year_simplified_lulc_path, p.fine_blocks_list, datatype=np.int64, output_path=p.lulc_baseline_path).astype(np.int64)

            else:
                starting_lulc_path = p.base_year_simplified_lulc_path
                lulc_baseline_array = hb.load_geotiff_chunk_by_cr_size(p.base_year_simplified_lulc_path, p.fine_blocks_list, datatype=np.int64, output_path=p.lulc_baseline_path).astype(np.int64)

                observed_lulc_array = hb.load_geotiff_chunk_by_cr_size(starting_lulc_path, p.fine_blocks_list, datatype=np.int64, output_path=None).astype(np.int64)
        elif p.is_calibration_run:
            observed_lulc_array = hb.load_geotiff_chunk_by_cr_size(p.base_year_simplified_lulc_path, p.fine_blocks_list, datatype=np.int64, output_path=None).astype(np.int64)
        valid_mask_array = np.where((observed_lulc_array != p.lulc_ndv), 1, 0).astype(np.int64)

        # Set how much change for each class needs to be allocated.
        projected_coarse_change_3d = np.zeros((len(p.class_labels), coarse_n_r, coarse_n_c)).astype(np.float64)
        for c, path in enumerate(list(p.projected_coarse_change_files[p.current_luh_scenario_label][p.current_year][p.current_policy_scenario_label].values())):

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
                projected_coarse_change_3d[c] = hb.load_geotiff_chunk_by_cr_size(path, p.coarse_blocks_list).astype(np.float64)

        p.seals_class_names = p.spatial_regressors_df.columns.values[3:]
        spatial_regressor_trained_coefficients = p.spatial_regressors_df[p.seals_class_names].values.astype(np.float64).T
        generation_best_parameters = np.copy(spatial_regressor_trained_coefficients)

        p.call_string = ''

        # L.setLevel(logging.DEBUG)
        lulc_baseline_array = lulc_baseline_array.astype(np.int64)
        p.L.debug('projected_coarse_change_3d', type(projected_coarse_change_3d), projected_coarse_change_3d.dtype)
        p.L.debug('lulc_baseline_array', type(lulc_baseline_array), lulc_baseline_array.dtype, lulc_baseline_array)
        p.L.debug('spatial_layers_3d', type(spatial_layers_3d), spatial_layers_3d.dtype, spatial_layers_3d)
        p.L.debug('generation_best_parameters', type(generation_best_parameters), generation_best_parameters.dtype, generation_best_parameters)
        p.L.debug('spatial_layer_function_types_1d', type(spatial_layer_function_types_1d), spatial_layer_function_types_1d.dtype, spatial_layer_function_types_1d)
        p.L.debug('valid_mask_array', type(valid_mask_array), valid_mask_array.dtype, valid_mask_array)
        p.L.debug('p.change_class_labels', type(p.change_class_labels), p.change_class_labels.dtype, p.change_class_labels)
        p.L.debug('observed_lulc_array', type(observed_lulc_array), observed_lulc_array.dtype, observed_lulc_array)
        p.L.debug('hectares_per_grid_cell', type(hectares_per_grid_cell), hectares_per_grid_cell.dtype, hectares_per_grid_cell)
        p.L.debug('p.cur_dir', type(p.cur_dir), p.cur_dir)
        p.L.debug('p.loss_function_sigma', type(p.loss_function_sigma), p.loss_function_sigma)
        p.L.debug('p.call_string', type(p.call_string), p.call_string)

        # Strange choice, but the allocation function both calibrates AND RUNS the final projection using the calibration. If instead you want to
        # Run on precalibrated parameters, you have to reload coarse_change_3d.
        overall_similarity_score, lulc_projected_array, overall_similarity_plot, class_similarity_scores, class_similarity_plots = \
            calibrate_exclusive(projected_coarse_change_3d,
                      lulc_baseline_array,
                      spatial_layers_3d,
                      generation_best_parameters,
                      spatial_layer_function_types_1d,
                      valid_mask_array,
                      p.change_class_labels,
                      observed_lulc_array,
                      hectares_per_grid_cell,
                      p.cur_dir,
                      p.reporting_level,
                      p.loss_function_sigma,
                      p.call_string)

        # LEARNING POINT: GDAL silently fails to write if you have a file path too long. This happened below.

        # Write generated arrays to disk
        lulc_baseline_path = os.path.join(p.cur_dir, 'lulc_baseline.tif')
        generated_gt = hb.generate_geotransform_of_chunk_from_cr_size_and_larger_path(p.fine_blocks_list, p.base_year_simplified_lulc_path)
        generated_projection = hb.common_projection_wkts['wgs84']
        lulc_projected_array = lulc_projected_array.astype(np.int8)
        hb.save_array_as_geotiff(lulc_projected_array, lulc_projected_path, lulc_baseline_path, ndv=255, data_type=1, compress=True, verbose=True)
        # hb.save_array_as_geotiff(lulc_projected_array, lulc_projected_path, lulc_baseline_path, geotransform_override=generated_gt, projection_override=generated_projection, ndv=255, data_type=1, compress=True, verbose=True)

        p.L.info(str(time.time() - start), 'Elapsed for generating lulc_projected_path', lulc_projected_path)
        if p.reporting_level > 11:
            lulc_baseline_path = os.path.join(p.cur_dir, 'lulc_baseline.png')
            hb.show(lulc_baseline_array, output_path=os.path.join(p.cur_dir, 'input_lulc.png'), title='input_lulc', vmin=0, vmax=7, ndv=255, block_plotting=False)

        if p.reporting_level > 4:
        # if p.reporting_level > 4 and p.calibrate:
            pass
            # seals_utils.plot_generation(p, generation_id)


def allocation_from_change_matrix(passed_p=None):
    # IN DEVELOPMENT/UNUSED
    if passed_p is None:
        global p
    else:
        p = passed_p

    start = time.time()

    # Set where CHUNK-specific maps will be saved.
    p.lulc_baseline_path = os.path.join(p.cur_dir, 'lulc_baseline.tif')
    p.zone_esa_seals7_lulc_base_year_path = os.path.join(p.cur_dir, 'zone_esa_seals7_lulc_base_year.tif')
    p.fine_match_path = p.lulc_baseline_path

    p.lulc_ndv = hb.get_ndv_from_path(p.base_year_simplified_lulc_path)

    p.loss_function_sigma = np.float64(7.0) # Set how much closeness vs farness matters in assessing accuracy. Sigma = 1 means you need to be REALLY close to count as a food prediction.

    # Load the coefficients as DF from either calibration dir or a prebuilt dir.
    if p.calibrate:
        zone_string = os.path.split(p.cur_dir_parent_dir)[1]
        current_pretrained_coefficients_path = os.path.join(p.calibration_dir, zone_string, 'calibration_zones', 'trained_coefficients_zone_' + zone_string + '.xlsx')
        p.L.info('Setting current_pretrained_coefficients_path to one generated in this project, at ' + current_pretrained_coefficients_path)
    else:
        current_pretrained_coefficients_path = p.pretrained_coefficients_path_dict[p.current_luh_scenario_label][p.current_year][p.current_policy_scenario_label]
        p.L.info('Setting current_pretrained_coefficients_path to one specified in run configuration, at ' + current_pretrained_coefficients_path)

    p.spatial_regressors_df = pd.read_excel(current_pretrained_coefficients_path)

    p.change_class_labels = np.asarray(p.class_indices, dtype=np.int64)  # For Cythonization, load these as the "labels", which is used for writing.

    lulc_projected_path = os.path.join(p.cur_dir, 'lulc_seals7_projected.tif')

    if p.skip_created_downscaling_zones:
        skip_this_zone = os.path.exists(lulc_projected_path)
    else:
        skip_this_zone = False

    p.projected_current_coarse_change_input_paths = []
    for class_label in p.class_labels:
        p.projected_current_coarse_change_input_paths.append(os.path.join(p.prepare_lulc_dir, class_label + '_projected_change.tif'))


    if p.run_this and not skip_this_zone:
        lulc_baseline_array = hb.load_geotiff_chunk_by_cr_size(p.base_year_simplified_lulc_path, p.fine_blocks_list, datatype=np.int64, output_path=p.lulc_baseline_path).astype(np.int64)

        p.spatial_layer_names = p.spatial_regressors_df['spatial_regressor_name'].values
        p.spatial_layer_paths = p.spatial_regressors_df['data_location'].values
        p.spatial_layer_types = p.spatial_regressors_df['type'].values

        # QUIRCK, adjacency is really just additive with preprocessing.
        spatial_layer_types_to_codes = {'multiplicative': 1,
                                        'additive': 2,
                                        }
        spatial_layer_types_to_codes.update({'gaussian_' + str(sigma): 2 for sigma in p.kernel_halflives})

        spatial_layer_function_types_1d = np.asarray([spatial_layer_types_to_codes[i] for i in p.spatial_layer_types], np.int64)

        # CREATE GAUSSIANS for all variables tagged as that type. Note this is a large performance loss and I use precached global convolutions in all cases to date.
        p.spatial_layer_chunk_paths = []
        for c, path in enumerate(p.spatial_layer_paths):
            if p.spatial_regressors_df['type'].values[c][0:8] == 'gaussian':
                _, class_id, _, sigma = p.spatial_regressors_df['spatial_regressor_name'].values[c].split('_')
                filename = os.path.split(path)[1]
                spatial_chunk_path = os.path.join(p.cur_dir, os.path.split(p.spatial_regressors_df['data_location'].values[c])[1])
                if not os.path.exists(spatial_chunk_path):
                    hb.load_geotiff_chunk_by_cr_size(path, p.fine_blocks_list, output_path=spatial_chunk_path)
                p.spatial_layer_chunk_paths.append(spatial_chunk_path)

        spatial_layer_chunk_counter = 0
        for c, class_label in enumerate(p.spatial_regressors_df['spatial_regressor_name'].values):
            if p.spatial_regressors_df['type'].values[c][0:8] == 'gaussian':
                _, class_id, _, sigma = class_label.split('_')

                kernel_path = os.path.join(p.generated_kernels_dir, 'gaussian_' + str(sigma) + '.tif')
                output_path = os.path.join(p.cur_dir, class_label + '_convolution.tif')

                # NOTE, fft_gaussian has to write to disk
                if not os.path.exists(output_path):
                    seals_utils.fft_gaussian(p.spatial_layer_chunk_paths[spatial_layer_chunk_counter], kernel_path, output_path, -9999.0, True)

                spatial_layer_chunk_counter += 1

        n_c, n_r = int(p.fine_blocks_list[2]), int(p.fine_blocks_list[3])
        coarse_n_c, coarse_n_r = int(p.coarse_blocks_list[2]), int(p.coarse_blocks_list[3])

        # Load things that dont ever change over generations or final run
        hectares_per_grid_cell = hb.load_geotiff_chunk_by_cr_size(p.ha_per_cell_10sec_path, p.fine_blocks_list).astype(np.float64)

        # Build the numpy array for spatial layers.
        spatial_layers_3d = np.zeros((len(p.spatial_layer_paths), n_r, n_c)).astype(np.float64)

        # Chose not to normalize anything.
        normalize_inputs = False

        # Add either the normalized or not normalized array to the spatial_layers_3d
        for c, path in enumerate(p.spatial_layer_paths):
            p.L.debug('Loading spatial layer at path ' + path)
            # TODOO p.L.critical('This replacement needs to be fixed.')
            path = path.replace('combined_policies', '\worldbank_rtk_feedback_model')
            if p.spatial_layer_types[c] == 'additive' or p.spatial_layer_types[c] == 'multiplicative':
                if normalize_inputs is True:
                    spatial_layers_3d[c] = hb.normalize_array(hb.load_geotiff_chunk_by_cr_size(path, p.fine_blocks_list))
                else:
                    spatial_layers_3d[c] = hb.load_geotiff_chunk_by_cr_size(path, p.fine_blocks_list)
            elif p.spatial_layer_types[c][0:8] == 'gaussian':
                updated_path = os.path.join(p.cur_dir, 'class_' + p.spatial_layer_names[c].split('_')[1] + '_gaussian_' + p.spatial_layer_names[c].split('_')[3] + '_convolution.tif')

                if normalize_inputs is True:
                    spatial_layers_3d[c] = hb.normalize_array(hb.as_array(updated_path))
                else:
                    spatial_layers_3d[c] = hb.as_array(updated_path) # NOTE assumes already clipped
            else:
                raise NameError('unspecified type')

        # Load baseline lulc and valid mask
        if p.is_gtap1_run:
            observed_lulc_array = hb.load_geotiff_chunk_by_cr_size(p.base_year_simplified_lulc_path, p.fine_blocks_list, datatype=np.int64, output_path=None).astype(np.int64)
        elif p.is_magpie_run:
            if not p.is_first_pass:

                # HACK. Until I create a full dynamic run framework for seals, these types of errors are going to keep arising.
                if p.adjust_baseline_to_match_magpie_2015:
                    starting_lulc_path = os.path.join(p.cur_dir.replace('rcp45_ssp2\\2050', 'baseline\\2015').replace('zones', 'zones_adjusted').replace('scenarios', 'esa_luh_baseline_lulc_adjustment'), 'lulc_projected.tif')
                    for i in p.magpie_policy_scenario_labels:
                        starting_lulc_path = starting_lulc_path.replace('\\' + i, '')
                else:
                    starting_lulc_path = p.base_year_simplified_lulc_path

                observed_lulc_array = hb.load_geotiff_chunk_by_cr_size(starting_lulc_path, p.fine_blocks_list, datatype=np.int64, output_path=None).astype(np.int64)
                lulc_baseline_array = hb.load_geotiff_chunk_by_cr_size(p.base_year_simplified_lulc_path, p.fine_blocks_list, datatype=np.int64, output_path=p.lulc_baseline_path).astype(np.int64)

            else:
                starting_lulc_path = p.base_year_simplified_lulc_path
                lulc_baseline_array = hb.load_geotiff_chunk_by_cr_size(p.base_year_simplified_lulc_path, p.fine_blocks_list, datatype=np.int64, output_path=p.lulc_baseline_path).astype(np.int64)

                observed_lulc_array = hb.load_geotiff_chunk_by_cr_size(starting_lulc_path, p.fine_blocks_list, datatype=np.int64, output_path=None).astype(np.int64)
        elif p.is_calibration_run:
            observed_lulc_array = hb.load_geotiff_chunk_by_cr_size(p.base_year_simplified_lulc_path, p.fine_blocks_list, datatype=np.int64, output_path=None).astype(np.int64)
        valid_mask_array = np.where((observed_lulc_array != p.lulc_ndv), 1, 0).astype(np.int64)

        # Set how much change for each class needs to be allocated.
        # coarse_change_matrix_4d = np.zeros((coarse_n_r, coarse_n_c, len(p.class_labels), len(p.class_labels))).astype(np.float64)
        # for c, path in enumerate(list(p.projected_coarse_change_files[p.current_luh_scenario_label][p.current_year][p.current_policy_scenario_label].values())):

        coarse_change_matrix_2d = hb.as_array(p.allocation_full_change_matrix_path)
        change_matrix_edge_length = len(p.class_labels)
        coarse_change_matrix_4d = np.zeros((coarse_n_r, coarse_n_c, change_matrix_edge_length, change_matrix_edge_length))
        for coarse_r in range(coarse_n_r):
            for coarse_c in range(coarse_n_c):
                coarse_change_matrix_4d[coarse_r, coarse_c] = coarse_change_matrix_2d[coarse_r * change_matrix_edge_length: (coarse_r + 1) * change_matrix_edge_length,
                                                                                      coarse_c * change_matrix_edge_length: (coarse_c + 1) * change_matrix_edge_length]

        p.seals_class_names = p.spatial_regressors_df.columns.values[3:]
        spatial_regressor_trained_coefficients = p.spatial_regressors_df[p.seals_class_names].values.astype(np.float64).T
        generation_best_parameters = np.copy(spatial_regressor_trained_coefficients)

        p.call_string = ''

        # L.setLevel(logging.DEBUG)
        lulc_baseline_array = lulc_baseline_array.astype(np.int64)
        p.L.debug('coarse_change_matrix_4d', type(coarse_change_matrix_4d), coarse_change_matrix_4d.dtype)
        p.L.debug('lulc_baseline_array', type(lulc_baseline_array), lulc_baseline_array.dtype, lulc_baseline_array)
        p.L.debug('spatial_layers_3d', type(spatial_layers_3d), spatial_layers_3d.dtype, spatial_layers_3d)
        p.L.debug('generation_best_parameters', type(generation_best_parameters), generation_best_parameters.dtype, generation_best_parameters)
        p.L.debug('spatial_layer_function_types_1d', type(spatial_layer_function_types_1d), spatial_layer_function_types_1d.dtype, spatial_layer_function_types_1d)
        p.L.debug('valid_mask_array', type(valid_mask_array), valid_mask_array.dtype, valid_mask_array)
        p.L.debug('p.change_class_labels', type(p.change_class_labels), p.change_class_labels.dtype, p.change_class_labels)
        p.L.debug('observed_lulc_array', type(observed_lulc_array), observed_lulc_array.dtype, observed_lulc_array)
        p.L.debug('hectares_per_grid_cell', type(hectares_per_grid_cell), hectares_per_grid_cell.dtype, hectares_per_grid_cell)
        p.L.debug('p.cur_dir', type(p.cur_dir), p.cur_dir)
        p.L.debug('p.loss_function_sigma', type(p.loss_function_sigma), p.loss_function_sigma)
        p.L.debug('p.call_string', type(p.call_string), p.call_string)

        # Strange choice, but the allocation function both calibrates AND RUNS the final projection using the calibration. If instead you want to
        # Run on precalibrated parameters, you have to reload coarse_change_3d.
        overall_similarity_score, lulc_projected_array, overall_similarity_plot, class_similarity_scores, class_similarity_plots = \
            calibrate_from_change_matrix(coarse_change_matrix_4d,
                      lulc_baseline_array,
                      spatial_layers_3d,
                      generation_best_parameters,
                      spatial_layer_function_types_1d,
                      valid_mask_array,
                      p.change_class_labels,
                      observed_lulc_array,
                      hectares_per_grid_cell,
                      p.cur_dir,
                      p.reporting_level,
                      p.loss_function_sigma,
                      p.call_string)

        # LEARNING POINT: GDAL silently fails to write if you have a file path too long. This happened below.

        # Write generated arrays to disk
        lulc_baseline_path = os.path.join(p.cur_dir, 'lulc_baseline.tif')
        generated_gt = hb.generate_geotransform_of_chunk_from_cr_size_and_larger_path(p.fine_blocks_list, p.base_year_simplified_lulc_path)
        generated_projection = hb.common_projection_wkts['wgs84']
        lulc_projected_array = lulc_projected_array.astype(np.int8)
        hb.save_array_as_geotiff(lulc_projected_array, lulc_projected_path, lulc_baseline_path, ndv=255, data_type=1, compress=True, verbose=True)

        p.L.info(str(time.time() - start), 'Elapsed for generating lulc_projected_path', lulc_projected_path)
        if p.reporting_level > 11:
            lulc_baseline_path = os.path.join(p.cur_dir, 'lulc_baseline.png')
            hb.show(lulc_baseline_array, output_path=os.path.join(p.cur_dir, 'input_lulc.png'), title='input_lulc', vmin=0, vmax=7, ndv=255, block_plotting=False)

            for i in range(len(p.class_indices)):
                path = os.path.join(p.cur_dir, 'output_to_rank_for_class_' + str(i) + '.tif')
                hb.show(hb.as_array(path), output_path=os.path.join(p.cur_dir, 'output_to_rank_' + str(i) + '.png'), title='output_to_rank_for_class_' + str(i) , vmin=-100, vmax=100, block_plotting=False)

                path = os.path.join(p.cur_dir, 'output_rank_for_class_' + str(i) + '.tif')
                hb.show(hb.as_array(path), output_path=os.path.join(p.cur_dir, 'output_rank_' + str(i) + '.png'), title= 'output_rank_' + str(i), vmin=-100, vmax=100, block_plotting=False)

            lulc_baseline_path = os.path.join(p.cur_dir, 'lulc_baseline.png')
            hb.show(lulc_baseline_array, output_path=os.path.join(p.cur_dir, 'input_lulc.png'), title='input_lulc', vmin=-100, vmax=100, block_plotting=False)

def change_pngs(passed_p=None):
    if passed_p is None:
        global p
    else:
        p = passed_p

    if p.run_this:

        p.lulc_baseline_af = hb.ArrayFrame(p.lulc_baseline_path)
        lulc_projected_path = os.path.join(p.cur_dir_parent_dir, 'allocation', 'lulc_seals7_projected.tif')
        p.lulc_projected_af = hb.ArrayFrame(lulc_projected_path)

        for c, path in enumerate(p.projected_current_coarse_change_input_paths):
            scaled_proportion_to_allocate_path = os.path.join(p.prepare_lulc_dir, os.path.split(path)[1])
            change_array = hb.as_array(scaled_proportion_to_allocate_path)
            output_path = os.path.join(p.cur_dir, 'class_' + str(c + 1) + '_projected_expansion_and_contraction.png')

            seals_utils.show_class_expansions_vs_change(p.lulc_baseline_af.data, p.lulc_projected_af.data, c + 1, change_array, output_path,
                                            title='Class ' + str(c + 1) + ' projected expansion and contraction on coarse change')

def change_exclusive_pngs(passed_p=None):
    # IN DEVELOPMENT/UNUSED
    if passed_p is None:
        global p
    else:
        p = passed_p

    if p.run_this:

        p.lulc_baseline_af = hb.ArrayFrame(p.lulc_baseline_path)
        lulc_projected_path = os.path.join(p.cur_dir_parent_dir, 'allocation_exclusive', 'lulc_seals7_projected.tif')
        p.lulc_projected_af = hb.ArrayFrame(lulc_projected_path)

        for c, path in enumerate(p.projected_current_coarse_change_input_paths):
            scaled_proportion_to_allocate_path = os.path.join(p.prepare_lulc_dir, os.path.split(path)[1])
            change_array = hb.as_array(scaled_proportion_to_allocate_path)
            output_path = os.path.join(p.cur_dir, 'class_' + str(c + 1) + '_projected_expansion_and_contraction.png')

            seals_utils.show_class_expansions_vs_change(p.lulc_baseline_af.data, p.lulc_projected_af.data, c + 1, change_array, output_path,
                                            title='Class ' + str(c + 1) + ' projected expansion and contraction on coarse change')

def change_from_change_matrix_pngs(passed_p=None):
    # IN DEVELOPMENT/UNUSED
    if passed_p is None:
        global p
    else:
        p = passed_p

    if p.run_this:

        p.lulc_baseline_af = hb.ArrayFrame(p.lulc_baseline_path)
        lulc_projected_path = os.path.join(p.cur_dir_parent_dir, 'allocation_from_change_matrix', 'lulc_seals7_projected.tif')
        p.lulc_projected_af = hb.ArrayFrame(lulc_projected_path)

        for c, path in enumerate(p.projected_current_coarse_change_input_paths):
            scaled_proportion_to_allocate_path = os.path.join(p.prepare_lulc_dir, os.path.split(path)[1])
            change_array = hb.as_array(scaled_proportion_to_allocate_path)
            output_path = os.path.join(p.cur_dir, 'class_' + str(c + 1) + '_pec.png')
            seals_utils.show_class_expansions_vs_change(p.lulc_baseline_af.data, p.lulc_projected_af.data, c + 1, change_array, output_path,
                                            title='Class ' + str(c + 1) + ' projected expansion and contraction on coarse change')

def stitched_lulc_simplified_scenarios(p):
    """Stitch together the lulc_projected.tif files in each of the zones (or in the case of magpie, also in zones_adjusted).
    Also write on top of a global base map if selected so that areas not downscaled (like oceans) have the correct LULC from
    the base map. (E.g., we don't downscale the Falkland islands becuase SSPs don't have any change there. We don't want to delete
    the Falklands either, though.)"""

    if p.run_this:

        # Will overwrite this if the stitching doesn't match extent of input
        p.aligned_seals7_output_base_map_path = p.output_base_map_path

        for luh_scenario_label in p.luh_scenario_labels:
            for year in p.scenario_years:
                for policy_scenario_label in p.policy_scenario_labels:
                    include_string = 'lulc_seals7_'
                    target_dir = os.path.join(p.scenarios_dir, luh_scenario_label, str(year), policy_scenario_label)

                    p.layers_to_stitch = hb.list_filtered_paths_recursively(target_dir, include_strings=include_string, include_extensions='.tif', depth=None)

                    stitched_output_name = 'lulc_seals7_gtap1_' + luh_scenario_label + '_' + str(year) + '_' + policy_scenario_label
                    p.L.info('Stitching ' + str(len(p.layers_to_stitch)) + ' layers, first 1 of which was: ' + str(p.layers_to_stitch[:1]))
                    p.lulc_projected_stitched_path = os.path.join(p.cur_dir, stitched_output_name + '.tif')

                    if not hb.path_exists(p.lulc_projected_stitched_path):
                        if len(p.layers_to_stitch) > 0:

                            vrt_of_tiles_path = os.path.join(p.cur_dir, 'vrt_of_tiles.vrt')

                            # First, make a stitched VRT to get the bb of the generated tiles. Is fastish because write_vrt_to_tif=False
                            hb.create_gdal_virtual_raster_using_file(p.layers_to_stitch, vrt_of_tiles_path, write_vrt_to_tif=False, bands='all',
                                                                     remove_generator_files=True,
                                                                     srcnodata=None, dstnodata=255, compress=True, output_pixel_size=None, s_srs=None, t_srs=None, resampling_method='near',
                                                                     output_datatype='Byte')
                            p.bb_of_tiles = hb.get_bounding_box(vrt_of_tiles_path)

                            p.L.info('Calculated bb of tiles to be ' + str(p.bb_of_tiles))
                            global_bb = hb.get_bounding_box(p.output_base_map_path)

                            if p.bb_of_tiles != global_bb and not p.force_to_global_bb:
                                current_force_to_global_bb = 0
                            else:
                                current_force_to_global_bb = 1
                            p.L.warning('When stamping lulc, found that the set of tiles was not global, current_force_to_global_bb = ' + str(current_force_to_global_bb))

                            p.L.info('Stamping generated lulcs with extent_shift_match_path of output_base_map_path ' + str(p.output_base_map_path))
                            ndv = hb.get_datatype_from_uri(p.output_base_map_path)

                            # The only difference between these is if it forces the written raster to have a global extent
                            # applied. Default vrt processing will fill those with NDV for now.


                            if not current_force_to_global_bb:
                                # We stil need something to stamp onto so that non-allocated locations still have values,
                                # but this needs to be clipped to size.
                                p.clipped_output_base_map_path = os.path.join(p.cur_dir, 'lulc_seals7_gtap1_baseline_' + str(p.base_years[0]) + '.tif')
                                if not hb.path_exists(p.clipped_output_base_map_path):
                                    for base_year in p.base_years:
                                        p.clipped_output_base_map_path = os.path.join(p.cur_dir, 'lulc_seals7_gtap1_baseline_' + str(base_year) + '.tif')
                                        hb.clip_raster_by_bb(p.output_base_map_path, p.bb_of_tiles, p.clipped_output_base_map_path)
                                hb.create_gdal_virtual_raster_using_file(p.layers_to_stitch, p.lulc_projected_stitched_path, write_vrt_to_tif=True, bands='all',
                                             remove_generator_files=True,
                                             srcnodata=None, dstnodata=255, compress=True, output_pixel_size=None, s_srs=None, t_srs=None, resampling_method='near',
                                             output_datatype='Byte')
                            else:
                                # NOTE By inserting this in the front of the list, it makes sure it is BEHIND the newly generated tiles. It also ensures the BB is fully global.
                                p.layers_to_stitch.insert(0, p.output_base_map_path)
                                L.info('Added to stitch ' + p.output_base_map_path)
                                hb.add_class_counts_file_to_raster(p.output_base_map_path)
                                hb.create_gdal_virtual_raster_using_file(p.layers_to_stitch, p.lulc_projected_stitched_path, write_vrt_to_tif=True, bands='all',
                                             vrt_extent_shift_match_path=p.output_base_map_path, extent_shift_match_path=p.output_base_map_path,
                                             remove_generator_files=True,
                                             srcnodata=None, dstnodata=255, compress=True, output_pixel_size=None, s_srs=None, t_srs=None, resampling_method='near',
                                             output_datatype='Byte')

                    if p.write_global_lulc_seals7_scenarios_overview_and_tifs:
                        if p.force_to_global_bb or p.bb == hb.global_bounding_box:
                            hb.make_path_global_pyramid(p.lulc_projected_stitched_path)




        if p.is_magpie_run:
            for baseline_label in p.baseline_labels:
                for year in p.base_years:
                    scenario_string = baseline_label + '_' + str(year)

                    include_string = 'lulc_seals7_'

                    target_dir = os.path.join(p.esa_luh_baseline_lulc_adjustment_dir, baseline_label, str(year))

                    p.layers_to_stitch = hb.list_filtered_paths_recursively(target_dir, include_strings=include_string, include_extensions='.tif')

                    stitched_output_name = 'lulc_seals7_gtap1_' + baseline_label + '_' + str(year) + '_adjusted'
                    p.L.info('Stitching ' + str(len(p.layers_to_stitch)) + ' layers, first 1 of which was: ' + str(p.layers_to_stitch[:1]))
                    p.lulc_projected_stitched_path = os.path.join(p.cur_dir, stitched_output_name + '.tif')


                    vrt_of_tiles_path = os.path.join(p.cur_dir, 'vrt_of_tiles.vrt')

                    # First, make a stitched VRT to get the bb of the generated tiles. Is fastish because write_vrt_to_tif=False
                    hb.create_gdal_virtual_raster_using_file(p.layers_to_stitch, vrt_of_tiles_path, write_vrt_to_tif=False, bands='all',
                                                             remove_generator_files=True,
                                                             srcnodata=None, dstnodata=255, compress=True, output_pixel_size=None, s_srs=None, t_srs=None, resampling_method='near',
                                                             output_datatype='Byte')
                    p.bb_of_tiles = hb.get_bounding_box(vrt_of_tiles_path)

                    p.L.info('Calculated bb of tiles to be ' + str(p.bb_of_tiles))
                    global_bb = hb.get_bounding_box(p.output_base_map_path)

                    if p.bb_of_tiles != global_bb and not p.force_to_global_bb:
                        current_force_to_global_bb = 0
                    else:
                        current_force_to_global_bb = 1
                    p.L.warning('When stamping lulc, found that the set of tiles was not global, current_force_to_global_bb = ' + str(current_force_to_global_bb))

                    p.L.info('Stamping generated lulcs with extent_shift_match_path of output_base_map_path ' + str(p.output_base_map_path))
                    ndv = hb.get_datatype_from_uri(p.output_base_map_path)

                    # The only difference between these is if it forces the written raster to have a global extent
                    # applied. Default vrt processing will fill those with NDV for now.

                    if not current_force_to_global_bb:
                        # We stil need something to stamp onto so that non-allocated locations still have values,
                        # but this needs to be clipped to size.
                        temp_tif_path = hb.temp('.tif', filename_start='base_clipped_to_tile_extent', remove_at_exit=True, folder=p.cur_dir)
                        hb.clip_raster_by_bb(p.output_base_map_path, p.bb_of_tiles, temp_tif_path)
                        p.layers_to_stitch.insert(0, temp_tif_path)
                        hb.create_gdal_virtual_raster_using_file(p.layers_to_stitch, p.lulc_projected_stitched_path, write_vrt_to_tif=True, bands='all',
                                                                 remove_generator_files=True,
                                                                 srcnodata=None, dstnodata=255, compress=True, output_pixel_size=None, s_srs=None, t_srs=None, resampling_method='near',
                                                                 output_datatype='Byte')
                    else:
                        # NOTE By inserting this in the front of the list, it makes sure it is BEHIND the newly generated tiles. It also ensures the BB is fully global.
                        p.layers_to_stitch.insert(0, p.output_base_map_path)
                        hb.create_gdal_virtual_raster_using_file(p.layers_to_stitch, p.lulc_projected_stitched_path, write_vrt_to_tif=True, bands='all',
                                                                 vrt_extent_shift_match_path=p.output_base_map_path, extent_shift_match_path=p.output_base_map_path,
                                                                 remove_generator_files=True,
                                                                 srcnodata=None, dstnodata=255, compress=True, output_pixel_size=None, s_srs=None, t_srs=None, resampling_method='near',
                                                                 output_datatype='Byte')
                if p.write_global_lulc_seals7_scenarios_overview_and_tifs:
                    if current_force_to_global_bb:
                        hb.make_path_global_pyramid(p.lulc_projected_stitched_path)

def stitched_lulc_esa_scenarios(p):

    def fill_where_not_changed(changed, baseline, esa):
        return np.where(changed == baseline, esa, changed)

    if p.run_this:


        # Note difference between simplified and full esa here.
        baseline_seals7_lulc_label = 'lulc_seals7_gtap1_baseline_' + str(p.base_years[0])

        baseline_esa_lulc_label = 'lulc_esa_gtap1_baseline_' + str(p.base_year)
        p.base_year_esa_lulc_path = os.path.join(p.cur_dir, baseline_esa_lulc_label + '.tif')

        if os.path.exists(os.path.join(p.stitched_lulc_simplified_scenarios_dir, baseline_seals7_lulc_label + '.tif')):
            p.aligned_seals7_output_base_map_path = os.path.join(p.stitched_lulc_simplified_scenarios_dir, baseline_seals7_lulc_label + '.tif')
        else:
            p.aligned_seals7_output_base_map_path = p.output_base_map_path

        # if not hb.path_exists(p.base_year_esa_lulc_path):
        #     hb.copy_shutil_flex(p.base_year_lulc_path, p.base_year_esa_lulc_path)

        if p.is_magpie_run:
            for baseline_label in p.baseline_labels:
                for year in p.base_years:
                    seals7_include_string = 'lulc_seals7_gtap1_' + baseline_label + '_' + str(year) + '_adjusted'
                    esa_include_string = 'lulc_esa_gtap1_' + baseline_label + '_' + str(year) + '_adjusted'

                    p.seals7_projected_stitched_path = os.path.join(p.stitched_lulc_simplified_scenarios_dir, seals7_include_string + '.tif')

                    stitched_bb = hb.get_bounding_box(p.seals7_projected_stitched_path)
                    global_bb = hb.get_bounding_box(p.base_year_lulc_path)

                    p.L.info('Stitched_bb: ' + str(stitched_bb))
                    p.L.info('global_bb: ' + str(global_bb))

                    if stitched_bb != global_bb:
                        baseline_seals7_lulc_label = 'lulc_esa_gtap1_baseline_' + str(p.base_year)

                        p.aligned_esa_output_base_map_path = os.path.join(p.cur_dir, baseline_seals7_lulc_label + '.tif')
                        if not hb.path_exists(p.aligned_esa_output_base_map_path):
                            hb.clip_raster_by_bb(p.base_year_lulc_path, stitched_bb, p.aligned_esa_output_base_map_path)
                    else:
                        p.aligned_esa_output_base_map_path = p.base_year_lulc_path

                    base_raster_path_band_const_list = [
                        (p.seals7_projected_stitched_path, 1),
                        (p.aligned_seals7_output_base_map_path, 1),
                        (p.aligned_esa_output_base_map_path, 1),
                    ]
                    target_raster_pre_path = hb.temp('.tif', seals7_include_string + '_prereclass', True, p.cur_dir)
                    # target_raster_pre_path = os.path.join(p.cur_dir, esa_include_string + '_pre.tif')
                    p.lulc_projected_stitched_path = os.path.join(p.cur_dir, esa_include_string + '.tif')

                    datatype_target = 1
                    nodata_target = 255

                    # TODOO Massive optimization here would be to just have the reclass happen in the fill_where_not_changed function..., and or make it parallel.
                    if not hb.path_exists(target_raster_pre_path) and not hb.path_exists(p.lulc_projected_stitched_path):
                        p.L.info('Starting raster calculator with ' + str(target_raster_pre_path) + ' and ' + str(base_raster_path_band_const_list))
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
                        p.L.info('Starting reclassify_flex with ' + str(target_raster_pre_path) + ' and ' + str(p.lulc_projected_stitched_path))

                        hb.reclassify_flex(target_raster_pre_path, rules_extended_with_existing_esa_classes, p.lulc_projected_stitched_path, output_data_type=1)

                    if p.write_global_lulc_seals7_scenarios_overview_and_tifs:
                        if p.force_to_global_bb or p.bb == hb.global_bounding_box:
                            hb.make_path_global_pyramid(p.lulc_projected_stitched_path)

        for luh_scenario_label in p.luh_scenario_labels:
            for year in p.scenario_years:
                for policy_scenario_label in p.policy_scenario_labels:
                    seals7_include_string = 'lulc_seals7_gtap1_' + luh_scenario_label + '_' + str(year) + '_' + policy_scenario_label
                    esa_include_string = 'lulc_esa_gtap1_' + luh_scenario_label + '_' + str(year) + '_' + policy_scenario_label

                    p.seals7_projected_stitched_path = os.path.join(p.stitched_lulc_simplified_scenarios_dir, seals7_include_string + '.tif')
                    p.lulc_projected_stitched_path = os.path.join(p.stitched_lulc_simplified_scenarios_dir, esa_include_string + '.tif')

                    stitched_bb = hb.get_bounding_box(p.seals7_projected_stitched_path)
                    global_bb = hb.get_bounding_box(p.base_year_lulc_path)

                    p.L.info('Stitched_bb: ' + str(stitched_bb))
                    p.L.info('global_bb: ' + str(global_bb))

                    if stitched_bb != global_bb:
                        baseline_esa_gtap1_lulc_label = 'lulc_esa_gtap1_baseline_' + str(p.base_year)

                        p.aligned_esa_output_base_map_path = os.path.join(p.cur_dir, baseline_esa_gtap1_lulc_label + '.tif')
                        if not hb.path_exists(p.aligned_esa_output_base_map_path):
                            hb.clip_raster_by_bb(p.base_year_lulc_path, stitched_bb, p.aligned_esa_output_base_map_path)
                    else:
                        p.aligned_esa_output_base_map_path = p.base_year_lulc_path

                    base_raster_path_band_const_list = [
                        (p.seals7_projected_stitched_path, 1),
                        (p.aligned_seals7_output_base_map_path, 1),
                        (p.aligned_esa_output_base_map_path, 1),
                    ]

                    target_raster_pre_path = hb.temp('.tif', esa_include_string + '_prereclass', True, p.cur_dir)

                    p.lulc_projected_stitched_path = os.path.join(p.cur_dir, esa_include_string + '.tif')

                    datatype_target = 1
                    nodata_target = 255

                    # TODOO Massive optimization here would be to just have the reclass happen in the fill_where_not_changed function..., and or make it parallel.
                    if not hb.path_exists(target_raster_pre_path) and not hb.path_exists(p.lulc_projected_stitched_path):
                        p.L.info('Starting raster calculator with ' + str(target_raster_pre_path) + ' and ' + str(base_raster_path_band_const_list))

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
                        p.L.info('Starting reclassify_flex with ' + str(target_raster_pre_path) + ' and ' + str(p.lulc_projected_stitched_path))

                        hb.reclassify_flex(target_raster_pre_path, rules_extended_with_existing_esa_classes, p.lulc_projected_stitched_path, output_data_type=1)

                    if p.write_global_lulc_seals7_scenarios_overview_and_tifs:
                        if p.force_to_global_bb or p.bb == hb.global_bounding_box:
                            hb.make_path_global_pyramid(p.lulc_projected_stitched_path)


def prepare_lulc_make_pngs(p):
    full_change_matrix_no_diagonal = None
    n_classes = None
    full_change_matrix_no_diagonal = None
    if p.run_this:
        from matplotlib import colors as colors
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 8)

        # cmap = ge.generate_custom_colorbar(full_change_matrix_no_diagonal, color_scheme='bold_spectral_white_left')
        # cmap = ge.generate_custom_colorbar(full_change_matrix_no_diagonal, color_scheme='bold_spectral_white_left')
        # Plot the heatmap
        vmin = np.min(full_change_matrix_no_diagonal)
        vmax = np.max(full_change_matrix_no_diagonal)
        im = ax.imshow(full_change_matrix_no_diagonal, cmap='YlGnBu', norm=colors.LogNorm(vmin=vmin + 1, vmax=vmax))
        #
        # pim = ax.pcolormesh(full_change_matrix_no_diagonal,
        #                     norm=colors.LogNorm(vmin=full_change_matrix_no_diagonal.min(), vmax=full_change_matrix_no_diagonal.max()),
        #                     cmap='PuBu_r')
        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label('Number of cells changed from class ROW to class COL', size=10)
        # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # We want to show all ticks...
        ax.set_xticks(np.arange(full_change_matrix_no_diagonal.shape[1]))
        ax.set_yticks(np.arange(full_change_matrix_no_diagonal.shape[0]))
        # ... and label them with the respective list entries.

        row_labels = []
        col_labels = []
        for i in range(n_classes * p.coarse_match.n_rows):
            class_id = i % n_classes
            coarse_grid_cell_counter = int(i / n_classes)
            row_labels.append(str(class_id))
            col_labels.append(str(class_id))

        trans = ax.get_xaxis_transform()  # x in data untis, y in axes fraction

        for i in range(p.coarse_match.n_rows):
            ann = ax.annotate('Zone ' + str(i + 1), xy=(-3.5, i / p.coarse_match.n_rows + .5 / p.coarse_match.n_rows), xycoords=trans)
            # ann = ax.annotate('Class ' + str(i + 1), xy=(-2.5, i / p.coarse_match.n_rows + .5 / p.coarse_match.n_rows), xycoords=trans)
            ann = ax.annotate('Zone ' + str(i + 1), xy=(i * (p.coarse_match.n_rows + 1) + .25 * p.coarse_match.n_rows, 1.05), xycoords=trans)  #
            # ann = ax.annotate('MgII', xy=(-2, 1 / (i * n_classes + n_classes / 2)), xycoords=trans)
            # plt.annotate('This is awesome!',
            #              xy=(-.1, i * n_classes + n_classes / 2),
            #              xycoords='data',
            #              textcoords='offset points',
            #              arrowprops=dict(arrowstyle="->"))

        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center", rotation_mode="anchor")

        # im, cbar = heatmap(full_change_matrix_no_diagonal, row_labels, col_labels, ax=ax,
        #                    cmap="YlGn", cbarlabel="harvest [t/year]")
        # Turn spines off and create white grid.
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(full_change_matrix_no_diagonal.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(full_change_matrix_no_diagonal.shape[0] + 1) - .5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        full_change_matrix_no_diagonal_png_path = os.path.join(p.cur_dir, 'full_change_matrix_no_diagonal.png')
        # texts = annotate_heatmap(im, valfmt="{x:.1f} t")

        major_gridline = False
        for i in range(n_classes * p.coarse_match.n_rows + 1):
            try:
                if i % n_classes == 0:
                    major_gridline = i
                else:
                    major_gridline = False
            except:
                major_gridline = 0

            if major_gridline is not False:
                xloc = major_gridline - .5
                yloc = major_gridline - .5
                ax.axvline(x=xloc, color='grey')
                ax.axhline(y=yloc, color='grey')

        # ax.axvline(x = 0 - .5, color='grey')
        # ax.axvline(x = 4 - .5, color='grey')
        # ax.axhline(y = 0 - .5, color='grey')
        # ax.axhline(y = 4 - .5, color='grey')

        # plt.title('Cross-zone change matrix')
        # ax.cbar_label('Number of cells changed from class ROW to class COL')
        plt.savefig(full_change_matrix_no_diagonal_png_path)

        vmax = np.max(full_change_matrix_no_diagonal)
        # full_change_matrix_no_diagonal_png_path = os.path.join(p.cur_dir, 'full_change_matrix_no_diagonal.png')
        # fig, ax = plt.subplots()
        # im = ax.imshow(full_change_matrix_no_diagonal)
        # ax.axvline(x=.5, color='red')
        # ax.axhline(y=.5, color='yellow')
        # plt.title('Draw a line on an image with matplotlib')

        # plt.savefig(full_change_matrix_no_diagonal_png_path)

        full_change_matrix_no_diagonal_png_path = os.path.join(p.cur_dir, 'full_change_matrix_no_diagona_auto.png')
        hb.full_show_array(full_change_matrix_no_diagonal, output_path=full_change_matrix_no_diagonal_png_path, cbar_label='Number of changes from class R to class C per tile', title='Change matrix mosaic',
                           num_cbar_ticks=2, vmin=0, vmid=vmax / 10.0, vmax=vmax, color_scheme='ylgnbu')
    return p




