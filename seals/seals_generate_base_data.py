import multiprocessing
import os

import geopandas as gpd
import hazelbean as hb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import geopandas as gpd
from hazelbean import config as hb_config

from seals import seals_utils

L = hb_config.get_logger()

def aligned_habitat_raster(p):

    if p.run_this:
        intput_raster_path = "C:/Users/jajohns/Files/gtap_invest/projects/habitat/input/preliminary_network_zt_run3.tif"
        aligned_habitat_raster_path = os.path.join(p.cur_dir, 'aligned_habitat_raster.tif')
        match_path = "C:/Users/jajohns/Files/gtap_invest/projects/habitat/input/quebec_zone_ids.tif"
        if not hb.path_exists(aligned_habitat_raster_path):
            hb.resample_to_match(intput_raster_path, match_path, aligned_habitat_raster_path)

        just_existing_protection_path = os.path.join(p.cur_dir, 'just_existing_protection.tif')
        if not hb.path_exists(just_existing_protection_path):
            hb.raster_calculator_flex(aligned_habitat_raster_path, lambda x: np.where(x == 2, 1, 0), output_path=just_existing_protection_path)

def biodiversity(p):
    # This is a dummy task to group things.
    pass

def kba(p):

    kbas_vector_path = p.get_path('biodiversity', 'kba', 'KBAsGlobal_2023_September_02_POL.shp', create_shortcut=True)
    p.kbas_rasterized_path = p.get_path('biodiversity', 'kba', 'kbas.tif', create_shortcut=True, verbose=True)
    if not hb.path_exists(p.kbas_rasterized_path):
        hb.convert_polygons_to_id_raster(kbas_vector_path, p.kbas_rasterized_path, p.base_year_lulc_path,
                                id_column_label='SitRecID', data_type=5, ndv=-9999, all_touched=False, compress=True)

        hb.make_path_global_pyramid(p.kbas_rasterized_path)


def star(p):
    star_threat_input_path = p.get_path('biodiversity', 'star', 'star_threat_input.tif')
    
    star_threat_input_path2 = p.get_path('star_threat_input.tif')

    
    # One solution would be to just require all paths to be defined relative to intermediate dir, duplicating the cur_dir approcah? downside here is it would't leverege paralleization, 
    star_threat_path = p.get_path('star_threat.tif') 
    
    if not hb.path_exists(star_threat_path):
        # WTF apparently this was all shifted by a random amount.?
        shift = 90.0 - 83.6361111111106226


        geotransform_input = hb.get_geotransform_path(star_threat_input_path)
        geotransform = list(geotransform_input)

        # Write a new file that corrects the weird shift without modifying the original file.
        star_threat_shifted_path = p.get_path('star_threat_shifted.tif')

        if geotransform[3] == 90.0:
            geotransform[3] -= shift
            if not hb.path_exists(star_threat_shifted_path):
                hb.set_geotransform_to_tuple(star_threat_input_path, geotransform, output_path=star_threat_shifted_path)

        # INTERSTING IMPLICATION: Because the cur_dir is already representing the base_data nesting location, adding it again doesn't work makes it be repeated 2x
        star_threat_padded_path = p.get_path('star_threat_padded.tif')
        if not hb.path_exists(star_threat_padded_path):
            hb.fill_to_match_extent(star_threat_shifted_path, p.aoi_ha_per_cell_fine_path, output_path=star_threat_padded_path, fill_value=-9999., remove_temporary_files=False)

        star_threat_path = p.get_path('star_threat.tif')
        if not hb.path_exists(star_threat_path):
            hb.make_path_global_pyramid(star_threat_padded_path, output_path=star_threat_path, verbose=True)
        pass


def fine_processed_inputs(p):
    p.task_note = """
    Contains all of the fine_processed_inputs that must be created given the state of the base_data and the aoi chosen.
    If the aoi is global, then the fine_processed_inputs will be the same as the base_data and wont be generated, assuming that year exists in the base_data. If that year doesn't exist, you may want to promote the results of this task to base_data status.
    Note also that some files, when run non-global, still will create a local copy of the file even tho in principle it could just access the rc_rowcol of the global version. This is for visualizatino tasks later on.
    Another case would be if a function (like raster_calculator) can only operate on paths. LAter on i would like to be able to pass the rc_rowcol to it and still operate on the single global version via pyramids.
    """
    pass



def lulc_clip(p):
    # Clip the fine LULC to the project AOI
    
    ## TODOO It is harder to implement task-level skipping when there is a base_data based file existence check
    
    if p.run_this:

        # In the event that aoi is not global, we will store aoi_lulc and global_lulc paths. In the global version
        # both of these dicts will be there, but they will be identical.
        p.base_data_lulc_src_paths = {}
        p.aoi_lulc_src_paths = {}
        p.lulc_src_paths = {}

        for index, row in p.scenarios_df.iterrows():
            seals_utils.assign_df_row_to_object_attributes(p, row)

            base_data_lulc_src_dir = os.path.join(p.base_data_dir, 'lulc', p.lulc_src_label)
            src_filename_start = 'lulc_' + p.lulc_src_label + '_'

            if p.scenario_type == 'baseline':
                if p.aoi != 'global':
                    for year in p.base_years:
                        p.base_data_lulc_src_paths[year] = os.path.join(base_data_lulc_src_dir, src_filename_start + str(year) + '.tif')
                        p.aoi_lulc_src_paths[year] = os.path.join(p.fine_processed_inputs_dir, 'lulc', p.lulc_src_label, src_filename_start + str(year) + '.tif')
                        p.lulc_src_paths[year] = p.aoi_lulc_src_paths[year] 
                        

                        if not hb.path_exists(p.aoi_lulc_src_paths[year]):
                            hb.create_directories(p.aoi_lulc_src_paths[year])
                            hb.clip_raster_by_bb(p.base_data_lulc_src_paths[year], p.bb, p.aoi_lulc_src_paths[year])
                else:
                    for year in p.base_years:
                        # filename = 'binary_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + str(year) + '_class_' + str(class_label) + '.tif'
                        # possible_dir = os.path.join('lulc', p.lulc_src_label, p.lulc_simplification_label, 'binaries', str(year))
                        # output_path = hb.get_first_extant_path(search_path, [p.fine_processed_inputs_dir, p.input_dir, p.base_data_dir])
                            
                        search_path = os.path.join('lulc', p.lulc_src_label, src_filename_start + str(year) + '.tif')
                        # p.base_data_lulc_src_paths[year] = hb.get_first_extant_path(search_path, [p.fine_processed_inputs_dir, p.input_dir, p.base_data_dir])
                        p.base_data_lulc_src_paths[year] = p.get_path(search_path)
                        p.aoi_lulc_src_paths[year] = p.base_data_lulc_src_paths[year] 
                        p.lulc_src_paths[year] = p.base_data_lulc_src_paths[year] 

                        if not hb.path_exists(p.aoi_lulc_src_paths[year]):
                            hb.create_directories(p.aoi_lulc_src_paths[year])
                            hb.clip_raster_by_bb(p.base_data_lulc_src_paths[year], p.bb, p.aoi_lulc_src_paths[year])



def lulc_clip_quick(p):
    # Clip the fine LULC to the project AOI
    
    ## TODOO It is harder to implement task-level skipping when there is a base_data based file existence check
    
    if p.run_this:

        # In the event that aoi is not global, we will store aoi_lulc and global_lulc paths. In the global version
        # both of these dicts will be there, but they will be identical.
        p.base_data_lulc_src_paths = {}
        p.aoi_lulc_src_paths = {}
        p.lulc_src_paths = {}  
        
        for index, row in p.scenarios_df.iterrows():
            seals_utils.assign_df_row_to_object_attributes(p, row)

            base_data_lulc_src_dir = os.path.join(p.base_data_dir, 'lulc', p.lulc_src_label)
            src_filename_start = 'lulc_' + p.lulc_src_label + '_'

            if p.scenario_type == 'baseline':
                if p.aoi != 'global':
                    for year in p.years:
                        p.base_data_lulc_src_paths[year] = os.path.join(base_data_lulc_src_dir, src_filename_start + str(year) + '.tif')
                        p.aoi_lulc_src_paths[year] = os.path.join(p.cur_dir, 'lulc', p.lulc_src_label, src_filename_start + str(year) + '.tif')
                        p.lulc_src_paths[year] = p.aoi_lulc_src_paths[year] 
                        

                        if not hb.path_exists(p.aoi_lulc_src_paths[year]):
                            hb.create_directories(p.aoi_lulc_src_paths[year])
                            hb.clip_raster_by_bb(p.base_data_lulc_src_paths[year], p.bb, p.aoi_lulc_src_paths[year])
                else:
                    for year in p.years:
                        # filename = 'binary_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + str(year) + '_class_' + str(class_label) + '.tif'
                        # possible_dir = os.path.join('lulc', p.lulc_src_label, p.lulc_simplification_label, 'binaries', str(year))
                        # output_path = hb.get_first_extant_path(search_path, [p.fine_processed_inputs_dir, p.input_dir, p.base_data_dir])

                        search_path = os.path.join('lulc', p.lulc_src_label, src_filename_start + str(year) + '.tif')
                        # p.base_data_lulc_src_paths[year] = hb.get_first_extant_path(search_path, [p.fine_processed_inputs_dir, p.input_dir, p.base_data_dir])
                        p.base_data_lulc_src_paths[year] = p.get_path(search_path)
                        p.aoi_lulc_src_paths[year] = p.base_data_lulc_src_paths[year]
                        p.lulc_src_paths[year] = p.base_data_lulc_src_paths[year]

                        if not hb.path_exists(p.aoi_lulc_src_paths[year]):
                            hb.create_directories(p.aoi_lulc_src_paths[year])
                            hb.clip_raster_by_bb(p.base_data_lulc_src_paths[year], p.bb, p.aoi_lulc_src_paths[year])



def lulc_simplifications(p):
    # Simplify the LULC

    if p.run_this:

        p.base_data_lulc_simplified_paths = {}
        p.aoi_lulc_simplified_paths = {}
        p.lulc_simplified_paths = {}

        for index, row in p.scenarios_df.iterrows():
            seals_utils.assign_df_row_to_object_attributes(p, row)

            base_data_lulc_simplified_dir = os.path.join(p.base_data_dir, 'lulc', p.lulc_src_label, p.lulc_simplification_label)
            simplified_filename_start = 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_'


            if p.scenario_type == 'baseline':
                if p.aoi != 'global':
                    for year in p.years:

                        p.base_data_lulc_simplified_paths[year] = os.path.join(base_data_lulc_simplified_dir, simplified_filename_start + str(year) + '.tif')
                        p.aoi_lulc_simplified_paths[year] = os.path.join(p.fine_processed_inputs_dir, 'lulc', p.lulc_src_label, p.lulc_simplification_label, simplified_filename_start + str(year) + '.tif')
                        p.lulc_simplified_paths[year] = p.aoi_lulc_simplified_paths[year]

                        if not hb.path_exists(p.aoi_lulc_simplified_paths[year]):
                            hb.create_directories(p.aoi_lulc_simplified_paths[year])
                            if hb.path_exists(p.base_data_lulc_simplified_paths[year]):
                                hb.clip_raster_by_bb(p.base_data_lulc_simplified_paths[year], p.bb, p.aoi_lulc_simplified_paths[year])
                            else:
                                rules = p.lulc_correspondence_dict['src_to_dst_reclassification_dict']
                                output_path = p.aoi_lulc_simplified_paths[year]
                                hb.reclassify_raster_hb(p.lulc_src_paths[year], rules, output_raster_path=output_path, output_data_type=1, array_threshold=10000, match_path=p.lulc_src_paths[year], verbose=False)

                else:
                    for year in p.base_years:
                        search_path = os.path.join('lulc', p.lulc_src_label, p.lulc_simplification_label, simplified_filename_start + str(year) + '.tif')
                        found_path = hb.get_first_extant_path(search_path, [p.fine_processed_inputs_dir, p.input_dir, p.base_data_dir])
                        p.base_data_lulc_simplified_paths[year] = found_path
                        p.aoi_lulc_simplified_paths[year] = found_path
                        p.lulc_simplified_paths[year] = found_path

                        if not hb.path_exists(p.aoi_lulc_simplified_paths[year]):
                            hb.create_directories(p.aoi_lulc_simplified_paths[year])

                            rules = p.lulc_correspondence_dict['src_to_dst_reclassification_dict']
                            output_path = p.aoi_lulc_simplified_paths[year]
                            hb.reclassify_raster_hb(p.lulc_src_paths[year], rules, output_raster_path=output_path, output_data_type=1, array_threshold=10000, match_path=p.lulc_src_paths[year], verbose=False)





def lulc_binaries(p):
    """Convert the simplified LULC into 1 binary presence map for each simplified LUC"""

    if p.run_this:

        p.base_data_binary_paths = {}
        p.aoi_binary_paths = {}
        p.binary_paths = {}

        for index, row in p.scenarios_df.iterrows():
            seals_utils.assign_df_row_to_object_attributes(p, row)

            base_data_lulc_binaries_dir = os.path.join(p.base_data_dir, 'lulc', p.lulc_src_label, p.lulc_simplification_label, 'binaries')
            binary_filename_start = 'binary_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_'

            if p.scenario_type == 'baseline':
                if p.aoi != 'global':
                    for year in p.base_years:

                        p.base_data_binary_paths[year] = {}
                        p.aoi_binary_paths[year] = {}
                        p.binary_paths[year] = {}

                        for class_label in p.all_class_labels:

                            p.base_data_binary_paths[year][class_label] = os.path.join(base_data_lulc_binaries_dir, str(year), binary_filename_start + str(year) + '_' + class_label +  '.tif')
                            p.aoi_binary_paths[year][class_label] = os.path.join(p.fine_processed_inputs_dir, 'lulc', p.lulc_src_label, p.lulc_simplification_label, 'binaries', str(year), binary_filename_start + str(year) + '_' + class_label  + '.tif')
                            p.binary_paths[year][class_label] = p.aoi_lulc_simplified_paths[year]

                            if not hb.path_exists(p.aoi_binary_paths[year][class_label]):
                                hb.create_directories(p.aoi_binary_paths[year][class_label])
                                if hb.path_exists(p.base_data_binary_paths[year][class_label]):
                                    hb.clip_raster_by_bb(p.base_data_binary_paths[year][class_label], p.bb, p.aoi_binary_paths[year][class_label])
                                else:
                                    output_path = p.aoi_binary_paths[year][class_label]
                                    hb.raster_calculator_flex(p.lulc_simplified_paths[year], lambda x: np.where(x == int(p.lulc_correspondence_dict['dst_labels_to_ids'][class_label]), 1, 0), output_path=output_path)

                else:
                    for year in p.base_years:

                        p.base_data_binary_paths[year] = {}
                        p.aoi_binary_paths[year] = {}
                        p.binary_paths[year] = {}

                        for class_label in p.all_class_labels:



                            search_path = os.path.join('lulc', p.lulc_src_label, p.lulc_simplification_label, 'binaries', str(year), binary_filename_start + str(year) + '_' + class_label +  '.tif')
                            found_path = hb.get_first_extant_path(search_path, [p.fine_processed_inputs_dir, p.input_dir, p.base_data_dir])
                            p.base_data_binary_paths[year][class_label] = found_path
                            p.aoi_binary_paths[year][class_label] = found_path
                            p.binary_paths[year][class_label] = found_path


                            if not hb.path_exists(p.aoi_binary_paths[year][class_label]):
                                hb.create_directories(p.aoi_binary_paths[year][class_label])
                                output_path = p.aoi_binary_paths[year][class_label]
                                hb.raster_calculator_flex(p.lulc_simplified_paths[year], lambda x: np.where(x == int(p.lulc_correspondence_dict['dst_labels_to_ids'][class_label]), 1, 0), output_path=output_path)


def generated_kernels(p):
    """Fast function that creates several tiny geotiffs of gaussian-like kernels for later use in ffn_convolve."""

    if p.run_this:
        starting_value = 1.0
        for halflife in p.gaussian_sigmas_to_test:
            filename = 'gaussian_' + str(halflife) + '.tif'
            kernel_path = os.path.join(p.cur_dir, filename)
            hb.create_directories(kernel_path)
            if not os.path.exists(kernel_path):
                radius = int(halflife * 9.0)
                kernel_array = seals_utils.get_array_from_two_dim_first_order_kernel_function(radius, starting_value, halflife)

                # Note the silly choice to save these kernels as geotiffs, which means they need spatial extents, which is BONKERS.
                hb.save_array_as_geotiff(kernel_array, kernel_path, p.base_year_lulc_path, n_cols_override=kernel_array.shape[1], n_rows_override=kernel_array.shape[0], data_type=7, ndv=-9999.0, compress=True)



def lulc_convolutions(p):
    # Convolve the lulc_binaries
    # Might want to promote to base_data to precache this for performance
    # NOTE THAT these follow different readwrite data than the other lulcs.
    # They do not write the convolutions if they exist in the base data
    # even if it's for a non global run. This is because there's
    # not much valu ein having them.

    p.lulc_simplified_convolution_paths = {}
    if True:
    # if hb.path_exists(p.regressors_starting_values_path) or True:
        # p.starting_coefficients_df = pd.read_csv(p.regressors_starting_values_path)

        parallel_iterable = []

        # if nyi:
        #     years_to_convolve = list(set([p.training_start_year, p.training_end_year, p.base_year]))
        # else:
        #     years_to_convolve = [p.base_year]


        if p.years_to_convolve_override is not None:
            years_to_convolve = p.years_to_convolve_override
        else:
            years_to_convolve = [p.key_base_year]
        # for c, row in p.starting_coefficients_df:
        for c, label in enumerate(p.all_class_labels):
            for sigma in p.gaussian_sigmas_to_test:
                for year in years_to_convolve:

                    # label = p.all_class_indices[c]
                    current_convolution_name = 'convolution_'+p.lulc_src_label+'_'+p.lulc_simplification_label+'_'+str(year)+'_' + str(label) + '_gaussian_' + str(sigma)

                    filename = 'gaussian_' + str(sigma) + '.tif'
                    kernel_path = os.path.join(p.fine_processed_inputs_dir, 'generated_kernels', filename)

                    # current_bulk_convolution_path = current_convolution_path.replace(hb.PRIMARY_DRIVE, hb.EXTERNAL_BULK_DATA_DRIVE)
                    # current_input_binary_path = os.path.join(p.base_data_dir, 'lulc', 'esa', p.lulc_simplification_label, 'binaries', str(label), 'lulc_esa_' + p.lulc_simplification_label + '_' + str(p.base_year) + '_class_' + str(class_id) + '_binary.tif')
                    current_file_root = 'binary_'+p.lulc_src_label+'_'+p.lulc_simplification_label+'_'+str(year)+'_' + str(label)
                    current_input_binary_path = os.path.join(p.fine_processed_inputs_dir, 'lulc', p.lulc_src_label, p.lulc_simplification_label, 'binaries', str(year), 'binary_'+p.lulc_src_label+'_'+p.lulc_simplification_label+'_'+str(year)+'_' + str(label)+'.tif')
                    # current_input_binary_path = p.lulc_simplified_binary_paths[current_file_root]

                    # First, define where the file should be created
                    current_convolution_ref_path = os.path.join('lulc', p.lulc_src_label,  p.lulc_simplification_label, 'convolutions', str(year), 'convolution_'+p.lulc_src_label+'_'+p.lulc_simplification_label+'_'+str(year)+'_' + str(label) + '_gaussian_' + str(sigma) + '.tif')
                    current_convolution_path = p.get_path(current_convolution_ref_path, raise_error_if_fail=False, verbose=True)

                    # TRICKY, the only way to get this to work without using the _get_first_extant as above was to have a second pass here if it didn't find one.
                    # INTERPRETATION, we are effectivly adding fine_processed_input_dir as a psuedo cannonical path                    
                    if not hb.path_exists(current_convolution_path):
                        current_convolution_path = os.path.join(p.fine_processed_inputs_dir, current_convolution_ref_path)
                    # Store this path in dictionary
                    p.lulc_simplified_convolution_paths[current_convolution_name] = current_convolution_path

                    # Then check if it exists and add to parallel processing if needed
                    if not os.path.exists(p.lulc_simplified_convolution_paths[current_convolution_name]):
                        
                        # A little awkward, but here i don't follow the full ref-path approach because in this project it is in a seals folder wrapper, but i don't really want to have that in the base data cause this lulc is not seals specific.
                        current_convolution_path = os.path.join(p.fine_processed_inputs_dir, 'lulc', p.lulc_src_label, p.lulc_simplification_label, 'convolutions', str(year), 'convolution_'+p.lulc_src_label+'_'+p.lulc_simplification_label+'_'+str(year)+'_' + str(label) + '_gaussian_' + str(sigma) + '.tif')
                        
                        hb.log(' Starting FFT Gaussian (in parallel) on ' + current_input_binary_path + 
                               ' and saving to ' + p.lulc_simplified_convolution_paths[current_convolution_name])
                        parallel_iterable.append([current_input_binary_path, kernel_path, 
                                                  p.lulc_simplified_convolution_paths[current_convolution_name], -9999.0, True])

        if len(parallel_iterable) > 0 and p.run_this:
            num_workers = max(min(multiprocessing.cpu_count() - 1, len(parallel_iterable)), 1)
            worker_pool = multiprocessing.Pool(num_workers)  # NOTE, worker pool and results are LOCAL variabes so that they aren't pickled when we pass the project object.
            result = worker_pool.starmap_async(seals_utils.fft_gaussian, parallel_iterable)
            finished_results = []
            for i in result.get():
                finished_results.append(i)
                del i
            worker_pool.close()
            worker_pool.join()



def local_data_regressors_starting_values(p):
    """TODOO Note the very confusing partial duplication with the regressors_starting_values task defined above. THIS task is the one that is used in calibration."""
    p.local_data_regressors_starting_values_path = os.path.join(p.cur_dir, 'local_data_regressors_starting_values.csv')
    if p.run_this:
        if not hb.path_exists(p.local_data_regressors_starting_values_path):

            column_headers = ['spatial_regressor_name', 'data_location', 'type']
            column_headers.extend(['class_' + str(i) for i in p.changing_class_labels])

            df_input_2d_list = []

            # Write the default starting coefficients
            # TODOO Note that I left out a possible optimization because here
            # (and more importantly in the seals_process_coarse_timeseries files) is always

            # rebuilds the global layers even if global.

            # Set Multiplicative (constraint) coefficients
            for c, label in enumerate(p.all_class_labels):
                # TODOO RENAME Everything Full and Simp instead of ESA and seals7_simplified_mosaic_is_natural.... This will generalize to any dataset.
                base_data_path = os.path.join(p.base_data_dir, 'lulc', p.lulc_src_label,  p.lulc_simplification_label, 'binaries', str(p.key_base_year), 'binary_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + str(p.key_base_year) + '_' + str(p.all_class_labels[c]) + '.tif')

                extant_path =  os.path.join(p.fine_processed_inputs_dir, 'lulc', p.lulc_src_label,  p.lulc_simplification_label, 'binaries', str(p.key_base_year), 'binary_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + str(p.key_base_year) + '_' + str(p.all_class_labels[c]) + '.tif')

                row = [label + '_presence_constraint', extant_path,
                       'multiplicative'] + \
                      [0 if i == p.all_class_labels[c] or p.all_class_labels[c] in [1, 6, 7] else 1 for i in p.changing_class_indices]
                df_input_2d_list.append(row)

            # Set additive coefficients
            # for class binaries
            for c, label in enumerate(p.all_class_labels):
                base_data_path = os.path.join(p.base_data_dir, 'lulc', p.lulc_src_label,  p.lulc_simplification_label, 'binaries', str(p.key_base_year), 'binary_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + str(p.key_base_year) + '_' + str(p.all_class_labels[c]) + '.tif')

                extant_path = os.path.join(p.fine_processed_inputs_dir, 'lulc', p.lulc_src_label,  p.lulc_simplification_label, 'binaries', str(p.key_base_year), 'binary_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + str(p.key_base_year) + '_' + str(p.all_class_labels[c]) + '.tif')


                row = [label + '_presence',
                       extant_path,
                       'additive'] + [0 if i == p.all_class_indices[c] else 0 for i in p.changing_class_indices]
                df_input_2d_list.append(row)

            for sigma in p.gaussian_sigmas_to_test:
                # Prior assumed for class convolutions
                # NOTE:
                #     p.class_labels = ['urban', 'cropland', 'grassland', 'forest', 'nonforestnatural',]
                #     p.nonchanging_class_labels = ['water', 'barren_and_other']
                #     p.changing_class_indices = [1, 2, 3, 4, 5]  # These are the indices of classes THAT CAN EXPAND/CONTRACT
                #     p.nonchanging_class_indices = [6, 7]  # These add other lulc classes that might have an effect on LUC but cannot change themselves (e.g. water, barren)
                #     p.changing_class_indices = p.class_labels + p.nonchanging_class_labels

                change_class_adjacency_effects = [
                    [10, 5, 1, 1, 1],
                    [1, 10, 1, 1, 1],
                    [1, 1, 10, 1, 1],
                    [1, 1, 1, 10, 1],
                    [1, 1, 1, 1, 10],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                ]

                # NZ_brazil patch (2026-05-31): iterate all_class_labels (7) to add
                # water/other gaussian regressors so the calibration matches Justin's
                # bundled CSV structure. The 7x5 change_class_adjacency_effects matrix
                # above already has rows for c=5 (water) and c=6 (other) = [1,1,1,1,1].
                for c, label in enumerate(p.all_class_labels):
                    base_data_path = os.path.join(p.base_data_dir, 'lulc', p.lulc_src_label,  p.lulc_simplification_label, 'convolutions', str(p.key_base_year), 'convolution_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + str(p.key_base_year) + '_' + str(p.all_class_labels[c]) + '_gaussian_' + str(sigma) + '.tif')

                    extant_path = os.path.join(p.fine_processed_inputs_dir, 'lulc', p.lulc_src_label,  p.lulc_simplification_label, 'convolutions', str(p.key_base_year), 'convolution_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + str(p.key_base_year) + '_' + str(p.all_class_labels[c]) + '_gaussian_' + str(sigma) + '.tif')

                    row = [label + '_gaussian_' + str(sigma),
                           extant_path,
                           'gaussian_' + str(sigma)] + [change_class_adjacency_effects[c][cc] * (1.0 / float(sigma)) for cc, label in enumerate(p.changing_class_labels)]
                    df_input_2d_list.append(row)

                # # for class convolutions of sigma 5, set to zero except for diagonal (self edge expansion)
                # for c, label in enumerate(p.regression_input_class_labels):
                #     row = [label + '_gaussian_5', os.path.join(
                #         p.base_data_dir, 'lulc', 'esa',  p.lulc_simplification_label, 'convolutions' + lulc_alternate_reclassification_string,  str(p.key_base_year), 'class_' + str(p.all_class_indices[c]) + '_gaussian_' + str(5) + '.tif'),
                #            'additive'] + [1 if i == p.all_class_indices[c] else 0 for i in p.all_class_indices]
                #     df_input_2d_list.append(row)

            # for all static variables, set to zero, except for as a hack one of them so that the it is edefined everyone.
            for static_regressor_label, path in p.static_regressor_paths.items():
                # base_data_path = os.path.join(p.base_data_dir, 'lulc', 'esa', 'seals7', 'binaries' + lulc_alternate_reclassification_string, str(p.key_base_year), 'lulc_esa_seals7_' + str(p.base_year) + '_class_' + str(p.changing_class_indices[c]) + '_binary.tif')
                # local_data_path = os.path.join(p.fine_processed_inputs_dir, p.aoi_label, 'lulc', 'esa', 'seals7', 'binaries' + lulc_alternate_reclassification_string, str(p.base_year), 'lulc_esa_seals7_' + str(p.base_year) + '_class_' + str(p.all_class_indicesall_class_indices[c]) + '_binary.tif')
                # extant_path = p.DataRef(local_data_path, p.fine_processed_inputs_dir).path

                row = [static_regressor_label, path,
                       'additive'] + [1 if static_regressor_label == 'soil_organic_content' else 1 for i in p.changing_class_indices]
                df_input_2d_list.append(row)

            df = pd.DataFrame(df_input_2d_list, columns=column_headers)
            df.set_index('spatial_regressor_name', inplace=True)

            df.to_csv(p.local_data_regressors_starting_values_path)


def prepare_global_lulc_DEPRECATED(p):
    """For the purposes of calibration, create change-matrices for each coarse grid-cell based on two observed ESA lulc maps.
    Does something similar to calc_observed_lulc_change.

    This function is slow and is not necessary for the actual allocation step (which uses a zone-level calc_observed_lulc_change task.
    However, it may be useful for things with larger zones, eg AEZs. Or, for just visualizing overall results.
    """




    if p.run_this:

        # from hazelbean.calculation_core import cython_functions
        from hazelbean.calculation_core.cython_functions import \
            calc_change_matrix_of_two_int_arrays

        t1 = hb.ArrayFrame(p.global_lulc_t1_path)
        t2 = hb.ArrayFrame(p.global_lulc_t2_path)
        t3 = hb.ArrayFrame(p.global_lulc_t3_path)  # Currently unused but could be for validation.
        p.coarse_match = hb.ArrayFrame(p.global_ha_per_cell_15m_path)

        p.global_ha_per_cell_15m = hb.ArrayFrame(p.global_ha_per_cell_15m_path)
        p.coarse_match = hb.ArrayFrame(p.global_ha_per_cell_15m_path)

        fine_cells_per_coarse_cell = round((p.global_ha_per_cell_15m.cell_size / t1.cell_size) ** 2)
        aspect_ratio = t1.num_cols / p.coarse_match.num_cols

        output_arrays = np.zeros((len(p.classes_that_might_change), p.coarse_match.shape[0], p.coarse_match.shape[1]))

        for r in range(p.coarse_match.num_rows):
            hb.log('Processing observed change row', r, ' to calculate global LULC change.')
            for c in range(p.coarse_match.num_cols):

                t1_subarray = t1.data[int(r * aspect_ratio): int((r + 1) * aspect_ratio), int(c * aspect_ratio): int((c + 1) * aspect_ratio)]
                t2_subarray = t2.data[int(r * aspect_ratio): int((r + 1) * aspect_ratio), int(c * aspect_ratio): int((c + 1) * aspect_ratio)]

                ha_per_coarse_cell_this_subarray = p.global_ha_per_cell_15m.data[r, c]

                # LIMITATION: Currently I do not use the full change matrix and instead summarize it down to a net change vector.
                change_matrix, counters = calc_change_matrix_of_two_int_arrays(t1_subarray.astype(np.int), t2_subarray.astype(np.int), p.classes_that_might_change)

                full_change_matrix = np.zeros((len(p.classes_that_might_change), len(p.classes_that_might_change)))
                vector = seals_utils.calc_change_vector_of_change_matrix(change_matrix)

                ha_per_cell_this_subarray = p.global_ha_per_cell_15m.data[r, c] / fine_cells_per_coarse_cell

                if vector:
                    for i in p.classes_that_might_change:
                        output_arrays[i - 1, r, c] = vector[i - 1] * ha_per_cell_this_subarray
                else:
                    output_arrays[i, r, c] = 0.0

        for c, i in enumerate(p.classes_that_might_change):
            output_path = os.path.join(p.cur_dir, str(i) + '_observed_change.tif')
            hb.save_array_as_geotiff(output_arrays[c], output_path, p.coarse_match.path)

        # Unused but potential for future visualization.
        numpy_output_path = os.path.join(p.cur_dir, 'change_matrices.npy')
        hb.save_array_as_npy(output_arrays, numpy_output_path)
        # change_3d = hb.load_npy_as_array(numpy_output_path)
def process_global_gpkg(p):


    # p.gtap_aez_10d_vector_path = os.path.join(p.cur_dir, 'gtap_aez_10d.gpkg')
    # p.gtap_aez_10d_10s_raster_path = os.path.join(p.cur_dir, 'gtap_aez_10d_10s.tif')

    p.gtap_aez_10d_vector_path = p.InputPath(p.cur_dir, 'gtap_aez_10d.gpkg', base_data_extension_dirs=None)
    p.gtap_aez_10d_10s_raster_path = p.InputPath(p.cur_dir, 'gtap_aez_10d_10s.tif', base_data_extension_dirs=None)

    if p.run_this:
        hb.create_directories(p.cur_dir)
        if not hb.path_exists(p.gtap_aez_10d_vector_path):

            # Merge in graticules
            gdf_input = gpd.read_file(p.gtap_aez_input_vector_path)



            p.gtap_aez_10d_intersect_path = os.path.join(p.cur_dir, 'gtap_aez_10d_intersect.gpkg')
            if not hb.path_exists(p.gtap_aez_10d_intersect_path):
                graticule_input = gpd.read_file(p.graticules_input_path)
                gtap_aez_10d_intersect = gpd.overlay(gdf_input, graticule_input, how='intersection')
                # Dumb fiona error fix here based on error message of:
                #         Wrong field type for fid              --- 2020-03-09 12:32:07,223 --- fiona._env ERROR
                #         Traceback (most recent call last):
                #         File "fiona/ogrext.pyx", line 1167, in fiona.ogrext.WritingSession.start
                #         File "fiona/_err.pyx", line 246, in fiona._err.exc_wrap_int
                #         fiona._err.CPLE_AppDefinedError: Wrong field type for fid
                #         File "fiona/ogrext.pyx", line 1173, in fiona.ogrext.WritingSession.start
                #         fiona.errors.SchemaError: Wrong field type for fid
                # One way to get rid is as below, rewriting column types. BUT this failed if clashing FIDS not unique. Thus, easies was to just drop.
                # if 'fid' in gtap_aez_10d_intersect.columns:
                #     if gtap_aez_10d_intersect['fid'].dtype != np.int64:
                #         gtap_aez_10d_intersect['fid'] = gtap_aez_10d_intersect['fid'].astype(np.int64)

                gtap_aez_10d_intersect = gtap_aez_10d_intersect.drop('fid', axis=1)
                hb.create_directories(p.cur_dir)
                gtap_aez_10d_intersect.to_file(p.gtap_aez_10d_intersect_path, driver='GPKG')


            hb.make_vector_path_global_pyramid(p.gtap_aez_10d_intersect_path, output_path=p.gtap_aez_10d_vector_path, pyramid_index_columns=p.pyramid_index_columns, drop_columns=False,
            clean_temporary_files=False, verbose=False)


        if not os.path.exists(p.gtap_aez_10d_10s_raster_path.get_path()):
            # TODOO Note that I skipped a coastline-membership vector calculation to have all_touched = True just on coastlines.
            hb.convert_polygons_to_id_raster(p.gtap_aez_10d_vector_path.__str__(), p.gtap_aez_10d_10s_raster_path.get_path(), p.match_10s_path,
                                             id_column_label='pyramid_id', data_type=5, ndv=-9999, all_touched=None, compress=True)

        gdf = gpd.read_file(str(p.gtap_aez_10d_vector_path))
        # unique_values = list(np.unique(gdf['GTAP19GTAP']))
        # ascii_fixed_unique_values = [i.replace('_', '%') for i in unique_values]
        # unique_sorted_values = [i.replace('%', '_') for i in sorted(ascii_fixed_unique_values)]
        # p.gtap_zones_to_ids = {v: c + 1 for c, v in enumerate(unique_sorted_values)}


def calc_observed_lulc_change(passed_p=None):
    ### DEPRECETAED BECAUSE PATHS LOGIC not uptaded
    # MOVING LOGIC TO UTILS, then remove.
    if passed_p is None:
        global p
    else:
        p = passed_p


    # p.current_region_id = p.region_ids
    # p.current_bb = p.region_bounding_boxes[p.current_region_id]
    #
    p.lulc_t1_path = r"D:\OneDrive\Projects\cge\seals\model_base_data\lulc_esa\simplified\lulc_esa_simplified_2000.tif"
    p.lulc_t2_path = r"D:\OneDrive\Projects\cge\seals\model_base_data\lulc_esa\simplified\lulc_esa_simplified_2010.tif"
    p.lulc_t3_path = r"D:\OneDrive\Projects\cge\seals\model_base_data\lulc_esa\simplified\lulc_esa_simplified_2014.tif"
    p.lulc_paths = [p.lulc_t1_path, p.lulc_t2_path, p.lulc_t3_path]

    p.ha_per_cell_15m_path = os.path.join(p.cur_dir, 'ha_per_cell_900sec.tif')

    p.ha_per_cell_15m = hb.ArrayFrame(p.global_ha_per_cell_15m_path)

    # TODOO, current problems: Change vector method needs to be expanded to Change matrix, full from-to relationships
    # but when doing from-to, that only works when doing observed time-period validation. What would be the assumption for going into
    # the future? Possibly attempt to match prior change matrices, but only as a slight increase in probability? Secondly, why is my
    # search algorithm not itself finding the from-to relationships just by minimizing difference? Basically, need to take seriously deallocation.

    full_change_matrix_no_diagonal_path = os.path.join(p.cur_dir, 'full_change_matrix_no_diagonal.tif')
    if p.run_this:
        from hazelbean.calculation_core.cython_functions import \
            calc_change_matrix_of_two_int_arrays

        # if p.run_this and not os.path.exists(full_change_matrix_no_diagonal_path):
        # Clip ha_per_cell and use it as the match
        ha_per_cell = hb.load_geotiff_chunk_by_cr_size(p.global_ha_per_cell_15m_path, p.coarse_blocks_list)

        # Clip all 30km change paths, then just use the last one to set the propoer (coarse) extent of the lulc.
        lulc_t1 = hb.load_geotiff_chunk_by_cr_size(p.lulc_t1_path, p.fine_blocks_list)
        lulc_t3 = hb.load_geotiff_chunk_by_cr_size(p.lulc_t3_path, p.fine_blocks_list)

        # # Clip all 30km change paths, then just use the last one to set the propoer (coarse) extent of the lulc.
        # for c, path in enumerate(p.global_lulc_paths):
        #     hb.load_geotiff_chunk_by_cr_size(path, p.fine_blocks_list, output_path=p.lulc_paths[c])

        lulc_afs = [hb.ArrayFrame(path) for path in p.lulc_paths]

        fine_cells_per_coarse_cell = round((p.ha_per_cell_15m.cell_size / lulc_afs[0].cell_size) ** 2)
        aspect_ratio = int(lulc_afs[0].num_cols / p.coarse_match.num_cols)

        net_change_output_arrays = np.zeros((len(p.classes_that_might_change), p.coarse_match.shape[0], p.coarse_match.shape[1]))

        full_change_matrix = np.zeros((len(p.classes_that_might_change * p.coarse_match.n_rows), len(p.classes_that_might_change) * p.coarse_match.n_cols))
        full_change_matrix_no_diagonal = np.zeros((len(p.classes_that_might_change * p.coarse_match.n_rows), len(p.classes_that_might_change) * p.coarse_match.n_cols))
        for r in range(p.coarse_match.num_rows):
            for c in range(p.coarse_match.num_cols):

                t1_subarray = lulc_afs[0].data[int(r * aspect_ratio): int((r + 1) * aspect_ratio), int(c * aspect_ratio): int((c + 1) * aspect_ratio)]
                t2_subarray = lulc_afs[1].data[int(r * aspect_ratio): int((r + 1) * aspect_ratio), int(c * aspect_ratio): int((c + 1) * aspect_ratio)]
                ha_per_coarse_cell_this_subarray = p.ha_per_cell_15m.data[r, c]

                change_matrix, counters = calc_change_matrix_of_two_int_arrays(t1_subarray.astype(np.int), t2_subarray.astype(np.int), p.classes_that_might_change)
                vector = seals_utils.calc_change_vector_of_change_matrix(change_matrix)

                ha_per_cell_this_subarray = p.ha_per_cell_15m.data[r, c] / fine_cells_per_coarse_cell

                if vector:
                    for i in p.classes_that_might_change:
                        net_change_output_arrays[i - 1, r, c] = vector[i - 1] * ha_per_cell_this_subarray
                else:
                    net_change_output_arrays[i, r, c] = 0.0

                n_classes = len(p.classes_that_might_change)
                full_change_matrix[r * n_classes: (r + 1) * n_classes, c * n_classes: (c + 1) * n_classes] = change_matrix

                # Fill diagonal with zeros.
                for i in range(n_classes):
                    change_matrix[i, i] = 0

                full_change_matrix_no_diagonal[r * n_classes: (r + 1) * n_classes, c * n_classes: (c + 1) * n_classes] = change_matrix

        for c, i in enumerate(p.classes_that_might_change):
            current_net_change_array_path = os.path.join(p.cur_dir, str(i) + '_observed_change.tif')
            hb.save_array_as_geotiff(net_change_output_arrays[c], current_net_change_array_path, p.coarse_match.path)

        p.projected_cooarse_change_files = hb.list_filtered_paths_nonrecursively(p.projected_coarse_change_dir, include_extensions='.tif')
        for path in p.projected_cooarse_change_files:
            file_front_int = os.path.split(path)[1].split('_')[0]
            current_net_change_array_path = os.path.join(p.cur_dir, str(file_front_int) + '_projected_change.tif')

            # TODO Get rid of all this wasteful writing.
            hb.load_geotiff_chunk_by_bb(path, p.coarse_blocks_list, output_path=current_net_change_array_path)
        # for c, i in enumerate(p.classes_that_might_change):
        #     projected_change_global_path = os.path.join(p.projected_coarse_change_dir, str(i) )
        #     current_net_change_array_path = os.path.join(p.cur_dir, str(i) + '_projected_change.tif')
        #     # hb.save_array_as_geotiff(net_change_output_arrays[c], current_net_change_array_path, p.coarse_match.path)
        #     hb.load_geotiff_chunk_by_bb(p.global_ha_per_cell_15m_path, p.coarse_blocks_list, output_path=current_net_change_array_path)


        # full_change_matrix_path = os.path.join(p.cur_dir, 'full_change_matrix.tif')
        # hb.save_array_as_geotiff(full_change_matrix, full_change_matrix_path, p.coarse_match.path, n_rows=full_change_matrix.shape[1], n_cols=full_change_matrix.shape[1])
        # full_change_matrix_no_diagonal_path = os.path.join(p.cur_dir, 'full_change_matrix_no_diagonal.tif')
        # hb.save_array_as_geotiff(full_change_matrix_no_diagonal, full_change_matrix_no_diagonal_path, p.coarse_match.path, n_rows=full_change_matrix_no_diagonal.shape[1], n_cols=full_change_matrix_no_diagonal.shape[1])


    p.ha_per_cell_15m = None
