import os
import hazelbean as hb
import numpy as np
import pandas as pd
import multiprocessing
from matplotlib import pyplot as plt
import geopandas as gpd

import seals_utils

L = hb.get_logger()

def regressors_starting_values(p):
    """TODOO Note the very confusing partial duplication with the regressors_starting_values task defined above. THIS task is the one that is used in calibration
    Create an xls with starting-guess parameters for the SEALS calibration run. This identifies where the
    input data (and generated base data) is stored, parsed to the class-simpliicaiton scheme used."""

    p.regressors_starting_values_path = os.path.join(p.cur_dir, 'regressors_starting_values.xlsx')
    if p.run_this:
        if not hb.path_exists(p.regressors_starting_values_path):

            column_headers = ['spatial_regressor_name', 'data_location', 'type']
            column_headers.extend(['class_' + str(i) for i in p.class_labels])

            df_input_2d_list = []

            # HACK, TODOO this should be tied to if it's a magpie run, or better yet, a more robust specification of the reclassification for the simplification.
            # lulc_alternate_reclassification_string = '_mosaic_is_natural'
            lulc_alternate_reclassification_string = ''

            # Write the default starting coefficients

            # Set Multiplicative (constraint) coefficients
            for c, label in enumerate(p.regression_input_class_labels):
                row = [label + '_presence_constraint', os.path.join(
                    p.base_data_dir, 'lulc', 'esa', 'seals7', 'binaries' + lulc_alternate_reclassification_string, str(p.training_start_year), 'lulc_esa_seals7_' + str(p.training_start_year) + '_class_' + str(p.regression_input_class_indices[c]) + '_binary.tif'),
                       'multiplicative'] + \
                      [0 if i == p.regression_input_class_indices[c] or p.regression_input_class_indices[c] in [1, 6, 7] else 1 for i in p.class_indices]
                df_input_2d_list.append(row)

            # Set additive coefficients
            # for class binaries
            for c, label in enumerate(p.regression_input_class_labels):
                row = [label + '_presence', os.path.join(
                    p.base_data_dir, 'lulc', 'esa', 'seals7', 'binaries' + lulc_alternate_reclassification_string, str(p.training_start_year), 'lulcf_esa_seals7_' + str(p.training_start_year) + '_class_' + str(p.regression_input_class_indices[c]) + '_binary.tif'),
                       'additive'] + [0 if i == p.regression_input_class_indices[c] else 0 for i in p.class_indices]
                df_input_2d_list.append(row)

            for sigma in p.gaussian_sigmas_to_test:
                # Prior assumed for class convolutions
                # NOTE:
                #     p.class_labels = ['urban', 'cropland', 'grassland', 'forest', 'nonforestnatural',]
                #     p.nonchanging_class_labels = ['water', 'barren_and_other']
                #     p.class_indices = [1, 2, 3, 4, 5]  # These are the indices of classes THAT CAN EXPAND/CONTRACT
                #     p.nonchanging_class_indices = [6, 7]  # These add other lulc classes that might have an effect on LUC but cannot change themselves (e.g. water, barren)
                #     p.regression_input_class_labels = p.class_labels + p.nonchanging_class_labels

                change_class_adjacency_effects = [
                    [10, 5, 1, 1, 1],
                    [1, 10, 1, 1, 1],
                    [1, 1, 10, 1, 1],
                    [1, 1, 1, 10, 1],
                    [1, 1, 1, 1, 10],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                ]
                for c, label in enumerate(p.regression_input_class_labels):
                    row = [label + '_gaussian_' + str(sigma), os.path.join(
                        p.base_data_dir, 'lulc', 'esa', 'seals7', 'convolutions' + lulc_alternate_reclassification_string, str(p.training_start_year),
                        'class_' + str(p.regression_input_class_indices[c]) + '_gaussian_' + str(sigma) + '_convolution.tif'),
                           'gaussian_' + str(sigma)] + [change_class_adjacency_effects[c][cc] * (1.0 / float(sigma)) for cc, class_index in enumerate(p.class_indices)]

                    df_input_2d_list.append(row)


            # for all static variables, set to 1, except for as a hack one of them so that the it is edefined everyone.
            for static_regressor_label, path in p.static_regressor_paths.items():
                row = [static_regressor_label, path,
                       'additive'] + [1 if static_regressor_label == 'soil_organic_content' else 1 for i in p.class_indices]
                df_input_2d_list.append(row)

            df = pd.DataFrame(df_input_2d_list, columns=column_headers)
            df.set_index('spatial_regressor_name', inplace=True)

            df.to_excel(p.regressors_starting_values_path)


def generated_data(p):
    "Dummy task just to group things."
    pass

def aoi_vector(p):
    graft_dir = os.path.split(p.generated_data_dir)[1]
    path_verbosity = False
    if p.run_this:
        if isinstance(p.aoi, str):

            if p.aoi == 'global':
                p.aoi_path = p.DataRef(p.countries_iso3_path, graft_dir=graft_dir, verbose=path_verbosity).path
                p.aoi_label = 'global'
                p.bb_exact = hb.global_bounding_box
                p.bb = p.bb_exact
            elif isinstance(p.aoi, str):
                if len(p.aoi) == 3: # Then it might be an ISO3 code. For now, assume so.
                    generated_path = os.path.join(p.generated_data_dir, p.aoi, 'pyramids', 'aoi.gpkg')

                    p.aoi_path = p.DataRef(generated_path, graft_dir=graft_dir, verbose=path_verbosity).path
                    if not hb.path_exists(p.aoi_path):
                        hb.extract_features_in_shapefile_by_attribute(p.countries_iso3_path, p.aoi_path, 'iso3', p.aoi.upper())
                    p.aoi_label = p.aoi
                else:
                    p.aoi_path = p.DataRef(p.aoi, graft_dir=graft_dir).path
                    if not hb.path_exists(p.aoi_path):
                        p.aoi_label = hb.file_root(p.aoi_path)
                p.bb_exact = hb.get_bounding_box(p.aoi_path)
                p.bb = hb.get_pyramid_compatible_bb_from_vector_and_resolution(p.aoi_path, p.coarse_resolution_arcseconds)
        else:
            raise NameError('Unable to interpret p.aoi.')
    else:
        if p.aoi == 'global':
            p.aoi_path = p.DataRef(p.countries_iso3_path, graft_dir=graft_dir, verbose=path_verbosity).path

        elif isinstance(p.aoi, str):
            if len(p.aoi) == 3:  # Then it might be an ISO3 code. For now, assume so.
                generated_path = os.path.join(p.generated_data_dir, p.aoi, 'pyramids', 'aoi.gpkg')

                p.aoi_path = p.DataRef(generated_path, graft_dir=graft_dir, verbose=path_verbosity).path
            else:
                p.aoi_path = p.DataRef(p.aoi, graft_dir=graft_dir).path


        p.bb_exact = hb.get_bounding_box(p.aoi_path)
        p.bb = hb.get_pyramid_compatible_bb_from_vector_and_resolution(p.aoi_path, p.coarse_resolution_arcseconds)


def lulc_clip(p):

    graft_dir = os.path.split(p.generated_data_dir)[1]
    if p.aoi != 'global':

        p.lulc_paths = {}
        p.lulc_paths['base_year_lulc'] = p.base_year_lulc_path
        p.lulc_paths['training_start_year_lulc'] = p.training_start_year_lulc_path
        p.lulc_paths['training_end_year_lulc'] = p.training_end_year_lulc_path

        p.lulc_aoi_paths = {}

        for input_key, input_path in p.lulc_paths.items():
            filename = os.path.split(input_path)[1]
            generated_path = os.path.join(p.generated_data_dir, p.aoi, 'lulc', 'esa', filename)
            output_path = p.DataRef(generated_path, graft_dir).path
            hb.create_directories(output_path)
            if p.run_this:
                if not hb.path_exists(output_path):
                    hb.clip_raster_by_bb(input_path, p.bb, output_path)

            p.lulc_aoi_paths[input_key] = output_path
            p.lulc_paths[input_key] = output_path # Stupid time-saving fix.


        p.aoi_ha_per_cell_fine_path = os.path.join(p.generated_data_dir, p.aoi, 'pyramids', 'aoi_ha_per_cell_fine.tif')
        if not hb.path_exists(p.aoi_ha_per_cell_fine_path):
            hb.clip_raster_by_bb(p.ha_per_cell_10sec_path, p.bb, p.aoi_ha_per_cell_fine_path)

        p.aoi_ha_per_cell_coarse_path = os.path.join(p.generated_data_dir, p.aoi, 'pyramids', 'aoi_ha_per_cell_coarse.tif')
        if not hb.path_exists(p.aoi_ha_per_cell_coarse_path):
            hb.clip_raster_by_bb(p.ha_per_cell_900sec_path, p.bb, p.aoi_ha_per_cell_coarse_path)

    else:
        p.aoi_ha_per_cell_coarse_path = p.ha_per_cell_paths[p.coarse_resolution_arcseconds]
        p.aoi_ha_per_cell_fine_path = p.ha_per_cell_paths[p.fine_resolution_arcseconds]

        p.lulc_paths = {}
        p.lulc_paths['base_year_lulc'] = p.base_year_lulc_path
        p.lulc_paths['training_start_year_lulc'] = p.training_start_year_lulc_path
        p.lulc_paths['training_end_year_lulc'] = p.training_end_year_lulc_path


def lulc_simplifications(p):
    p.lulc_reclassification_rules = dict(zip(p.lulc_simplification_remap.keys(), [i[0] for i in p.lulc_simplification_remap.values()]))

    p.lulc_simplified_paths = {}

    graft_dir = os.path.split(p.generated_data_dir)[1]

    for input_key, input_path in p.lulc_paths.items():
        year = hb.file_root(input_path).split('_')[-1]
        filename = 'lulc_esa_' + p.lulc_simplification_label + '_' + str(year) + '.tif'
        generated_path = os.path.join(p.generated_data_dir, p.aoi, 'lulc', 'esa',  p.lulc_simplification_label, filename)

        output_path = p.DataRef(generated_path, graft_dir).path
        hb.create_directories(output_path)
        if not hb.path_exists(output_path):
            p.L.info('Reclassifying ' + input_path + ' into ' + output_path + ' with rules ' + str(p.lulc_reclassification_rules))
            hb.reclassify_flex(input_path, p.lulc_reclassification_rules, output_path, output_data_type=1)
        p.lulc_simplified_paths[os.path.splitext(filename)[0]] = output_path



def lulc_binaries(p):
    p.lulc_simplified_binary_paths = {}

    graft_dir = os.path.split(p.generated_data_dir)[1]

    for input_key, input_path in p.lulc_simplified_paths.items():
        year = hb.file_root(input_path).split('_')[-1]

        for class_id in np.unique(list(p.lulc_reclassification_rules.values())):
            filename = 'lulc_esa_' + p.lulc_simplification_label + '_' + year + '_class_' + str(class_id) + '_binary.tif'

            generated_path = os.path.join(p.generated_data_dir, p.aoi, 'lulc', 'esa', p.lulc_simplification_label, 'binaries', str(year), filename)
            output_path = p.DataRef(generated_path, graft_dir).path
            hb.create_directories(output_path)
            if not os.path.exists(output_path):
                if p.run_this:
                    hb.raster_calculator_af_flex(input_path, lambda x: np.where(x == int(class_id), 1, 0), output_path=output_path)
            p.lulc_simplified_binary_paths[os.path.splitext(filename)[0]] = output_path


def generated_kernels(p):
    """Fast function that creates several tiny geotiffs of gaussian-like kernels for later use in ffn_convolve."""
    p.match_300sec_path = os.path.join(p.base_data_dir, 'pyramids', "ha_per_cell_300sec.tif")
    p.kernel_halflives = [1, 2, 3, 4, 5, 7, 9, 12, 15, 30]
    if p.run_this:
        starting_value = 1.0
        for halflife in p.kernel_halflives:
            filename = 'gaussian_' + str(halflife) + '.tif'
            kernel_path = os.path.join(p.generated_data_dir, 'kernels', filename)
            hb.create_directories(kernel_path)
            if not os.path.exists(kernel_path):
                radius = int(halflife * 9.0)
                kernel_array = seals_utils.get_array_from_two_dim_first_order_kernel_function(radius, starting_value, halflife)
                hb.save_array_as_geotiff(kernel_array, kernel_path, p.match_300sec_path, n_cols_override=kernel_array.shape[1], n_rows_override=kernel_array.shape[0], data_type=7, ndv=-9999.0, compress=True)



def lulc_convolutions(p):
    p.lulc_simplified_convolution_paths = {}
    if hb.path_exists(p.regressors_starting_values_path):
        p.starting_coefficients_df = pd.read_excel(p.regressors_starting_values_path)

        parallel_iterable = []

        years_to_convolve = [p.training_start_year, p.base_year]
        # for c, row in p.starting_coefficients_df:
        for c, class_index in enumerate(p.regression_input_class_indices):

            for sigma in p.gaussian_sigmas_to_test:

                for year in years_to_convolve:

                # if 'gaussian' in p.starting_coefficients_df['type'].values[c]:
                #     sigma = int(p.starting_coefficients_df['type'].values[c].split('_')[-1])

                    label = p.regression_input_class_labels[c]
                    current_convolution_name = 'class_' + str(class_index) + '_gaussian_' + str(sigma) + '_convolution'

                    filename = 'gaussian_' + str(sigma) + '.tif'
                    kernel_path = os.path.join(p.generated_data_dir, 'kernels', filename)

                    # current_bulk_convolution_path = current_convolution_path.replace(hb.PRIMARY_DRIVE, hb.EXTERNAL_BULK_DATA_DRIVE)
                    # current_input_binary_path = os.path.join(p.base_data_dir, 'lulc', 'esa', p.lulc_simplification_label, 'binaries', str(class_index), 'lulc_esa_' + p.lulc_simplification_label + '_' + str(p.base_year) + '_class_' + str(class_index) + '_binary.tif')
                    current_file_root = 'lulc_esa_' + p.lulc_simplification_label + '_' + str(year) + '_class_' + str(class_index) + '_binary'
                    current_input_binary_path = p.lulc_simplified_binary_paths[current_file_root]

                    current_convolution_path = os.path.join(p.generated_data_dir, p.aoi, 'lulc', 'esa', p.lulc_simplification_label, 'convolutions', str(year), 'class_' + str(class_index) + '_gaussian_' + str(sigma) + '_convolution.tif')
                    current_convolution_path = p.DataRef(current_convolution_path, p.generated_data_dir).path
                    p.lulc_simplified_convolution_paths[current_convolution_name] = current_convolution_path

                    # current_bulk_convolution_path = os.path.join(p.model_base_data_dir, 'convolutions', 'lulc_esa_simplified_' + str(year), current_convolution_name + '.tif')
                    # NOTE, fft_gaussian has to write to disk, which i think i have to embrace.
                    if not os.path.exists(current_convolution_path):

                        p.L.info('  Starting FFT Gaussian (in parallel) on ' + current_input_binary_path + ' and saving to ' + current_convolution_path)
                        parallel_iterable.append([current_input_binary_path, kernel_path, current_convolution_path, -9999.0, True])

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
    else:
        parallel_iterable = []

        # HACK Shortcut to not just list them all or generate before. Really it should be part of the unified input.
        p.kernel_paths = {}
        for sigma in p.gaussian_sigmas_to_test:
            p.kernel_paths[sigma] = os.path.join(p.generated_kernels_dir, 'gaussian_' + str(sigma) + '.tif')

        for sigma, path in p.regression_input_class_indices.items():
            for class_label in p.classes_to_convolve:
                current_convolution_name = 'class_' + str(class_label) + '_gaussian_' + str(sigma) + '_convolution'

                kernel_path = os.path.join(p.generated_kernels_dir, 'gaussian_' + str(sigma) + '.tif')
                current_convolution_path = os.path.join(p.cur_dir, current_convolution_name + '.tif')


                data_path = os.path.join(p.lulc_binaries_dir, 'lulc_esa_simplified_binary_2015_class_' + str(class_label) + '.tif')
                p.L.info('  Starting FFT Gaussian (in parallel) on ' + data_path)

                parallel_iterable.append([data_path, kernel_path, current_convolution_path, -9999.0, True])

                p.lulc_simplified_convolution_paths[current_convolution_name] = current_convolution_path
        hb.pp(parallel_iterable)

        if len(parallel_iterable) > 0 and p.run_this:
            num_workers = max(min(multiprocessing.cpu_count() - 1, len(parallel_iterable)), 1)
            worker_pool = multiprocessing.Pool(num_workers)  # NOTE, worker pool and results are LOCAL variabes so that they aren't pickled when we pass the project object.
            # result = worker_pool.map(seals_utils.fft_gaussian, parallel_iterable)
            result = worker_pool.starmap_async(seals_utils.fft_gaussian, parallel_iterable)
            finished_results = []
            for i in result.get():
                finished_results.append(i)
                del i
            worker_pool.close()
            worker_pool.join()

def local_data_regressors_starting_values(p):
    """TODOO Note the very confusing partial duplication with the regressors_starting_values task defined above. THIS task is the one that is used in calibration."""
    p.local_data_regressors_starting_values_path = os.path.join(p.cur_dir, 'local_data_regressors_starting_values.xlsx')
    if p.run_this:
        if not hb.path_exists(p.local_data_regressors_starting_values_path):

            column_headers = ['spatial_regressor_name', 'data_location', 'type']
            column_headers.extend(['class_' + str(i) for i in p.class_labels])

            df_input_2d_list = []

            # HACK, TODOO this should be tied to if it's a magpie run, or better yet, a more robust specification of the reclassification for the simplification.
            # lulc_alternate_reclassification_string = '_mosaic_is_natural'
            lulc_alternate_reclassification_string = ''

            # Write the default starting coefficients
            # TODOO Note that I left out a possible optimization because here
            # (and more importantly in the seals_process_luh2 files) is always
            # rebuilds the global layers even if global.

            # Set Multiplicative (constraint) coefficients
            for c, label in enumerate(p.regression_input_class_labels):
                base_data_path = os.path.join(p.base_data_dir, 'lulc', 'esa', 'seals7', 'binaries' + lulc_alternate_reclassification_string, str(p.training_start_year), 'lulc_esa_seals7_' + str(p.training_start_year) + '_class_' + str(p.regression_input_class_indices[c]) + '_binary.tif')
                local_data_path = os.path.join(p.generated_data_dir, p.aoi_label, 'lulc', 'esa', 'seals7', 'binaries' + lulc_alternate_reclassification_string, str(p.training_start_year), 'lulc_esa_seals7_' + str(p.training_start_year) + '_class_' + str(p.regression_input_class_indices[c]) + '_binary.tif')
                extant_path = p.DataRef(local_data_path, p.generated_data_dir).path

                row = [label + '_presence_constraint', extant_path,
                       'multiplicative'] + \
                      [0 if i == p.regression_input_class_indices[c] or p.regression_input_class_indices[c] in [1, 6, 7] else 1 for i in p.class_indices]
                df_input_2d_list.append(row)

            # Set additive coefficients
            # for class binaries
            for c, label in enumerate(p.regression_input_class_labels):
                base_data_path = os.path.join(p.base_data_dir, 'lulc', 'esa', 'seals7', 'binaries' + lulc_alternate_reclassification_string, str(p.training_start_year), 'lulc_esa_seals7_' + str(p.training_start_year) + '_class_' + str(p.regression_input_class_indices[c]) + '_binary.tif')
                local_data_path = os.path.join(p.generated_data_dir, p.aoi_label, 'lulc', 'esa', 'seals7', 'binaries' + lulc_alternate_reclassification_string, str(p.training_start_year), 'lulc_esa_seals7_' + str(p.training_start_year) + '_class_' + str(p.regression_input_class_indices[c]) + '_binary.tif')
                extant_path = p.DataRef(local_data_path, p.generated_data_dir).path

                row = [label + '_presence',
                       extant_path,
                       'additive'] + [0 if i == p.regression_input_class_indices[c] else 0 for i in p.class_indices]
                df_input_2d_list.append(row)

            for sigma in p.gaussian_sigmas_to_test:
                # Prior assumed for class convolutions
                # NOTE:
                #     p.class_labels = ['urban', 'cropland', 'grassland', 'forest', 'nonforestnatural',]
                #     p.nonchanging_class_labels = ['water', 'barren_and_other']
                #     p.class_indices = [1, 2, 3, 4, 5]  # These are the indices of classes THAT CAN EXPAND/CONTRACT
                #     p.nonchanging_class_indices = [6, 7]  # These add other lulc classes that might have an effect on LUC but cannot change themselves (e.g. water, barren)
                #     p.regression_input_class_labels = p.class_labels + p.nonchanging_class_labels

                change_class_adjacency_effects = [
                    [10, 5, 1, 1, 1],
                    [1, 10, 1, 1, 1],
                    [1, 1, 10, 1, 1],
                    [1, 1, 1, 10, 1],
                    [1, 1, 1, 1, 10],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                ]

                for c, label in enumerate(p.regression_input_class_labels):
                    base_data_path = os.path.join(p.base_data_dir, 'lulc', 'esa', 'seals7', 'convolutions' + lulc_alternate_reclassification_string, str(p.training_start_year), 'class_' + str(p.regression_input_class_indices[c]) + '_gaussian_' + str(sigma) + '_convolution.tif')
                    local_data_path = os.path.join(p.generated_data_dir, p.aoi_label, 'lulc', 'esa', 'seals7', 'convolutions' + lulc_alternate_reclassification_string, str(p.training_start_year), 'class_' + str(p.regression_input_class_indices[c]) + '_gaussian_' + str(sigma) + '_convolution.tif')
                    extant_path = p.DataRef(local_data_path, p.generated_data_dir).path

                    row = [label + '_gaussian_' + str(sigma),
                           extant_path,
                           'gaussian_' + str(sigma)] + [change_class_adjacency_effects[c][cc] * (1.0 / float(sigma)) for cc, class_index in enumerate(p.class_indices)]
                    df_input_2d_list.append(row)

                # # for class convolutions of sigma 5, set to zero except for diagonal (self edge expansion)
                # for c, label in enumerate(p.regression_input_class_labels):
                #     row = [label + '_gaussian_5', os.path.join(
                #         p.base_data_dir, 'lulc', 'esa', 'seals7', 'convolutions' + lulc_alternate_reclassification_string,  str(p.training_start_year), 'class_' + str(p.regression_input_class_indices[c]) + '_gaussian_' + str(5) + '_convolution.tif'),
                #            'additive'] + [1 if i == p.regression_input_class_indices[c] else 0 for i in p.class_indices]
                #     df_input_2d_list.append(row)

            # for all static variables, set to zero, except for as a hack one of them so that the it is edefined everyone.
            for static_regressor_label, path in p.static_regressor_paths.items():
                # base_data_path = os.path.join(p.base_data_dir, 'lulc', 'esa', 'seals7', 'binaries' + lulc_alternate_reclassification_string, str(p.training_start_year), 'lulc_esa_seals7_' + str(p.training_start_year) + '_class_' + str(p.regression_input_class_indices[c]) + '_binary.tif')
                # local_data_path = os.path.join(p.generated_data_dir, p.aoi_label, 'lulc', 'esa', 'seals7', 'binaries' + lulc_alternate_reclassification_string, str(p.training_start_year), 'lulc_esa_seals7_' + str(p.training_start_year) + '_class_' + str(p.regression_input_class_indices[c]) + '_binary.tif')
                # extant_path = p.DataRef(local_data_path, p.generated_data_dir).path

                row = [static_regressor_label, path,
                       'additive'] + [1 if static_regressor_label == 'soil_organic_content' else 1 for i in p.class_indices]
                df_input_2d_list.append(row)

            df = pd.DataFrame(df_input_2d_list, columns=column_headers)
            df.set_index('spatial_regressor_name', inplace=True)

            df.to_excel(p.local_data_regressors_starting_values_path)


def prepare_global_lulc(p):
    """For the purposes of calibration, create change-matrices for each coarse grid-cell based on two observed ESA lulc maps.
    Does something similar to calc_observed_lulc_change.

    This function is slow and is not necessary for the actual allocation step (which uses a zone-level calc_observed_lulc_change task.
    However, it may be useful for things with larger zones, eg AEZs. Or, for just visualizing overall results.
    """




    if p.run_this:
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
            p.L.info('Processing observed change row', r, ' to calculate global LULC change.')
            for c in range(p.coarse_match.num_cols):

                t1_subarray = t1.data[int(r * aspect_ratio): int((r + 1) * aspect_ratio), int(c * aspect_ratio): int((c + 1) * aspect_ratio)]
                t2_subarray = t2.data[int(r * aspect_ratio): int((r + 1) * aspect_ratio), int(c * aspect_ratio): int((c + 1) * aspect_ratio)]

                ha_per_coarse_cell_this_subarray = p.global_ha_per_cell_15m.data[r, c]

                # LIMITATION: Currently I do not use the full change matrix and instead summarize it down to a net change vector.
                change_matrix, counters = hb.calc_change_matrix_of_two_int_arrays(t1_subarray.astype(np.int), t2_subarray.astype(np.int), p.classes_that_might_change)

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
    if passed_p is None:
        global p
    else:
        p = passed_p

    p.prepare_lulc_make_pngs = 0

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

                change_matrix, counters = hb.calc_change_matrix_of_two_int_arrays(t1_subarray.astype(np.int), t2_subarray.astype(np.int), p.classes_that_might_change)
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

        if p.prepare_lulc_make_pngs:
            seals_utils.prepare_lulc_pngs()

            from matplotlib.mlab import bivariate_normal
            from matplotlib import colors as colors
            fig, ax = plt.subplots()
            fig.set_size_inches(10, 8)

            # Plot the heatmap
            vmin = np.min(full_change_matrix_no_diagonal)
            vmax = np.max(full_change_matrix_no_diagonal)
            im = ax.imshow(full_change_matrix_no_diagonal, cmap='YlGnBu', norm=colors.LogNorm(vmin=vmin+1, vmax=vmax))
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
                ann = ax.annotate('Zone ' + str(i + 1), xy=(i * (p.coarse_match.n_rows + 1) + .25 * p.coarse_match.n_rows, 1.05), xycoords=trans) #
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
            plt.setp(ax.get_xticklabels(), rotation=90, ha="center",  rotation_mode="anchor")

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
                               num_cbar_ticks=2, vmin=0, vmid=vmax/10.0, vmax=vmax, color_scheme='ylgnbu')

    p.ha_per_cell_15m = None