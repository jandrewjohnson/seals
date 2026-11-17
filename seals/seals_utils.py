import logging, os, math, sys, json, glob
from osgeo import gdal
import numpy as np
import scipy
import scipy.stats as st
import scipy.ndimage
import hazelbean as hb
from hazelbean import config as hb_config
from hazelbean import globals as hb_globals
import pandas as pd
import geopandas as gpd
# from hazelbean.ui import model, inputs
from collections import OrderedDict

import geopandas as gpd
import hazelbean as hb
# import geoecon as ge
import matplotlib
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import scipy
import scipy.ndimage
import scipy.stats as st
from google.cloud import storage
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from osgeo import gdal

# from seals import seals_initialize_project


logging.basicConfig(level=logging.WARNING)
# hb.ui.model.LOGGER.setLevel(logging.WARNING)
# hb.ui.inputs.LOGGER.setLevel(logging.WARNING)

L = hb_config.get_logger('seals_utils')
L.setLevel(logging.INFO)


logging.getLogger('Fiona').setLevel(logging.WARNING)
logging.getLogger('fiona.collection').setLevel(logging.WARNING)

np.seterr(divide='ignore', invalid='ignore')

dev_mode = True

def recompile_cython(env_name):

    # Recompile if needed and configured.
    recompile_cython = 0
    if recompile_cython:
        import subprocess

        ### NOTE HACK the commented out code was added to make it work for 8601
        # try:
        #     if hb.check_conda_env_exists(env_name):
        #         env_name_to_use = env_name
        #     else:
        #         # script_conda_env = sys.prefix
        #         env_name_to_use = sys.prefix
        # except:
        #     env_name_to_use = None
        env_name_to_use = None
        if env_name is None:
            L.critical('You gave a cython environment for recompilation (' + env_name + '), but it does not exist.')
        else:
            env_name_to_use = env_name
        old_cwd = os.getcwd()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)

        cython_command = "python compile_cython_functions.py build_ext -i clean"
        if env_name_to_use:
            cython_command = "conda activate " + env_name_to_use + " && " + cython_command
            process = subprocess.Popen(cython_command, shell=True, stdout=subprocess.PIPE)
            output, err = process.communicate()
            if err:
                raise Exception(err)

            # returned = os.system(cython_command)
        else:
            process = subprocess.Popen(cython_command, shell=True, stdout=subprocess.PIPE)
            output, err = process.communicate()
            if err:
                raise Exception(err)

        if err:
            print('Cythonization failed. This means that you will not be able to import the cython functions if you have edited them. You can still run the model parts of the model, so this will not fully stop execution.')
        process.terminate()
        os.chdir(old_cwd)


def calc_fit_of_projected_against_observed_loss_function(baseline_path, projected_path, observed_path, similarity_class_ids, loss_function_type='l1', save_dir=None):
    """Compare allocation success of baseline to projected against some observed using a l2 loss function
    If similarity_class_ids is given, only calculates score based on the given values (otherwise considers all).
    """

    baseline_af = hb.ArrayFrame(baseline_path)
    projected_af = hb.ArrayFrame(projected_path)
    observed_af = hb.ArrayFrame(observed_path)

    overall_similarity_plot = np.zeros(baseline_af.shape, dtype=np.float64)

    class_similarity_scores = []
    class_similarity_plots = []

    for id in similarity_class_ids:
        similarity_plot = np.zeros(baseline_af.shape, dtype=np.float64)

        baseline_binary = np.where(baseline_af.data.astype(np.float64) == id, 1.0, 0.0)
        projected_binary = np.where(projected_af.data.astype(np.float64) == id, 1.0, 0.0)
        observed_binary = np.where(observed_af.data.astype(np.float64) == id, 1.0, 0.0)

        pb_difference = projected_binary - baseline_binary
        ob_difference = observed_binary - baseline_binary

        pb_expansions = np.where(baseline_binary == 0, projected_binary, 0)
        ob_expansions = np.where(baseline_binary == 0, observed_binary, 0)
        pb_contractions = np.where((baseline_binary == 1) & (projected_binary == 0), 1, 0)
        ob_contractions = np.where((baseline_binary == 1) & (observed_binary == 0), 1, 0)

        hb.show(baseline_binary, save_dir=save_dir)
        hb.show(projected_binary, save_dir=save_dir)
        hb.show(observed_binary, save_dir=save_dir)
        hb.show(pb_expansions, save_dir=save_dir)
        hb.show(ob_expansions, save_dir=save_dir)
        hb.show(pb_contractions, save_dir=save_dir)
        hb.show(ob_contractions, save_dir=save_dir)

        # l1_direct = abs(pb_expansions - ob_expansions) + abs(pb_conctractions - ob_contractions)
        # hb.show(l2a, save_dir=save_dir)

        sgm = 3
        pb_expansions_blurred = scipy.ndimage.filters.gaussian_filter(pb_expansions, sigma=sgm)
        ob_expansions_blurred = scipy.ndimage.filters.gaussian_filter(ob_expansions, sigma=sgm)
        pb_contractions_blurred = scipy.ndimage.filters.gaussian_filter(pb_contractions, sigma=sgm)
        ob_contractions_blurred = scipy.ndimage.filters.gaussian_filter(ob_contractions, sigma=sgm)

        l1_gaussian = abs(pb_expansions_blurred - ob_expansions_blurred) + abs(pb_contractions_blurred - ob_contractions_blurred)
        class_similarity_plots.append(l1_gaussian)
        class_similarity_scores.append(np.sum(l1_gaussian))

        overall_similarity_plot += l1_gaussian


    overall_similarity_score = sum(class_similarity_scores)
    return overall_similarity_score, overall_similarity_plot, class_similarity_scores, class_similarity_plots

def calc_fit_of_projected_against_observed(baseline_uri, projected_uri, observed_uri, similarity_class_ids=None):
    """Compare allocation success of baseline to projected against some observed.
    If similarity_class_ids is given, only calculates score based on the given values (otherwise considers all).

    """
    observed = hb.ArrayFrame(observed_uri)
    projected = hb.ArrayFrame(projected_uri)
    baseline = hb.ArrayFrame(baseline_uri)

    observed_array = observed.data
    projected_array = projected.data
    baseline_array = baseline.data

    observed_ids, observed_counts = np.unique(observed_array, return_counts=True)
    projected_ids, projected_counts = np.unique(projected_array, return_counts=True)
    baseline_ids, baseline_counts = np.unique(baseline_array, return_counts=True)

    if not similarity_class_ids:
        # Define ids to test as Union of observed and projected and baseline
        similarity_class_ids = set(list(observed_ids) + list(projected_ids) + list(baseline_ids))

    changes_by_class = []
    sum_metrics_to = []
    sum_metrics_from = []
    n_predictions_to = []
    n_predictions_from = []
    class_from_scores = []
    similarity_plots = []
    for class_counter, class_id in enumerate(similarity_class_ids):
        L.debug('Calculating similarity for class ' + str(class_id))

        observed_changes_to = np.where((observed_array == class_id) & (baseline_array != class_id), 1, 0)
        observed_changes_from = np.where((observed_array != class_id) & (baseline_array == class_id), 1, 0)

        projected_changes_to = np.where((projected_array == class_id) & (baseline_array != class_id), 1, 0)
        projected_changes_from = np.where((projected_array != class_id) & (baseline_array == class_id), 1, 0)

        sum_metric, n_predictions, observed_not_projected_to_metric, projected_not_observed_to_metric = calc_similarity_of_two_arrays(observed_changes_to, projected_changes_to)

        if n_predictions:
            avg = sum_metric / n_predictions
        else:
            avg = 0

        L.debug('TO sum_metric: ' + str(sum_metric) + ' n_predictions: ' + str(n_predictions) + ' avg: ' + str(avg))

        sum_metrics_to.append(sum_metric)
        n_predictions_to.append(n_predictions)

        sum_metric, n_predictions, observed_not_projected_from_metric, projected_not_observed_from_metric = calc_similarity_of_two_arrays(observed_changes_from,
                                                                                                                                          projected_changes_from)

        if n_predictions:
            avg = sum_metric / n_predictions
        else:
            avg = 0

        L.debug('FROM sum_metric: ' + str(sum_metric) + ' n_predictions: ' + str(n_predictions) + ' avg: ' + str(avg))

        similarity_plots.append(observed_not_projected_to_metric + projected_not_observed_to_metric + observed_not_projected_from_metric + projected_not_observed_from_metric)

        sum_metrics_from.append(sum_metric)
        n_predictions_from.append(n_predictions)

    overall_similarity = 0
    if sum(n_predictions_to) or sum(n_predictions_from):
        overall_similarity = (sum(sum_metrics_to) + sum(sum_metrics_from)) / (sum(n_predictions_to) + sum(n_predictions_from))

    return overall_similarity, similarity_plots


def calc_similarity_of_two_arrays(a1, a2):
    a1_not_a2_flipped_array = np.where((a1 == 1) & (a2 == 0), 0, 1)
    a2_not_a1_flipped_array = np.where((a2 == 1) & (a1 == 0), 0, 1)

    a1_has_values = np.any(a1_not_a2_flipped_array == 0)
    a2_has_values = np.any(a2_not_a1_flipped_array == 0)

    if a1_has_values:
        if a2_has_values:
            a1_not_a2_distance = scipy.ndimage.morphology.distance_transform_edt(a1_not_a2_flipped_array.astype(np.float64)) ** 2
            a1_not_a2_metric = np.where(a2 == 1, a1_not_a2_distance, 0).astype(np.float64)
            a2_not_a1_distance = scipy.ndimage.morphology.distance_transform_edt(a2_not_a1_flipped_array.astype(np.float64)) ** 2
            a2_not_a1_metric = np.where(a1 == 1, a2_not_a1_distance, 0).astype(np.float64)
        else:
            L.debug('Projected zero change, but there was observed change. It is unclear to assess the similarity of these metrics.')
            a1_not_a2_distance = scipy.ndimage.morphology.distance_transform_edt(a1_not_a2_flipped_array.astype(np.float64)) ** 2
            a1_not_a2_metric = np.where(a2 == 1, a1_not_a2_distance, 0).astype(np.float64)
            a2_not_a1_metric = np.zeros(a1_has_values.shape)
    else:
        if a2_has_values:
            L.debug('Observed zero change, but there was projected change. It is unclear to assess the similarity of these metrics.')
            a1_not_a2_metric = np.zeros(a1_has_values.shape).astype(np.float64)
            a2_not_a1_distance = scipy.ndimage.morphology.distance_transform_edt(a2_not_a1_flipped_array.astype(np.float64)) ** 2
            a2_not_a1_metric = np.where(a1 == 1, a2_not_a1_distance, 0).astype(np.float64)
        else:
            L.debug('Projected and observed zero change. It is unclear to assess the similarity of these metrics.')
            a1_not_a2_metric = np.zeros(a1_has_values.shape).astype(np.float64)
            a2_not_a1_metric = np.zeros(a1_has_values.shape).astype(np.float64)

    if a1_has_values:
        if a2_has_values:
            sum_metric = np.sum(a1_not_a2_metric) + np.sum(a2_not_a1_metric)
            n_predictions = np.count_nonzero(a1_not_a2_metric) + np.count_nonzero(a2_not_a1_metric)
            # nd.show(a1_not_a2_metric, title='a1_not_a2_metric')
            # nd.show(a2_not_a1_metric, title='a2_not_a1_metric')
        else:
            sum_metric = np.sum(a1_not_a2_metric)
            n_predictions = np.count_nonzero(a1_not_a2_metric)
    else:
        if a2_has_values:
            sum_metric = np.sum(a2_not_a1_metric)
            n_predictions = np.count_nonzero(a2_not_a1_metric)
        else:
            sum_metric = 0
            n_predictions = 0

    return sum_metric, n_predictions, a1_not_a2_metric, a2_not_a1_metric


def get_classes_net_changes_from_lulc_comparison(lulc_1_uri, lulc_2_uri):
    af1 = hb.ArrayFrame(lulc_1_uri)
    af2 = hb.ArrayFrame(lulc_2_uri)
    array1 = af1.data
    array2 = af2.data
    n_cells = np.count_nonzero(array1)
    unique_items_1, counts_1 = np.unique(array1, return_counts=True)
    unique_items_2, counts_2 = np.unique(array2, return_counts=True)
    return_odict = OrderedDict()

    for i in range(len(unique_items_1)):
        a = (counts_1[i] / n_cells) * ((counts_2[i] - counts_1[i]) / counts_1[i])
        if not (i == 0 and a == 0):
            return_odict[unique_items_1[i]] = a

    return return_odict

def normalize_array(array, low=0, high=1, log_transform=True):
    # TODOO Could be made faster by giving pre-known minmix values
    if log_transform:
        min = np.min(array)
        max = np.max(array)
        to_add = float(min * -1.0 + 1.0)
        array = array + to_add

        array = np.log(array)

    min = np.min(array)
    max = np.max(array)

    normalizer = (high - low) / (max - min)

    output_array = (array - min) * normalizer

    return output_array

def distance_from_blurred_threshold(input_array, sigma, threshold, decay):
    """
    Blur the input with a guassian using sigma (higher sigma means more blueas of the blur above the threshold,
    return 1 - blurred so thtat values near zero indicate strong presence. In areas r). In arbelow the threshold, return


    The positive attribute of this func is it gives an s-curve relationship between 0 and 1 with a smoothed discontinuity
    around the threshold while never converging too close to 1 even at extreme distances without requiring slow calculation
    of a large footrint convolution.
    """

    blurred = scipy.ndimage.filters.gaussian_filter(input_array, sigma).astype(np.float32)

    blurred_below = np.where(blurred < threshold, 1, 0).astype(np.float32)  # NOTE opposite logic because EDT works only where true.

    if np.any(blurred_below == 0):
        distance = scipy.ndimage.morphology.distance_transform_edt(blurred_below).astype(np.float32)
    else:
        # Interesting choice here that I wasn't sure about how to address:
        # In the event that there are NO significant shapes, and thus blurred_below is all ones, what should be the distance?
        L.warning('WARNING NOTHING came up as above the blurring threshold!')

        # CRITICAL CHANGE, shouldnt it be zeros?
        metric = np.zeros(blurred.shape)
        return metric

    outside = 1.0 - (1.0 / ((1 + float(decay)) ** (distance) + (1.0 / float(threshold) - 1)))  # 1 -  eponential distance decay from blur above threshold minus scalar that makes it match the level of
    inside = np.ones(blurred.shape).astype(np.float32) - blurred  # lol. just use 1 - the blurred value when above the threshold.

    metric = np.where(blurred_below == 1, outside, inside).astype(np.float32)

    metric = np.where(metric > 0.9999999, 1, metric)
    metric = np.where(metric < 0.0000001, 0, metric)
    metric = 1 - metric

    return metric


def calc_change_vector_of_change_matrix(change_matrix):
    k = [0] * change_matrix.shape[0]
    for i in range(change_matrix.shape[0]):
        for j in range(change_matrix.shape[1]):
            if i != j:
                k[i] -= change_matrix[i, j]
                k[j] += change_matrix[i, j]
    return k

# from pygeo.geoprocessing import convolve_2d
from pygeoprocessing import convolve_2d


def fft_gaussian(signal_path, kernel_path, target_path, target_nodata=-9999.0, compress=False, n_threads=1):
    """
    Blur the input with a guassian using sigma (higher sigma means more blur). In areas of the blur above the threshold,
    return 1 - blurred so thtat values near zero indicate strong presence. In areas below the threshold, return


    The positive attribute of this func is it gives an s-curve relationship between 0 and 1 with a smoothed discontinuity
    around the threshold while never converging too close to 1 even at extreme distances without requiring slow calculation
    of a large footrint convolution.
    """
    signal_path_band = (signal_path, 1)
    kernel_path_band = (kernel_path, 1)

    if compress:
        gtiff_creation_options = hb.DEFAULT_GTIFF_CREATION_OPTIONS
    else:
        gtiff_creation_options = hb.DEFAULT_GTIFF_CREATION_OPTIONS

    raster_driver_creation_tuple = ('GTiff', gtiff_creation_options)
    convolve_2d(
        signal_path_band, kernel_path_band, target_path,
        ignore_nodata_and_edges=False, mask_nodata=False, normalize_kernel=False,
        target_datatype=gdal.GDT_Float32,
        target_nodata=target_nodata,
        raster_driver_creation_tuple=raster_driver_creation_tuple)


def get_array_from_two_dim_first_order_kernel_function(radius, starting_value, halflife):
    diameter = radius * 2
    x = np.linspace(-radius, radius + 1, diameter)
    y = np.linspace(-radius, radius + 1, diameter)

    X, Y = np.meshgrid(x, y)
    output_array = np.zeros((int(diameter), int(diameter)))

    for i in range(int(diameter)):
        for j in range(int(diameter)):
            x = i - radius
            y = j - radius
            output_array[i, j] = two_dim_first_order_kernel_function(x, y, starting_value, halflife)

    return output_array


def two_dim_first_order_kernel_function(x, y, starting_value, halflife):
    steepness = 4 / halflife  # Chosen to have a decent level of curvature difference across interval
    return regular_sigmoidal_first_order((x ** 2 + y ** 2) ** 0.5, left_value=starting_value, inflection_point=halflife, steepness=steepness)


def regular_sigmoidal_first_order(x,
                                  left_value=1.0,
                                  inflection_point=5.0,
                                  steepness=1.0,
                                  magnitude=1.0,
                                  scalar=1.0,
                                  ):
    return scalar * sigmoidal_curve(x, left_value, inflection_point, steepness, magnitude)


def sigmoidal_curve(x, left_value, inflection_point, steepness, magnitude, e=hb_globals.e):
    return left_value / ((1. / magnitude) + e ** (steepness * (x - inflection_point)))



### UNUSED
def generate_gaussian_kernel(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array. kernlen determines the size (always choose ODD numbers unless you're baller cause of asymmetric results.
    nsig is the signma blur. HAving it too small makes the blur not hit zero before the edge."""

    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()

    return kernel


def vector_to_kernel(input_vector, output_path=None, match_path=None):
    """Returns a 2D Gaussian kernel array."""
    kernel_raw = np.sqrt(np.outer(input_vector, input_vector))
    kernel = kernel_raw / kernel_raw.sum()

    if output_path is not None:
        hb.save_array_as_geotiff(kernel, output_path, match_path)
        res = -180.0 / float(len(input_vector)) * 2.0
        dummy_geotransform = (-180.0, res, 0.0, 90.0, 0.0, -res)

        hb.save_array_as_geotiff(kernel, output_path, data_type=7, ndv=-9999.0,
                                 geotransform_override=dummy_geotransform, projection_override=hb.common_projection_wkts['wgs84'], n_cols_override=kernel.shape[1],
                                 n_rows_override=kernel.shape[0])

    return kernel


def regular_sigmoidal_second_order(x,
                                   left_value_1=-1.0,
                                   inflection_point_1=5.0,
                                   steepness_1=1.0,
                                   magnitude_1=1.0,
                                   scalar_1=1.0,
                                   left_value_2=1.,
                                   inflection_point_2=15.0,
                                   steepness_2=1.0,
                                   magnitude_2=1.0,
                                   scalar_2=1.0,
                                   ):
    """
    Magnitude, needs to be 1 for last order.
    """

    return scalar_1 * sigmoidal_curve(x, left_value_1, inflection_point_1, steepness_1, magnitude_1) + \
           scalar_2 * sigmoidal_curve(x, left_value_2, inflection_point_2, steepness_2, magnitude_2)


def regular_sigmoidal_third_order(x,
                                  left_value_1=1.0,
                                  inflection_point_1=5.0,
                                  steepness_1=1.0,
                                  magnitude_1=1.0,
                                  scalar_1=1.0,
                                  left_value_2=-1.,
                                  inflection_point_2=15.0,
                                  steepness_2=1.0,
                                  magnitude_2=1.0,
                                  scalar_2=1.0,
                                  left_value_3=1.0,
                                  inflection_point_3=25.0,
                                  steepness_3=1.0,
                                  magnitude_3=1.0,
                                  scalar_3=1.0,
                                  ):
    return scalar_1 * sigmoidal_curve(x, left_value_1, inflection_point_1, steepness_1, magnitude_1) + \
           scalar_2 * sigmoidal_curve(x, left_value_2, inflection_point_2, steepness_2, magnitude_2) + \
           scalar_3 * sigmoidal_curve(x, left_value_3, inflection_point_3, steepness_3, magnitude_3)

def one_dim_first_order_kernel_function(x, starting_value, halflife):
    steepness = 4 / halflife # Chosen to have a decent level of curvature difference across interval
    return regular_sigmoidal_first_order(x, left_value=starting_value, inflection_point=halflife, steepness=steepness)


def two_dim_distance_on_function_with_2_args(x, y, input_function, starting_value, halflife):
    return input_function((x ** 2 + y ** 2) ** 0.5, starting_value, halflife)


def carbon(lulc_path, carbon_zones_path, carbon_table_path, output_path):
    lookup_table_df = pd.read_csv(carbon_table_path, index_col=0)
    lookup_table = np.float32(lookup_table_df.values)
    row_names = {int(v): int(c) for c, v in enumerate(lookup_table_df.index)}
    col_names = {int(v): int(c) for c, v in enumerate(lookup_table_df.columns)}


    # HACK! I wasn't able to get gtap_invest to rebuild the cython on my new computer. so I just copied this into seals.
    # import gtap_invest
    # import gtap_invest.global_invest
    # import gtap_invest.global_invest.carbon_biophysical
    from seals import seals_cython_functions as seals_cython_functions
    # from seals_cython_functions import write_carbon_table_to_array
    base_raster_path_band = [(lulc_path, 1), (carbon_zones_path, 1), (lookup_table, 'raw'), (row_names, 'raw'), (col_names, 'raw')]
    hb.raster_calculator_hb(base_raster_path_band, seals_cython_functions.write_carbon_table_to_array, output_path, 6, -9999, hb.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS_HB)


def calculate_carbon_naively(luh_ag_expansion_ha_path,
                             luh_grassland_expansion_ha_path,
                             luh_urban_expansion_ha_path,
                             carbon_per_ha_fine_path,
                             ha_per_cell_fine_path,
                             ha_per_cell_coarse_path,
                             fine_resolution_arcseconds,
                             coarse_resolution_arcseconds, ):
    ## Load cropland change in hectares
    ### WARNING!! To make a tight manuscript deadline, I hacked the shape of the arrays to fit with eg [:, 0:-90] . Need to still make this general for any bb.
    luh_ag_expansion_ha_array = hb.as_array(luh_ag_expansion_ha_path)
    luh_ag_expansion_ha_array = luh_ag_expansion_ha_array[:, 0:-1]

    luh_grassland_expansion_ha_array = hb.as_array(luh_grassland_expansion_ha_path)
    luh_grassland_expansion_ha_array = luh_grassland_expansion_ha_array[:, 0:-1]

    luh_urban_expansion_ha_array = hb.as_array(luh_urban_expansion_ha_path)
    luh_urban_expansion_ha_array = luh_urban_expansion_ha_array[:, 0:-1]

    ## Load  cell-hectare conversion tifs.
    ha_per_cell_fine = hb.as_array(ha_per_cell_fine_path)[:, 0:-90]
    ha_per_cell_coarse = hb.as_array(ha_per_cell_coarse_path)[:, 0:-1]

    ## Load carbon fine

    carbon_per_ha_fine = hb.as_array(carbon_per_ha_fine_path)

    ## Convert to per-cell
    carbon_per_cell_fine = carbon_per_ha_fine * ha_per_cell_fine


    ## Aggregate carbon PER CELL to coarse resolution
    upscale_factor = int(coarse_resolution_arcseconds / fine_resolution_arcseconds)
    carbon_per_cell_upscaled = hb.upscale_retaining_sum(carbon_per_cell_fine.astype(np.float64), upscale_factor)

    carbon_per_ha_upscaled = carbon_per_cell_upscaled / ha_per_cell_coarse

    nat_to_ag_conversion_loss =  1 - 5.0 / (69.0) # Comes from the ratio of forested to ag land in carbon zone 618 following Ruesch and Gibbs 2008.
    nat_to_grassland_conversion_loss =  1 - 6.0 / (69.0) # Comes from the ratio of forested to ag land in carbon zone 618 following Ruesch and Gibbs 2008.
    nat_to_urban_conversion_loss =  1 - 0.0 / (69.0) # Comes from the ratio of forested to ag land in carbon zone 618 following Ruesch and Gibbs 2008.

    carbon_loss_from_ag_per_cell = nat_to_ag_conversion_loss * carbon_per_cell_upscaled * (luh_ag_expansion_ha_array / ha_per_cell_coarse)
    carbon_loss_from_grassland_per_cell = nat_to_grassland_conversion_loss * carbon_per_cell_upscaled * (luh_grassland_expansion_ha_array / ha_per_cell_coarse)



    carbon_loss_from_urban_per_cell = nat_to_urban_conversion_loss * carbon_per_cell_upscaled * (luh_urban_expansion_ha_array / ha_per_cell_coarse)

    # Hack to make just urb crop
    carbon_loss_from_grassland_per_cell = carbon_loss_from_grassland_per_cell * 0.0
    carbon_loss_from_urban_per_cell = carbon_loss_from_urban_per_cell * 0.0

    carbon_per_cell_after_loss = carbon_per_cell_upscaled - carbon_loss_from_ag_per_cell - carbon_loss_from_grassland_per_cell - carbon_loss_from_urban_per_cell


    return carbon_per_cell_after_loss, carbon_per_ha_upscaled, carbon_per_cell_fine


def assign_defaults_from_model_spec(input_object, model_spec_dict):

    # Next version will be replaced by seals_model_spec version.

    for k, v in model_spec_dict.items():
        if not hasattr(input_object, k):
            setattr(input_object, k, v)


def assign_df_row_to_object_attributes(input_object, input_row):
    # Thin shim. The per-row value grammar (cat-ears, [list]/{dict} literals, path
    # detection, year/dimensions parsing, nan handling) is owned by hazelbean now and
    # is shared with gtappy and the vertical parser -- this eliminates the duplication
    # the old TODO complained about. See hb.assign_to_object. The SEALS-specific tail
    # below (model_spec defaults + seals_years override) stays here on purpose.
    hb.assign_row_to_object_attributes(input_object, input_row)

    model_spec = {}
    model_spec['regional_projections_input_path'] = ''
    assign_defaults_from_model_spec(input_object, model_spec)

    # If seals_years is set and this is a non-baseline scenario, override p.years with p.seals_years.
    # This lets SEALS iterate over a coarser year grid (e.g. MAgPIE's 5-year timesteps) while GTAP
    # tasks keep using annual p.years (GTAP uses gtappy_utils.assign_df_row_to_object_attributes,
    # not this SEALS version, so GTAP tasks are unaffected by this override).
    seals_years = getattr(input_object, 'seals_years', None)
    scenario_type = getattr(input_object, 'scenario_type', None)
    if (isinstance(seals_years, list) and len(seals_years) > 0
            and all(isinstance(y, int) for y in seals_years)
            and scenario_type is not None and str(scenario_type) != 'baseline'):
        input_object.years = seals_years

def set_derived_attributes(p):

    # Resolutions come from the fine and coarse maps
    p.fine_resolution = hb.get_cell_size_from_path(p.base_year_lulc_path)
    p.fine_resolution_arcseconds = hb.pyramid_compatible_resolution_to_arcseconds[p.fine_resolution]
    
    # Check for explicit coarse resolution from scenarios.csv column.
    # Regional (non-global) NetCDFs need this because auto-detection assumes
    # global extent (180/num_lat_cells) and computes wrong values.
    explicit_coarse_resolution = getattr(p, 'coarse_resolution_arcseconds', None)
    if explicit_coarse_resolution is not None:
        p.coarse_resolution_arcseconds = float(explicit_coarse_resolution)
        p.coarse_resolution = hb.pyramid_compatible_resolutions[p.coarse_resolution_arcseconds]
    elif hb.path_exists(p.coarse_projections_input_path):
        p.coarse_resolution = hb.get_cell_size_from_path(p.coarse_projections_input_path)
        p.coarse_resolution_arcseconds = hb.pyramid_compatible_resolution_to_arcseconds[p.coarse_resolution]
    else:
        p.coarse_resolution_arcseconds = float(p.coarse_projections_input_path)
        p.coarse_resolution = hb.pyramid_compatible_resolutions[p.coarse_resolution_arcseconds]

    p.fine_resolution_degrees = hb.pyramid_compatible_resolutions[p.fine_resolution_arcseconds]
    p.coarse_resolution_degrees = hb.pyramid_compatible_resolutions[p.coarse_resolution_arcseconds]
    p.fine_resolution = p.fine_resolution_degrees
    p.coarse_resolution = p.coarse_resolution_degrees



    # Set the derived-attributes too whenever the core attributes are set
    p.lulc_correspondence_path = p.get_path(p.lulc_correspondence_path)
    # p.lulc_correspondence_path = hb.get_first_extant_path(p.lulc_correspondence_path, [p.input_dir, p.base_data_dir])
    p.lulc_correspondence_dict = hb.utils.get_reclassification_dict_from_df(p.lulc_correspondence_path, 'src_id', 'dst_id', 'src_label', 'dst_label')


    p.coarse_correspondence_path = p.get_path(p.coarse_correspondence_path)
    # p.coarse_correspondence_path = hb.get_first_extant_path(p.coarse_correspondence_path, [p.input_dir, p.base_data_dir])
    p.coarse_correspondence_dict = hb.utils.get_reclassification_dict_from_df(p.coarse_correspondence_path, 'src_id', 'dst_id', 'src_label', 'dst_label')

    ## Load the indices and labels from the COARSE correspondence. We need this go get waht calsses are changing.
    if p.coarse_correspondence_dict is not None:

        coarse_dst_ids = p.coarse_correspondence_dict['dst_ids']
        p.coarse_correspondence_class_indices = sorted([int(i) for i in coarse_dst_ids])

        coarse_dst_ids_to_labels = p.coarse_correspondence_dict['dst_ids_to_labels']
        p.coarse_correspondence_class_labels = [str(coarse_dst_ids_to_labels[i]) for i in p.coarse_correspondence_class_indices]


    if p.lulc_correspondence_dict is not None:
        lulc_dst_ids = p.lulc_correspondence_dict['dst_ids']
        p.lulc_correspondence_class_indices = sorted([int(i) for i in lulc_dst_ids])

        lulc_dst_ids_to_labels = p.lulc_correspondence_dict['dst_ids_to_labels']
        p.lulc_correspondence_class_labels = [str(lulc_dst_ids_to_labels[i]) for i in p.lulc_correspondence_class_indices]

    # Define the nonchanging class indices as anything in the lulc simplification classes that is not in the coarse simplification classes
    p.nonchanging_class_indices = [int(i) for i in p.lulc_correspondence_class_indices if i not in p.coarse_correspondence_class_indices] # These are the indices of classes THAT CANNOT EXPAND/CONTRACT

    # Classes a project wants protected beyond the non-changing ones: classes nothing may
    # expand onto even though they do have a coarse demand. Empty by default. The no-demand
    # classes are derived above; anything further is a scenario statement rather than a
    # property of the correspondences, so it has to be named. Urban is the standing example,
    # since it expands but built land is not un-built.

    p.changing_coarse_correspondence_class_indices = [int(i) for i in p.coarse_correspondence_class_indices if i not in p.nonchanging_class_indices] # These are the indices of classes THAT CAN EXPAND/CONTRACT
    p.changing_coarse_correspondence_class_labels = [str(p.coarse_correspondence_dict['dst_ids_to_labels'][i]) for i in p.changing_coarse_correspondence_class_indices if i not in p.nonchanging_class_indices]
    p.changing_lulc_correspondence_class_indices = [int(i) for i in p.lulc_correspondence_class_indices if i not in p.nonchanging_class_indices] # These are the indices of classes THAT CAN EXPAND/CONTRACT
    p.changing_lulc_correspondence_class_labels = [str(p.lulc_correspondence_dict['dst_ids_to_labels'][i]) for i in p.changing_lulc_correspondence_class_indices if i not in p.nonchanging_class_indices]

    # From the changing/nonchanging class sets as defined in the lulc correspondence AND the coarse correspondence.
    p.changing_class_indices = p.changing_coarse_correspondence_class_indices + [i for i in p.changing_lulc_correspondence_class_indices if i not in p.changing_coarse_correspondence_class_indices]
    p.changing_class_labels = p.changing_coarse_correspondence_class_labels + [i for i in p.changing_lulc_correspondence_class_labels if i not in p.changing_coarse_correspondence_class_labels]

    p.all_class_indices = p.coarse_correspondence_class_indices + [i for i in p.lulc_correspondence_class_indices if i not in p.coarse_correspondence_class_indices]
    p.all_class_labels = p.coarse_correspondence_class_labels + [i for i in p.lulc_correspondence_class_labels if i not in p.coarse_correspondence_class_labels]
    p.class_labels = p.all_class_labels

    # Check if processing_resolution exists
    if not hasattr(p, 'processing_resolution'):
        p.processing_resolution = 1.0

    p.processing_resolution_arcseconds = p.processing_resolution * 3600.0 # MUST BE FLOAT

def download_google_cloud_blob(bucket_name, source_blob_name, credentials_path, destination_file_name, chunk_size=262144*5,):

    print("""DEPRECATED FOR HB VERSION. Downloads a blob from the bucket.""")

    # raise
    require_database = True
    if hb.path_exists(credentials_path) and require_database:
        client = storage.Client.from_service_account_json(credentials_path)
    else:
        raise NameError('Unable to find database license. Your code currently is stating it should be found at ' + str(credentials_path) + ' relative to where the run script was launched. If you would like to generate your own database, disable this check (set require_database to False). Otherwise, feel free to reach out to the developer of SEALS to acquire the license key. It is only limited because the data are huge and expensive to host (600+ Gigabytes).')

    try:
        bucket = client.get_bucket(bucket_name)
    except Exception as e:
        L.critical('Unable to get bucket ' + str(bucket_name) + ' with exception ' + str(e))

    try:
        blob = bucket.get_blob(source_blob_name) # LEARNING POINT, difference between bucket.blob and bucket.get_blob is the latter sets extra attributes like blob.size.
    except Exception as e:
        L.critical('Unable to get blob ' + str(source_blob_name) + ' with exception ' + str(e))

    if blob is None:
        L.critical('Unable to get blob ' + str(source_blob_name) + ' from ' + source_blob_name + ' in ' + bucket_name + '.')


    L.info('Starting to download to ' + destination_file_name + ' from ' + source_blob_name + ' in ' + bucket_name + '. The size of the object is ' + str(blob.size))

    current_dir = os.path.split(destination_file_name)[0]
    hb.create_directories(current_dir)


    try:
        blob.download_to_filename(destination_file_name)
    except Exception as e:
        L.critical('Blob download_to_file failed for ' + str(destination_file_name) + ' with exception ' + str(e))



def generate_scenarios_csv_and_put_in_input_dir(p):
    # In the event that the scenarios csv was not set, this currently wouldn't yet be a scenarios_df
    # Yet, I still want to be able to iterate over it. So thus, I need to GENERATE the scenarios_df from the project_flow
    # attributes
    list_of_attributes_to_write = [

        'scenario_label',
        'scenario_type',
        'aoi',
        'exogenous_label',
        'climate_label',
        'model_label',
        'counterfactual_label',
        'years',
        'baseline_reference_label',
        'base_years',
        'key_base_year',
        'comparison_counterfactual_labels',
        'time_dim_adjustment',
        'coarse_projections_input_path',
        'lulc_src_label',
        'lulc_simplification_label',
        'lulc_correspondence_path',
        'coarse_src_label',
        'coarse_simplification_label',
        'coarse_correspondence_path',
        'lc_class_varname',
        'dimensions',
        'calibration_parameters_source',
        'base_year_lulc_path',
    ]


    data = {i: [] for i in list_of_attributes_to_write}

    # Add a baseline row. For the next scenario specific row it will actually take from the p attributes
    # however for the baseline row we have to override a few things (like setting years to the base years)
    data['scenario_label'].append('baseline_' + p.model_label)
    data['scenario_type'].append('baseline')
    data['aoi'].append(p.aoi)
    data['exogenous_label'].append('baseline')
    data['climate_label'].append(''	)
    data['model_label'].append(p.model_label)
    data['counterfactual_label'].append('')
    data['years'].append(' '.join([str(int(i)) for i in p.base_years]))
    data['baseline_reference_label'].append('' )
    data['base_years'].append(' '.join([str(int(i)) for i in p.base_years]))
    data['key_base_year'].append(p.key_base_year)
    data['comparison_counterfactual_labels'].append('')
    data['time_dim_adjustment'].append('add2015')
    data['coarse_projections_input_path'].append(p.coarse_projections_input_path)
    data['lulc_src_label'].append('esa')
    data['lulc_simplification_label'].append('seals7')
    data['lulc_correspondence_path'].append('seals/default_inputs/esa_seals7_correspondence.csv')
    data['coarse_src_label'].append('luh2-14')
    data['coarse_simplification_label'].append('seals7')
    data['coarse_correspondence_path'].append('seals/default_inputs/luh2-14_seals7_correspondence.csv')
    data['lc_class_varname'].append('all_variables')
    data['dimensions'].append('time')
    data['calibration_parameters_source'].append('seals/default_inputs/default_global_coefficients.csv')
    data['base_year_lulc_path'].append(p.base_year_lulc_path)


    # Add non baseline. Now that the baseline was added, we can now just iterate over the existing attributes
    for i in list_of_attributes_to_write:
        current_attribute = getattr(p, i)
        if type(current_attribute) is str:
            if current_attribute.startswith('['):
                current_attribute = ' '.join(list(current_attribute))
            elif os.path.isabs(current_attribute):
                current_attribute = hb.path_split_at_dir(current_attribute, os.path.split(p.base_data_dir)[1])[2].replace('\\\\', '\\').replace('\\', '/') # NOTE Awkward hack of assuming there is only 1 dir with the same name as base_data_dir

        elif type(current_attribute) is list:
            current_attribute = ' '.join([str(i) for i in current_attribute])

        data[i].append(current_attribute)

    p.scenarios_df = pd.DataFrame(data=data, columns=list_of_attributes_to_write)

    hb.create_directories(p.scenario_definitions_path)
    p.scenarios_df.to_csv(p.scenario_definitions_path, index=False)



def set_attributes_to_default(p):
    # Set all ProjectFlow attributes to SEALS default.
    # This is used if the user has never run something before, and therefore doesn't
    # have a scenario_definitions.csv in their input dir.
    # This function will set the attributes, and can be paired with
    # generate_scenarios_csv_and_put_in_input_dir to write the file.

    ###--- SET DEFAULTS ---###

    # String that uniquely identifies the scenario. Will be referenced by other scenarios for comparison.
    p.scenario_label = 'ssp2_rcp45_luh2-globio_bau'

    # Scenario type determines if it is historical (baseline) or future (anything else) as well
    # as what the scenario should be compared against. I.e., Policy minus BAU.
    p.scenario_type = 'bau'

    # Sets the area of interest. If set as a country-ISO3 code, all data will be generated based
    # that countries boundaries (as defined in the base data). Other options include setting it to
    # 'global' or a specific shapefile, or iso3 code. Good small examples include RWA, BTN
    p.aoi = 'RWA'

    # Exogenous label references some set of exogenous drivers like population, TFP growth, LUH2 pattern, SSP database etc
    p.exogenous_label = 'ssp2'

    # One of the climate RCPs
    p.climate_label = 'rcp45'

    # Indicator of which model led to the coarse projection
    p.model_label = 'luh2-message'

    # AKA policy scenario, or a label of something that you have tweaked to assess it's efficacy
    p.counterfactual_label = 'bau'

    # If is not a baseline scenario, these are years into the future. Duplicate the base_year variable below if it is a base year
    # From the csv, this is a space-separated list.
    p.years = [2100]

    # For calculating difference from base year, this references which baseline (must be consistent due to different
    # models having different depictions of the base year)
    p.baseline_reference_label = 'baseline_luh2-message'

    # Which year in the observed data constitutes the base year. There may be multiple if, for instance, you
    # want to use seals to downscale model outputs that update a base year to a more recent year base year
    # which would now be based on model results but is for an existing year. Paper Idea: Do this for validation.
    p.base_years = [2017]

    # Even with multiple years, we will designate one as the key base year, which will be used e.g.
    # for determining the resolution, extent, projection and other attributes of the fine resolution
    # lulc data. It must refer to a year that has observed LULC data to load.
    p.key_base_year = 2017


    # If set to one of the other policies, like BAU, this will indicate which scenario to compare the performance
    # of this scenario to. The tag 'no_policy' indicates that it should not be compared to anything.
    p.comparison_counterfactual_labels = 'no_policy'

    # Path to the coarse land-use change data. SEALS supports 2 types: a netcdf directory of geotiffs following
    # the documented structure.
    p.coarse_projections_input_path = "luh2/raw_data/rcp45_ssp2/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-MESSAGE-ssp245-2-1-f_gn_2015-2100.nc"
    # p.coarse_projections_input_path = 'luh2/raw_data/rcp26_ssp1/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-IMAGE-ssp126-2-1-f_gn_2015-2100.nc'

    # Label of the LULC data being reclassified into the simplified form.
    p.lulc_src_label = 'esa'

    # Label of the new LULC simplified classification
    p.lulc_simplification_label = 'seals7'

    # Path to a csv that will map the a many-to-one reclassification of
    # the src LULC map to a simplified version
    # includes at least src_id, dst_id, src_label, dst_label
    p.lulc_correspondence_path = 'seals/default_inputs/esa_seals7_correspondence.csv'

    # Label of the coarse LUC data that will be reclassified and downscaled
    p.coarse_src_label = 'luh2-14'

    # Label of the new coarse LUC data
    p.coarse_simplification_label = 'seals7'

    # Path to a csv that includes at least src_id, dst_id, src_label, dst_label
    p.coarse_correspondence_path = 'seals/default_inputs/luh2-14_seals7_correspondence.csv'

    # Often NetCDF files can have the time dimension in something other than just the year. This string allows
    # for doing operations on the time dimension to match what is desired. e.g., multiply5 add2015
    p.time_dim_adjustment = 'add2015'

    # Because different NetCDF files have different arrangements (e.g. time is in the dimension
    # versus LU_class is in the dimension), this option allows you to specify where in the input
    # NC the information is. If 'all_variables', assumes the LU classes will be the different variables named
    # otherwise it can be a subset of variables, otherwise, if it is a named variable, e.g. LC_area_share
    # then assume that the lc_class variable is stored in the last-listed dimension (see p.dimensions)
    p.lc_class_varname = 'all_variables'

    # Lists which dimensions are stored in the netcdf in addition to lat and lon. Ideally
    # this is just time but sometimes there are more.  # From the csv, this is a space-separated list.
    p.dimensions = 'time'

    # # To speed up processing, select which classes you know won't change. For default seals7, this is
    # # the urban classes, the water classes, and the bare land class.
    # p.nonchanging_class_indices = [0, 6, 7]

    # Path to a csv which contains all of the pretrained regressor variables. Can also
    # be 'from_calibration' indicating that this run will actually create the calibration``
    # or it can be from a tile-designated file of location-specific regressor variables.
    p.calibration_parameters_source = 'seals/default_inputs/default_global_coefficients.csv'

    # Some data, set to default inputs here, are required to make the model run becayse they determine which classes, which resolutions, ...
    p.key_base_year_lulc_simplified_path = os.path.join('lulc', p.lulc_src_label, p.lulc_simplification_label, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + str(p.key_base_year) + '.tif')
    p.key_base_year_lulc_src_path = os.path.join('lulc', p.lulc_src_label, 'lulc_' + p.lulc_src_label  + '_' + str(p.key_base_year) + '.tif')

    # For convenience, set a synonym for the key_base_year_lulc_simplified_path
    p.base_year_lulc_path = p.key_base_year_lulc_src_path


# TODOO Figure out where is best to store these. For the latest gtap-invest version, I put them in utils as well, but not sure that makese sense
def set_attributes_to_dynamic_default(p):
    # Set all ProjectFlow attributes to SEALS default.
    # This is used if the user has never run something before, and therefore doesn't
    # have a scenario_definitions.csv in their input dir.
    # This function will set the attributes, and can be paired with
    # generate_scenarios_csv_and_put_in_input_dir to write the file.

    ###--- SET DEFAULTS ---###

    # String that uniquely identifies the scenario. Will be referenced by other scenarios for comparison.
    p.scenario_label = 'ssp2_rcp45_luh2-globio_bau'

    # Scenario type determines if it is historical (baseline) or future (anything else) as well
    # as what the scenario should be compared against. I.e., Policy minus BAU.
    p.scenario_type = 'bau'

    # Sets the area of interest. If set as a country-ISO3 code, all data will be generated based
    # that countries boundaries (as defined in the base data). Other options include setting it to
    # 'global' or a specific shapefile, or iso3 code. Good small examples include RWA, BTN
    p.aoi = 'RWA'

    # Exogenous label references some set of exogenous drivers like population, TFP growth, LUH2 pattern, SSP database etc
    p.exogenous_label = 'ssp2'

    # One of the climate RCPs
    p.climate_label = 'rcp45'

    # Indicator of which model led to the coarse projection
    p.model_label = 'luh2-message'

    # AKA policy scenario, or a label of something that you have tweaked to assess it's efficacy
    p.counterfactual_label = 'bau'

    # If is not a baseline scenario, these are years into the future. Duplicate the base_year variable below if it is a base year
    # From the csv, this is a space-separated list.
    p.years = [2045, 2075]

    # For calculating difference from base year, this references which baseline (must be consistent due to different
    # models having different depictions of the base year)
    p.baseline_reference_label = 'baseline_luh2-message'

    # Which year in the observed data constitutes the base year. There may be multiple if, for instance, you
    # want to use seals to downscale model outputs that update a base year to a more recent year base year
    # which would now be based on model results but is for an existing year. Paper Idea: Do this for validation.
    p.base_years = [2017]

    # Even with multiple years, we will designate one as the key base year, which will be used e.g.
    # for determining the resolution, extent, projection and other attributes of the fine resolution
    # lulc data. It must refer to a year that has observed LULC data to load.
    p.key_base_year = 2017


    # If set to one of the other policies, like BAU, this will indicate which scenario to compare the performance
    # of this scenario to. The tag 'no_policy' indicates that it should not be compared to anything.
    p.comparison_counterfactual_labels = 'no_policy'

    # Path to the coarse land-use change data. SEALS supports 2 types: a netcdf directory of geotiffs following
    # the documented structure.
    p.coarse_projections_input_path = "luh2/raw_data/rcp45_ssp2/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-MESSAGE-ssp245-2-1-f_gn_2015-2100.nc"
    # p.coarse_projections_input_path = 'luh2/raw_data/rcp26_ssp1/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-IMAGE-ssp126-2-1-f_gn_2015-2100.nc'

    # Label of the LULC data being reclassified into the simplified form.
    p.lulc_src_label = 'esa'

    # Label of the new LULC simplified classification
    p.lulc_simplification_label = 'seals7'

    # Path to a csv that will map the a many-to-one reclassification of
    # the src LULC map to a simplified version
    # includes at least src_id, dst_id, src_label, dst_label
    p.lulc_correspondence_path = 'seals/default_inputs/esa_seals7_correspondence.csv'

    # Label of the coarse LUC data that will be reclassified and downscaled
    p.coarse_src_label = 'luh2-14'

    # Label of the new coarse LUC data
    p.coarse_simplification_label = 'seals7'

    # Path to a csv that includes at least src_id, dst_id, src_label, dst_label
    p.coarse_correspondence_path = 'seals/default_inputs/luh2-14_seals7_correspondence.csv'

    # Often NetCDF files can have the time dimension in something other than just the year. This string allows
    # for doing operations on the time dimension to match what is desired. e.g., multiply5 add2015
    p.time_dim_adjustment = 'add2015'

    # Because different NetCDF files have different arrangements (e.g. time is in the dimension
    # versus LU_class is in the dimension), this option allows you to specify where in the input
    # NC the information is. If 'all_variables', assumes the LU classes will be the different variables named
    # otherwise it can be a subset of variables, otherwise, if it is a named variable, e.g. LC_area_share
    # then assume that the lc_class variable is stored in the last-listed dimension (see p.dimensions)
    p.lc_class_varname = 'all_variables'

    # Lists which dimensions are stored in the netcdf in addition to lat and lon. Ideally
    # this is just time but sometimes there are more.  # From the csv, this is a space-separated list.
    p.dimensions = 'time'

    # # To speed up processing, select which classes you know won't change. For default seals7, this is
    # # the urban classes, the water classes, and the bare land class.
    # p.nonchanging_class_indices = [0, 6, 7]

    # Path to a csv which contains all of the pretrained regressor variables. Can also
    # be 'from_calibration' indicating that this run will actually create the calibration``
    # or it can be from a tile-designated file of location-specific regressor variables.
    p.calibration_parameters_source = 'seals/default_inputs/default_global_coefficients.csv'

    # Some data, set to default inputs here, are required to make the model run becayse they determine which classes, which resolutions, ...
    p.key_base_year_lulc_simplified_path = os.path.join('lulc', p.lulc_src_label, p.lulc_simplification_label, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + str(p.key_base_year) + '.tif')
    p.key_base_year_lulc_src_path = os.path.join('lulc', p.lulc_src_label, 'lulc_' + p.lulc_src_label  + '_' + str(p.key_base_year) + '.tif')

    # For convenience, set a synonym for the key_base_year_lulc_simplified_path
    p.base_year_lulc_path = p.key_base_year_lulc_src_path


def set_attributes_to_dynamic_default_with_different_inputs(p):
    # Set all ProjectFlow attributes to SEALS default.
    # This is used if the user has never run something before, and therefore doesn't
    # have a scenario_definitions.csv in their input dir.
    # This function will set the attributes, and can be paired with
    # generate_scenarios_csv_and_put_in_input_dir to write the file.

    ###--- SET DEFAULTS ---###

    # String that uniquely identifies the scenario. Will be referenced by other scenarios for comparison.
    p.scenario_label = 'ssp2_rcp45_luh2-globio_bau'

    # Scenario type determines if it is historical (baseline) or future (anything else) as well
    # as what the scenario should be compared against. I.e., Policy minus BAU.
    p.scenario_type = 'bau'

    # Sets the area of interest. If set as a country-ISO3 code, all data will be generated based
    # that countries boundaries (as defined in the base data). Other options include setting it to
    # 'global' or a specific shapefile, or iso3 code. Good small examples include RWA, BTN
    p.aoi = 'RWA'

    # Exogenous label references some set of exogenous drivers like population, TFP growth, LUH2 pattern, SSP database etc
    p.exogenous_label = 'ssp2'

    # One of the climate RCPs
    p.climate_label = 'rcp45'

    # Indicator of which model led to the coarse projection
    p.model_label = 'luh2-message'

    # AKA policy scenario, or a label of something that you have tweaked to assess it's efficacy
    p.counterfactual_label = 'bau'

    # If is not a baseline scenario, these are years into the future. Duplicate the base_year variable below if it is a base year
    # From the csv, this is a space-separated list.
    p.years = [2045, 2075]

    # For calculating difference from base year, this references which baseline (must be consistent due to different
    # models having different depictions of the base year)
    p.baseline_reference_label = 'baseline_luh2-message'

    # Which year in the observed data constitutes the base year. There may be multiple if, for instance, you
    # want to use seals to downscale model outputs that update a base year to a more recent year base year
    # which would now be based on model results but is for an existing year. Paper Idea: Do this for validation.
    p.base_years = [2015, 2017]

    # Even with multiple years, we will designate one as the key base year, which will be used e.g.
    # for determining the resolution, extent, projection and other attributes of the fine resolution
    # lulc data. It must refer to a year that has observed LULC data to load.
    p.key_base_year = 2017


    # If set to one of the other policies, like BAU, this will indicate which scenario to compare the performance
    # of this scenario to. The tag 'no_policy' indicates that it should not be compared to anything.
    p.comparison_counterfactual_labels = 'no_policy'

    # Path to the coarse land-use change data. SEALS supports 2 types: a netcdf directory of geotiffs following
    # the documented structure.
    p.coarse_projections_input_path = "luh2/raw_data/rcp45_ssp2/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-MESSAGE-ssp245-2-1-f_gn_2015-2100.nc"
    # p.coarse_projections_input_path = 'luh2/raw_data/rcp26_ssp1/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-IMAGE-ssp126-2-1-f_gn_2015-2100.nc'

    # Label of the LULC data being reclassified into the simplified form.
    p.lulc_src_label = 'esa'

    # Label of the new LULC simplified classification
    p.lulc_simplification_label = 'seals7'

    # Path to a csv that will map the a many-to-one reclassification of
    # the src LULC map to a simplified version
    # includes at least src_id, dst_id, src_label, dst_label
    p.lulc_correspondence_path = 'seals/default_inputs/esa_seals7_correspondence.csv'

    # Label of the coarse LUC data that will be reclassified and downscaled
    p.coarse_src_label = 'luh2-14'

    # Label of the new coarse LUC data
    p.coarse_simplification_label = 'seals7'

    # Path to a csv that includes at least src_id, dst_id, src_label, dst_label
    p.coarse_correspondence_path = 'seals/default_inputs/luh2-14_seals7_correspondence.csv'

    # Often NetCDF files can have the time dimension in something other than just the year. This string allows
    # for doing operations on the time dimension to match what is desired. e.g., multiply5 add2015
    p.time_dim_adjustment = 'add2015'

    # Because different NetCDF files have different arrangements (e.g. time is in the dimension
    # versus LU_class is in the dimension), this option allows you to specify where in the input
    # NC the information is. If 'all_variables', assumes the LU classes will be the different variables named
    # otherwise it can be a subset of variables, otherwise, if it is a named variable, e.g. LC_area_share
    # then assume that the lc_class variable is stored in the last-listed dimension (see p.dimensions)
    p.lc_class_varname = 'all_variables'

    # Lists which dimensions are stored in the netcdf in addition to lat and lon. Ideally
    # this is just time but sometimes there are more.  # From the csv, this is a space-separated list.
    p.dimensions = 'time'

    # # To speed up processing, select which classes you know won't change. For default seals7, this is
    # # the urban classes, the water classes, and the bare land class.
    # p.nonchanging_class_indices = [0, 6, 7]

    # Path to a csv which contains all of the pretrained regressor variables. Can also
    # be 'from_calibration' indicating that this run will actually create the calibration``
    # or it can be from a tile-designated file of location-specific regressor variables.
    p.calibration_parameters_source = 'seals/default_inputs/default_global_coefficients.csv'

    # Some data, set to default inputs here, are required to make the model run becayse they determine which classes, which resolutions, ...
    p.key_base_year_lulc_simplified_path = os.path.join('lulc', p.lulc_src_label, p.lulc_simplification_label, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + str(p.key_base_year) + '.tif')
    p.key_base_year_lulc_src_path = os.path.join('lulc', p.lulc_src_label, 'lulc_' + p.lulc_src_label  + '_' + str(p.key_base_year) + '.tif')

    # For convenience, set a synonym for the key_base_year_lulc_simplified_path
    p.base_year_lulc_path = p.key_base_year_lulc_src_path



def set_attributes_to_dynamic_many_year_default(p):
    # Set all ProjectFlow attributes to SEALS default.
    # This is used if the user has never run something before, and therefore doesn't
    # have a scenario_definitions.csv in their input dir.
    # This function will set the attributes, and can be paired with
    # generate_scenarios_csv_and_put_in_input_dir to write the file.

    ###--- SET DEFAULTS ---###

    # String that uniquely identifies the scenario. Will be referenced by other scenarios for comparison.
    p.scenario_label = 'ssp2_rcp45_luh2-globio_bau'

    # Scenario type determines if it is historical (baseline) or future (anything else) as well
    # as what the scenario should be compared against. I.e., Policy minus BAU.
    p.scenario_type = 'bau'

    # Sets the area of interest. If set as a country-ISO3 code, all data will be generated based
    # that countries boundaries (as defined in the base data). Other options include setting it to
    # 'global' or a specific shapefile, or iso3 code. Good small examples include RWA, BTN
    p.aoi = 'RWA'

    # Exogenous label references some set of exogenous drivers like population, TFP growth, LUH2 pattern, SSP database etc
    p.exogenous_label = 'ssp2'

    # One of the climate RCPs
    p.climate_label = 'rcp45'

    # Indicator of which model led to the coarse projection
    p.model_label = 'luh2-message'

    # AKA policy scenario, or a label of something that you have tweaked to assess it's efficacy
    p.counterfactual_label = 'bau'

    # If is not a baseline scenario, these are years into the future. Duplicate the base_year variable below if it is a base year
    # From the csv, this is a space-separated list.
    p.years = [2020, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060, 2065, 2070, 2075, 2080, 2085, 2090, 2095, 2100]

    # For calculating difference from base year, this references which baseline (must be consistent due to different
    # models having different depictions of the base year)
    p.baseline_reference_label = 'baseline_luh2-message'

    # Which year in the observed data constitutes the base year. There may be multiple if, for instance, you
    # want to use seals to downscale model outputs that update a base year to a more recent year base year
    # which would now be based on model results but is for an existing year. Paper Idea: Do this for validation.
    p.base_years = [2015, 2017]

    # Even with multiple years, we will designate one as the key base year, which will be used e.g.
    # for determining the resolution, extent, projection and other attributes of the fine resolution
    # lulc data. It must refer to a year that has observed LULC data to load.
    p.key_base_year = 2017


    # If set to one of the other policies, like BAU, this will indicate which scenario to compare the performance
    # of this scenario to. The tag 'no_policy' indicates that it should not be compared to anything.
    p.comparison_counterfactual_labels = 'no_policy'

    # Path to the coarse land-use change data. SEALS supports 2 types: a netcdf directory of geotiffs following
    # the documented structure.
    p.coarse_projections_input_path = "luh2/raw_data/rcp45_ssp2/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-MESSAGE-ssp245-2-1-f_gn_2015-2100.nc"
    # p.coarse_projections_input_path = 'luh2/raw_data/rcp26_ssp1/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-IMAGE-ssp126-2-1-f_gn_2015-2100.nc'

    # Label of the LULC data being reclassified into the simplified form.
    p.lulc_src_label = 'esa'

    # Label of the new LULC simplified classification
    p.lulc_simplification_label = 'seals7'

    # Path to a csv that will map the a many-to-one reclassification of
    # the src LULC map to a simplified version
    # includes at least src_id, dst_id, src_label, dst_label
    p.lulc_correspondence_path = 'seals/default_inputs/esa_seals7_correspondence.csv'

    # Label of the coarse LUC data that will be reclassified and downscaled
    p.coarse_src_label = 'luh2-14'

    # Label of the new coarse LUC data
    p.coarse_simplification_label = 'seals7'

    # Path to a csv that includes at least src_id, dst_id, src_label, dst_label
    p.coarse_correspondence_path = 'seals/default_inputs/luh2-14_seals7_correspondence.csv'

    # Often NetCDF files can have the time dimension in something other than just the year. This string allows
    # for doing operations on the time dimension to match what is desired. e.g., multiply5 add2015
    p.time_dim_adjustment = 'add2015'

    # Because different NetCDF files have different arrangements (e.g. time is in the dimension
    # versus LU_class is in the dimension), this option allows you to specify where in the input
    # NC the information is. If 'all_variables', assumes the LU classes will be the different variables named
    # otherwise it can be a subset of variables, otherwise, if it is a named variable, e.g. LC_area_share
    # then assume that the lc_class variable is stored in the last-listed dimension (see p.dimensions)
    p.lc_class_varname = 'all_variables'

    # Lists which dimensions are stored in the netcdf in addition to lat and lon. Ideally
    # this is just time but sometimes there are more.  # From the csv, this is a space-separated list.
    p.dimensions = 'time'

    # # To speed up processing, select which classes you know won't change. For default seals7, this is
    # # the urban classes, the water classes, and the bare land class.
    # p.nonchanging_class_indices = [0, 6, 7]

    # Path to a csv which contains all of the pretrained regressor variables. Can also
    # be 'from_calibration' indicating that this run will actually create the calibration``
    # or it can be from a tile-designated file of location-specific regressor variables.
    p.calibration_parameters_source = 'seals/default_inputs/default_global_coefficients.csv'

    # Some data, set to default inputs here, are required to make the model run becayse they determine which classes, which resolutions, ...
    p.key_base_year_lulc_simplified_path = os.path.join('lulc', p.lulc_src_label, p.lulc_simplification_label, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + str(p.key_base_year) + '.tif')
    p.key_base_year_lulc_src_path = os.path.join('lulc', p.lulc_src_label, 'lulc_' + p.lulc_src_label  + '_' + str(p.key_base_year) + '.tif')

    # For convenience, set a synonym for the key_base_year_lulc_simplified_path
    p.base_year_lulc_path = p.key_base_year_lulc_src_path


def calc_observed_lulc_change_for_two_lulc_paths(lulc_1_path, lulc_2_path, coarse_ha_per_cell_path, classes_that_might_change, output_dir):

    lulc_paths = [lulc_1_path, lulc_2_path]


    # ha_per_cell_15m = hb.ArrayFrame(p.global_ha_per_cell_15m_path)

    # TODOO, current problems: Change vector method needs to be expanded to Change matrix, full from-to relationships
    # but when doing from-to, that only works when doing observed time-period validation. What would be the assumption for going into
    # the future? Possibly attempt to match prior change matrices, but only as a slight increase in probability? Secondly, why is my
    # search algorithm not itself finding the from-to relationships just by minimizing difference? Basically, need to take seriously deallocation.

    full_change_matrix_no_diagonal_path = os.path.join(output_dir, 'full_change_matrix_no_diagonal.tif')

    from hazelbean.calculation_core.cython_functions import \
        calc_change_matrix_of_two_int_arrays

    # if p.run_this and not os.path.exists(full_change_matrix_no_diagonal_path):
    # Clip all 30km change paths, then just use the last one to set the propoer (coarse) extent of the lulc.
    lulc_1 = hb.as_array(lulc_1_path)
    lulc_2 = hb.as_array(lulc_2_path)
    coarse_ha_per_cell = hb.as_array(coarse_ha_per_cell_path)

    # # Clip all 30km change paths, then just use the last one to set the propoer (coarse) extent of the lulc.
    # for c, path in enumerate(p.global_lulc_paths):
    #     hb.load_geotiff_chunk_by_cr_size(path, p.fine_blocks_list, output_path=p.lulc_paths[c])

    # lulc_afs = [hb.ArrayFrame(path) for path in p.lulc_paths]

    fine_cell_size = hb.get_cell_size_from_path(lulc_1_path)
    coarse_cell_size = hb.get_cell_size_from_path(coarse_ha_per_cell_path)



    fine_cells_per_coarse_cell = round((coarse_cell_size/ fine_cell_size) ** 2)
    aspect_ratio = int(lulc_1.shape[1] / coarse_ha_per_cell.shape[1])

    net_change_output_arrays = np.zeros((len(classes_that_might_change), coarse_ha_per_cell.shape[0], coarse_ha_per_cell.shape[1]))

    full_change_matrix = np.zeros((len(classes_that_might_change * coarse_ha_per_cell.shape[0]), len(classes_that_might_change) * coarse_ha_per_cell.shape[1]))
    full_change_matrix_no_diagonal = np.zeros((len(classes_that_might_change * coarse_ha_per_cell.shape[0]), len(classes_that_might_change) * coarse_ha_per_cell.shape[1]))

    for r in range(coarse_ha_per_cell.shape[0]):
        for c in range(coarse_ha_per_cell.shape[1]):

            t1_subarray = lulc_1[int(r * aspect_ratio): int((r + 1) * aspect_ratio), int(c * aspect_ratio): int((c + 1) * aspect_ratio)]
            t2_subarray = lulc_2[int(r * aspect_ratio): int((r + 1) * aspect_ratio), int(c * aspect_ratio): int((c + 1) * aspect_ratio)]
            # ha_per_coarse_cell_this_subarray = p.coarse_cell_size.data[r, c]

            change_matrix, counters = calc_change_matrix_of_two_int_arrays(t1_subarray.astype(np.int32), t2_subarray.astype(np.int32), classes_that_might_change)
            vector = calc_change_vector_of_change_matrix(change_matrix)

            ha_per_cell_this_subarray = coarse_ha_per_cell[r, c] / fine_cells_per_coarse_cell

            if vector:
                for i in classes_that_might_change:
                    net_change_output_arrays[i - 1, r, c] = vector[i - 1] * ha_per_cell_this_subarray
            else:
                net_change_output_arrays[i, r, c] = 0.0

            n_classes = len(classes_that_might_change)
            full_change_matrix[r * n_classes: (r + 1) * n_classes, c * n_classes: (c + 1) * n_classes] = change_matrix

            # Fill diagonal with zeros.
            for i in range(n_classes):
                change_matrix[i, i] = 0

            full_change_matrix_no_diagonal[r * n_classes: (r + 1) * n_classes, c * n_classes: (c + 1) * n_classes] = change_matrix

    for c, i in enumerate(classes_that_might_change):
        current_net_change_array_path = os.path.join(output_dir, str(i) + '_observed_change.tif')
        hb.save_array_as_geotiff(net_change_output_arrays[c], current_net_change_array_path, coarse_ha_per_cell_path)

    full_change_matrix_path = os.path.join(output_dir, 'full_change_matrix.tif')
    hb.save_array_as_geotiff(full_change_matrix, full_change_matrix_path, coarse_ha_per_cell_path, n_rows=coarse_ha_per_cell.shape[0] * n_classes, n_cols=coarse_ha_per_cell.shape[1] * n_classes)

    full_change_matrix_no_diagonal_path = os.path.join(output_dir, 'full_change_matrix_no_diagonal.tif')
    hb.save_array_as_geotiff(full_change_matrix_no_diagonal, full_change_matrix_no_diagonal_path, coarse_ha_per_cell_path, n_rows=coarse_ha_per_cell.shape[0] * n_classes, n_cols=coarse_ha_per_cell.shape[1] * n_classes)

    # p.projected_cooarse_change_files = hb.list_filtered_paths_nonrecursively(p.projected_coarse_change_dir, include_extensions='.tif')
    # for path in p.projected_cooarse_change_files:
    #     file_front_int = os.path.split(path)[1].split('_')[0]
    #     current_net_change_array_path = os.path.join(p.cur_dir, str(file_front_int) + '_projected_change.tif')

    #     # TODO Get rid of all this wasteful writing.
    #     hb.load_geotiff_chunk_by_bb(path, p.coarse_blocks_list, output_path=current_net_change_array_path)
    # for c, i in enumerate(p.classes_that_might_change):
    #     projected_change_global_path = os.path.join(p.projected_coarse_change_dir, str(i) )
    #     current_net_change_array_path = os.path.join(p.cur_dir, str(i) + '_projected_change.tif')
    #     # hb.save_array_as_geotiff(net_change_output_arrays[c], current_net_change_array_path, p.coarse_match.path)
    #     hb.load_geotiff_chunk_by_bb(p.global_ha_per_cell_15m_path, p.coarse_blocks_list, output_path=current_net_change_array_path)


    # full_change_matrix_path = os.path.join(p.cur_dir, 'full_change_matrix.tif')
    # hb.save_array_as_geotiff(full_change_matrix, full_change_matrix_path, p.coarse_match.path, n_rows=full_change_matrix.shape[1], n_cols=full_change_matrix.shape[1])
    # full_change_matrix_no_diagonal_path = os.path.join(p.cur_dir, 'full_change_matrix_no_diagonal.tif')
    # hb.save_array_as_geotiff(full_change_matrix_no_diagonal, full_change_matrix_no_diagonal_path, p.coarse_match.path, n_rows=full_change_matrix_no_diagonal.shape[1], n_cols=full_change_matrix_no_diagonal.shape[1])


# p.ha_per_cell_15m = None

def load_blocks_list(p, input_dir):
    possible_prefixes = [
        "coarse_",
        "fine_",
        "global_coarse_",
        "global_fine_",
        "global_processing_",
        "processing_",
    ]
    for prefix in possible_prefixes:
        file_root = prefix + 'blocks_list'
        possible_path = os.path.join(input_dir, file_root + '.csv')
        if hb.path_exists(possible_path):
            # Load the csv but add column headers just based on increasing integers using pandas
            blocks_list = list(hb.file_to_python_object(possible_path, '2d_list'))
            setattr(p, file_root, blocks_list)
        else:
            raise NameError('Unable to load block lists in dir: ' + input_dir)



def convert_regional_change_to_coarse(regional_change_vector_path, regional_change_classes_path, coarse_ha_per_cell_path, scenario_label, output_dir, output_filename_end, columns_to_process, regions_column_label, years, region_ids_raster_path=None,  distribution_algorithm='proportional', coarse_change_raster_path=None):
    # Converts a regional vector change map to a coarse gridded representation. 
    # - Regiona_change_vector_path is a gpkg with the boundaries of the regions that are changing.
    # - regional_change_classes_path is a csv with columns for each changing landuse class (in columns to process) that reports net change in hectares per polygon.
    # - Coarse_ha_per_cell_path is a raster that reports the number of hectares per cell in the coarse grid. Necessary to work with unprojected data, which is our assumed form.
    # - Output path is the path to save the output raster.
    # - Columns to process is a list of the columns headers in the gpkg that represent net changes in LU classes.
    # - distribution_algorithm is a string that indicates how to distribute the change across the cells. default is "proportional".
    # - Coarse_change_raster_path is an optional path that, if provided, will be combined with the regional change to produce a combined change raster.  
    
    # Read both inputs                          
    regional_change_vector = gpd.read_file(regional_change_vector_path)    
    regional_change_classes = pd.read_csv(regional_change_classes_path)
   
    # Merge them, but note awkwardness that there are two different merge types, on label and on id. 
    # BOTH are required because in gtappy we are concatenating AEZ and reg_id
    if str(regions_column_label).endswith('_label'):
        regional_change_vector[regions_column_label] = regional_change_vector[regions_column_label].astype(str).str.upper()
        regional_change_vector['region_label'] = regional_change_vector[regions_column_label]        
        regional_change_classes[regions_column_label] = regional_change_classes[regions_column_label].astype(str).str.upper()
        regional_change_classes['region_label'] = regional_change_classes[regions_column_label]        
        merged = hb.df_merge_quick(regional_change_vector, regional_change_classes, left_on='region_label', right_on='region_label', how='inner')
    elif str(regions_column_label).endswith('_id'):
        regional_change_vector[regions_column_label] = regional_change_vector[regions_column_label].astype(int)
        regional_change_vector['region_label'] = regional_change_vector[regions_column_label]
        regional_change_classes[regions_column_label] = regional_change_classes[regions_column_label].astype(int)
        regional_change_classes['region_label'] = regional_change_classes[regions_column_label]        
        merged = hb.df_merge_quick(regional_change_vector, regional_change_classes, left_on='region_label', right_on='region_label', how='inner')   
    else:
        raise NameError('Regions column label must end with _label or _id')

    
    if region_ids_raster_path is None:
        region_ids_raster_path = os.path.join(output_dir, 'region_ids.tif')
        
    merged_vector_output_path = os.path.join(os.path.split(region_ids_raster_path)[0], 'regional_change_with_ids_' + scenario_label + '.gpkg')
    

    merged.to_file(merged_vector_output_path, driver='GPKG')
    
    ### Rasterize the regions vector to a raster.
    
    # This is the one case where it's okay to infer a name. labels to ids always exist together
    regions_column_id = regions_column_label.replace('label', 'id')
    # Make the region_ids raster if it doesn't exist
    if not hb.path_exists(region_ids_raster_path):

        # TODOO NOTE that here we are not using all_touched. This is a fundamental problem with coarse reclassification. Lots of the polygon will be missed. Ideally, you use all_touched=False for
        # country-country borders but all_touched=True for country-coastline boarders. Or join with EEZs?
        hb.rasterize_to_match(merged_vector_output_path, coarse_ha_per_cell_path, region_ids_raster_path, burn_column_name=regions_column_id, burn_values=None, datatype=13, ndv=0, all_touched=False)

    ### Get the number of cells per zone. 
    
    # We need to know how big the zone is in terms of coarse cells so we can calculate how much of the total change happens in each coarse gridcell    
    # TODOOO: Think about how I should deal with giving the whole regional_change_vector or if I should have it subset out the line it needs, cause this is a utility function.
    n_cells_per_zone = hb.enumerate_raster_path(region_ids_raster_path)

    ### Define the allocation of the total to individual cells
    
    # Creates a dict for each zone_id: to_allocate, which will be reclassified onto the zone ids.
    # NOTE: This function is called once per year by regional_change(), with output_dir and
    # output_filename_end already set for that specific year. The year loop here is kept for
    # backward compatibility but we filter the merged data to the correct year when possible.
    # The caller passes `years` (all scenario years) — we use `output_filename_end` to infer
    # which year the caller wants.

    # Extract the target year from output_filename_end (format: _YYYY_PREVYYYY_ha_diff_...)
    import re as _re
    _year_match = _re.match(r'_(\d{4})_', output_filename_end)
    _target_year = int(_year_match.group(1)) if _year_match else None

    for year_c, year in enumerate(years):
        allocate_per_zone_dict = {}

        # Filter to target year if the CSV has a 'year' column (multi-year regional projections).
        # Without this filter, the inner loop overwrites allocate_per_zone_dict for each year's row,
        # and only the last year's values survive.
        if 'year' in merged.columns and _target_year is not None:
            merged_this_year = merged[merged['year'] == _target_year]
        else:
            merged_this_year = merged

        for column in columns_to_process:
            output_path = os.path.join(output_dir, column + output_filename_end)

            if not hb.path_exists(output_path):
                hb.log('Processing ' + column + ' for ' + scenario_label + ',  writing to ' + output_path)
                regions_column_id = regions_column_id.replace('label', 'id')


                for i, change in merged_this_year[column].items():
                    zone_id = int(merged_this_year[regions_column_id][i])
                    
                    if int(zone_id) in n_cells_per_zone:
                        n_cells = n_cells_per_zone[int(zone_id)] # BAD HACK, should be generalized to know ahead of time if it's an int or string
                    else:
                        n_cells = 0
                        
                    if n_cells > 0  and change != 0:
                        result = change / n_cells
                    else:
                        result = 0.0

                    if 'nan' in str(result).lower():
                        allocate_per_zone_dict[zone_id] = 0.0
                    else: 
                        allocate_per_zone_dict[zone_id] = result
                            
                print('Allocate per zone dict for ' + column + ': ' + str(allocate_per_zone_dict))
                hb.reclassify_raster_hb(region_ids_raster_path, allocate_per_zone_dict, output_path, output_data_type=7, match_path=None, invoke_full_callback=False, existing_values='zero', verbose=False)
                    
def combine_coarsified_regional_with_coarse_estimate(coarsified_path, coarse_estimate_path, combination_algorithm, output_path):
    ### ABANDONED?
    if combination_algorithm == 'local_sum':
        def local_sum(a, b):
            return a + b    
        hb.raster_calculator_flex([coarsified_path, coarse_estimate_path], local_sum, output_path)
    elif combination_algorithm == 'covariate_regavg_shift':
        def covariate_regavg_shift(a, b):
            return a + b
        hb.raster_calculator_flex([coarsified_path, coarse_estimate_path], covariate_regavg_shift, output_path)




# Columns a coefficient table carries that are not classes. Everything else is one, under
# either naming convention: older files name the column after the class, newer ones prefix
# it with class_.
NON_CLASS_COEFFICIENT_COLUMNS = frozenset(
    ['spatial_regressor_name', 'data_location', 'type', 'calibration_block_index'])


def coefficient_class_columns(coefficients_df):
    """The column names holding per-class values, under either naming convention."""
    return [c for c in coefficients_df.columns
            if c not in NON_CLASS_COEFFICIENT_COLUMNS and not str(c).startswith('Unnamed')]


def coefficient_class_labels(coefficients_df):
    """The class labels a coefficient table was fitted for, in column order."""
    return [c[len('class_'):] if str(c).startswith('class_') else c
            for c in coefficient_class_columns(coefficients_df)]


def check_coefficients_match_class_scheme(coefficients_df, changing_class_labels,
                                          coefficients_path=None):
    """Raise if a coefficient table was fitted under a different class scheme.

    A coefficient file only means anything alongside the correspondence it was fitted with.
    Class ids move between schemes, because SEALS requires the classes that can change to
    come first, so inserting one shifts every class after it. Reading a file under the wrong
    scheme therefore misassigns classes **silently** and produces a plausible-looking map
    rather than an error.

    Only membership is checked, not column order. The per-class columns are selected by name
    in the order the correspondence defines, so a file carrying the same classes in a
    different order is read correctly.
    """
    found = coefficient_class_labels(coefficients_df)
    expected = list(changing_class_labels)

    missing = [c for c in expected if c not in found]

    # Extra columns are not an error. The per-class columns are selected by name, so a file
    # carrying classes this run does not need is read correctly and the surplus ignored. That
    # is a real configuration: a coarse correspondence can route one class into another, so a
    # run legitimately needs fewer classes than the file it was fitted with.
    if not missing:
        return

    where = ' in ' + str(coefficients_path) if coefficients_path else ''
    reason = ('the correspondence expects %s, which the file does not carry'
              % ', '.join(missing))

    raise ValueError(
        'The trained coefficients%s were fitted for a different class scheme than the '
        'correspondence this run uses: %s.\n'
        '  correspondence: %s\n'
        '  coefficients:   %s\n'
        'Use the coefficient set fitted for this scheme, or point the scenario at the '
        'correspondence these coefficients were fitted with.'
        % (where, reason, ', '.join(expected) or '(none)', ', '.join(found) or '(none)'))


def assert_calibration_matches_processing_grid(coefficients_df, processing_block_ids,
                                              processing_resolution, coefficients_path,
                                              minimum_coverage=0.5):
    """Refuse a coefficient file whose block keys were fitted on a different grid than this run.

    The per-zone calibration key is the zone's own name -- its column and row on the processing grid
    -- with a constant suffix. Nothing in that string records WHICH grid it counts on, so a file
    fitted at one processing resolution and used at another still matches as a string, for the
    minority of names that happen to exist in both index spaces, and silently supplies coefficients
    belonging to a different part of the world. The zones that match nothing get an empty table, no
    spatial regressor, and therefore no allocation at all: their land-use demand is dropped without
    any error.

    That is exactly what happened on this project. Coefficients fitted at 1 degree (columns 0..359)
    were used by a run tiled at 4 degrees (columns 0..89): 56% of changing zones allocated nothing,
    and the 44% that matched drew coefficients from elsewhere -- a zone over Honduras took its
    parameters from a block on the Alaskan North Slope. Every map was affected and nothing failed.

    Two tests, and NEITHER proves the geography is right. Be precise about what each one shows.

    The first detects an incompatible index EXTENT: a calibration key whose column or row lies
    outside the run's own grid could not have been produced on that grid, so the file is fitted on a
    finer one. This catches the 1-degree-file-at-4-degrees case, which coverage alone misses badly --
    such a file still supplies keys for 87% of the 4-degree zones, because columns 0..89 are a subset
    of columns 0..359, and those keys match as strings while pointing somewhere else entirely.

    **The absence of out-of-range keys does NOT establish that the grids agree.** A file on a coarser
    grid, or on one anchored differently with the same extent, produces no out-of-range key and would
    pass. This test is a cheap necessary condition checked at load time, not a proof.

    The second catches the coarse or regional case, where too few of this run's zones find any key.

    Geographic correctness is established SEPARATELY, per zone, by comparing each tile's own bounds
    against the bounds its key implies on the calibration grid -- see the `--audit-all-lookups` pass
    in the reconciliation tooling, whose `same_location` result is the actual evidence. Keep the two
    apart: this guard stops an obviously incompatible file before a run burns hours; the audit is
    what shows the coefficients belonged to the places that used them.

    Args:
        coefficients_df (pd.DataFrame): the loaded coefficients; a table with no
            calibration_block_index column addresses every zone and is not checked.
        processing_block_ids (iterable): the zone names this run will use, as '<column>_<row>'.
        processing_resolution (float): degrees per processing block, for the message.
        coefficients_path (str): named in the message, because the remedy is a different file or a
            different resolution and the reader needs to know which file was read.
        minimum_coverage (float): the share of this run's zones that must find a key.

    Raises:
        NameError: naming the coverage, the implied grids and the two ways out.
    """
    import numpy as np

    if 'calibration_block_index' not in getattr(coefficients_df, 'columns', []):
        return None
    keys = {str(i) for i in coefficients_df['calibration_block_index'].dropna().unique()}
    zone_ids = [str(i) for i in processing_block_ids]
    if not keys or not zone_ids:
        return None
    wanted = {z + '_1_1' for z in zone_ids}
    matched = wanted & keys
    coverage = len(matched) / len(wanted)

    columns = [int(k.split('_')[0]) for k in keys]
    rows = [int(k.split('_')[1]) for k in keys]
    implied = 360.0 / (max(columns) + 1)
    run_max_column = int(360 / processing_resolution) - 1
    run_max_row = int(180 / processing_resolution) - 1
    outside = [k for k, c, r in zip(sorted(keys), columns, rows)
               if c > run_max_column or r > run_max_row]
    if outside:
        raise NameError(
            'The trained coefficients were fitted on a FINER processing grid than this run.\n'
            '  coefficients: %s\n'
            '  %d of %d keys lie outside this run\'s grid, e.g. %s\n'
            '  their columns reach %d, implying about %g degrees per block\n'
            '  this run is tiled at %g degrees, so no zone index can exceed %d_%d\n'
            '  %.1f%% of zones still find a key BY COINCIDENCE of naming, and those keys belong to\n'
            '  a different part of the world -- a key is a grid position with no grid attached.\n'
            'Either set p.processing_resolution to %g to match the calibration, or point\n'
            'calibration_parameters_source at a file fitted for %g degrees.'
            % (coefficients_path, len(outside), len(keys), outside[:3], max(columns), implied,
               processing_resolution, run_max_column, run_max_row, 100 * coverage, implied,
               processing_resolution))

    if coverage >= minimum_coverage:
        return coverage

    uncovered = sorted(z for z in zone_ids if z + '_1_1' not in keys)
    if minimum_coverage >= 1.0 and len(uncovered) < 0.02 * len(wanted):
        # The extent test above passed, so these keys are not from another grid. The table was
        # never fitted for these blocks, and a zone without coefficients allocates nothing.
        raise NameError(
            'The trained coefficients do not cover every tile this run will allocate.\n'
            '  coefficients: %s\n'
            '  %d of %d scheduled zones (%.4f%%) have no calibration key\n'
            '  uncovered zones: %s\n'
            'Their block indices lie inside this run\'s grid, so this is not a resolution\n'
            'mismatch -- the table was never fitted for them. A zone with no coefficients\n'
            'allocates nothing and drops its land-use demand silently. Fill these blocks\n'
            '(for example from the nearest fitted block) and point\n'
            'calibration_parameters_source at the filled table.'
            % (coefficients_path, len(uncovered), len(wanted), 100 * (1 - coverage), uncovered))

    raise NameError(
        'The trained coefficients were fitted on a different processing grid than this run.\n'
        '  coefficients: %s\n'
        '  their block columns run 0..%d, implying about %g degrees per block\n'
        '  this run is tiled at %g degrees, giving zone columns 0..%d\n'
        '  only %d of %d zones (%.1f%%) find a calibration key, below the required %.0f%%\n'
        'A key is a grid position with no grid attached, so the names still match for the few that\n'
        'coincide -- and those take coefficients from a different place. Either set\n'
        'p.processing_resolution to %g to match the calibration, or point\n'
        'calibration_parameters_source at a file fitted for %g degrees.'
        % (coefficients_path, max(columns), implied, processing_resolution,
           max(int(z.split('_')[0]) for z in zone_ids), len(matched), len(wanted),
           100 * coverage, 100 * minimum_coverage, implied, processing_resolution))


def protected_class_labels(all_class_labels, changing_class_labels,
                           additional_protected_class_labels=None):
    """The classes nothing may be allocated onto.

    Two different things, which is why no single rule produces the set:

    The non-changing classes are derived. A class is non-changing if it appears in the
    land-cover correspondence but not in the coarse one, meaning SEALS is given no budget
    for it. Letting another class expand onto one would shrink it with nothing authorising
    the loss, so the map would stop conserving area. That argument holds for any scheme, and
    it is also what lets a project add a no-expansion area, a solar plant say, purely by
    listing it in its own correspondence.

    Anything further is a scenario statement and has to be named. Urban is the standing
    example: it has a budget and does expand, so it is not non-changing, but built land is
    not un-built and the allocator places expansion only, so urban taken by another class
    could never be recovered.
    """
    protected = [c for c in all_class_labels if c not in set(changing_class_labels)]
    for label in (additional_protected_class_labels or []):
        if label not in protected:
            protected.append(label)
    return protected


def resolve_additional_protected_class_labels(p):
    """The classes this SCENARIO declares protected, from the scenario row or the project.

    Belongs in the scenario CSV rather than on the project, because protecting a class is a
    scenario statement, not a property of the correspondences. Protecting urban and not
    protecting it are two scenarios of one study, and with the setting on the project they
    cannot sit in one scenarios CSV -- the Brazil work had to stand up a second project purely
    to flip it, which is the level being wrong.

    assign_df_row_to_object_attributes already puts every scenario column on p, so a column
    named additional_protected_class_labels arrives here as a string. Accepts a list (project
    default), a space- or comma-separated string (scenario column), or blank/NaN for none.
    """
    value = getattr(p, 'additional_protected_class_labels', None)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    # A blank cell arrives as NaN through pandas.
    if not text or text.lower() in ('nan', 'none'):
        return []
    return [part.strip() for part in text.replace(',', ' ').split() if part.strip()]


def _stitched_check_record_path(projected_path):
    return projected_path + '.non_changing_check.json'


def _stitched_check_record(projected_path, base_path):
    """The record of a passed non-changing check: size and mtime of the two rasters it read."""
    record = {}
    for label, path in (('map', projected_path), ('base', base_path)):
        st = os.stat(path)
        record[label] = {'path': str(path), 'size': st.st_size, 'mtime': st.st_mtime}
    return record


def stitched_map_already_checked(projected_path, base_path):
    """True when the non-changing-classes check already passed on these very files (same size and
    mtime of map and base). A rebuilt or touched map is checked again; a missing record means the
    map predates the record and is checked once more."""
    record_path = _stitched_check_record_path(projected_path)
    if not os.path.exists(record_path):
        return False
    try:
        with open(record_path, encoding='utf-8') as f:
            old = json.load(f)
    except (OSError, ValueError):
        return False
    return old == _stitched_check_record(projected_path, base_path)


def record_stitched_map_checked(projected_path, base_path):
    """Write the record a passed check leaves beside the map."""
    with open(_stitched_check_record_path(projected_path), 'w', encoding='utf-8') as f:
        json.dump(_stitched_check_record(projected_path, base_path), f, indent=2)


def assert_non_changing_classes_unchanged(projected_path, base_path, all_class_labels,
                                          all_class_indices, changing_class_labels,
                                          additional_protected_class_labels=None):
    """Raise if the allocation moved land out of a class it was forbidden to touch.

    The companion to check_coefficients_match_class_scheme. That one guards the INPUT, that
    the coefficients belong to this scheme; this guards the OUTPUT, that the allocation
    respected the presence constraints. They fail independently: a malformed constraint layer
    or a kernel change violates only this one.

    Worth asserting rather than reviewing, because the failure is invisible. A contaminated
    run leaks a few hundred pixels out of a hundred thousand changed -- a well-formed map, no
    exception, no NoData, nothing to grep for, and far too small to see on a 300 m map.

    Two claims, deliberately separated, because they are true for different reasons and a
    future config change should read as a decision rather than a bug:

      derived   A non-changing class has no coarse budget, so nothing authorises a loss and
                the map would stop conserving area. True for any scheme, always.
      declared  An additionally protected class (urban is the standing case) DOES have a
                budget and does expand. It is frozen by a scenario choice, so this half can
                legitimately be relaxed by changing that choice.

    Classes are resolved BY LABEL, never by id. Ids shift between schemes -- water is 6 under
    seals7 and 7 under seals8 -- so a hardcoded id would check the wrong class and pass
    confidently, which is the very failure this module exists to prevent.
    """
    import numpy as np
    from osgeo import gdal

    label_to_index = dict(zip(all_class_labels, all_class_indices))
    derived = protected_class_labels(all_class_labels, changing_class_labels)
    declared = [c for c in (additional_protected_class_labels or []) if c not in derived]

    if not derived and not declared:
        return

    projected_ds, base_ds = gdal.Open(projected_path), gdal.Open(base_path)
    if projected_ds is None or base_ds is None:
        hb.log('  non-changing check: could not open both rasters; NOT CHECKED')
        return
    if (projected_ds.RasterXSize, projected_ds.RasterYSize) != (base_ds.RasterXSize, base_ds.RasterYSize):
        # A forced-global-bb run legitimately differs in extent. Say so loudly rather than
        # skipping quietly -- an unrun check must never look like a passed one.
        hb.log('  non-changing check: NOT CHECKED, extent differs (%s vs %s). %s'
               % ((projected_ds.RasterXSize, projected_ds.RasterYSize),
                  (base_ds.RasterXSize, base_ds.RasterYSize), projected_path))
        return

    # Refuse to compare rasters from different class schemes. The raw source LULC and the
    # simplified map use the same small integers for DIFFERENT classes -- MapBiomas 6 is
    # floodable_forest where seals7 6 is water -- so comparing them reports millions of
    # impossible conversions with total confidence. Handing this the raw map instead of the
    # simplified one is an easy mistake; it was made, and a live run caught it. Sample the
    # base cheaply and bail if it carries values this scheme does not define.
    known = set(int(i) for i in all_class_indices)
    sample = base_ds.GetRasterBand(1).ReadAsArray(
        0, 0, base_ds.RasterXSize, base_ds.RasterYSize,
        buf_xsize=max(1, min(base_ds.RasterXSize, 512)),
        buf_ysize=max(1, min(base_ds.RasterYSize, 512)))
    seen = set(int(v) for v in np.unique(sample) if v > 0)
    stray = sorted(seen - known)
    if stray:
        hb.log('  non-changing check: NOT CHECKED, the base map carries values this scheme does '
               'not define (%s). It looks like the RAW source LULC rather than the simplified '
               'map. base: %s' % (', '.join(str(s) for s in stray[:8]), base_path))
        projected_ds = base_ds = None
        return

    wanted = [(c, label_to_index[c]) for c in derived + declared if c in label_to_index]
    left = {label: 0 for label, _ in wanted}
    destinations = {label: {} for label, _ in wanted}

    projected_band, base_band = projected_ds.GetRasterBand(1), base_ds.GetRasterBand(1)
    n_rows, n_cols = projected_ds.RasterYSize, projected_ds.RasterXSize
    step = max(1, min(2048, n_rows))
    for row in range(0, n_rows, step):
        rows = min(step, n_rows - row)
        after = projected_band.ReadAsArray(0, row, n_cols, rows)
        before = base_band.ReadAsArray(0, row, n_cols, rows)
        for label, index in wanted:
            moved = (before == index) & (after != index) & (after > 0)
            n = int(moved.sum())
            if not n:
                continue
            left[label] += n
            for value, count in zip(*np.unique(after[moved], return_counts=True)):
                destinations[label][int(value)] = destinations[label].get(int(value), 0) + int(count)
    projected_ds = base_ds = None

    index_to_label = {v: k for k, v in label_to_index.items()}

    def describe(label):
        top = sorted(destinations[label].items(), key=lambda kv: -kv[1])[:3]
        return '%s: %d px -> %s' % (label, left[label],
                                    ', '.join('%s %d' % (index_to_label.get(v, v), c) for v, c in top))

    broken_derived = [c for c in derived if left.get(c)]
    broken_declared = [c for c in declared if left.get(c)]

    if broken_derived:
        raise ValueError(
            'The allocation moved land OUT of a non-changing class, which cannot be right: '
            'these classes have no coarse budget, so nothing authorises the loss and the map '
            'no longer conserves area.\n'
            '  %s\n'
            '  map:  %s\n'
            '  base: %s\n'
            'Non-changing classes are those in the land-cover correspondence but absent from '
            'the coarse one. Check that the presence-constraint rows for them are zeroed in '
            'the coefficient file this run used.'
            % ('\n  '.join(describe(c) for c in broken_derived), projected_path, base_path))

    if broken_declared:
        raise ValueError(
            'The allocation moved land OUT of a class this scenario declared protected via '
            'additional_protected_class_labels.\n'
            '  %s\n'
            '  map:  %s\n'
            'Unlike a non-changing class this one does have a coarse budget, so the freeze is '
            'a scenario choice rather than an accounting requirement. Either the constraint '
            'did not reach the coefficients, or the class should no longer be listed as '
            'protected -- which is a decision, not a bug.'
            % ('\n  '.join(describe(c) for c in broken_declared), projected_path))


def apply_presence_constraints(coefficients_df, all_class_labels, changing_class_labels,
                               additional_protected_class_labels=None):
    """Zero the multiplicative rows of the classes nothing may be allocated onto.

    A presence constraint says "no expanding class may take this cell". The rows are written
    for every class when the starting values are generated; this sets the protected ones to
    zero and leaves the rest neutral at 1.0.

    Calibration cannot produce these. It fits coefficients only for the classes that change
    and never learns that a class must be excluded, so the constraint rows come out of it
    neutral and are imposed afterwards.

    See protected_class_labels for which classes those are and why.

    The fitted coefficients are never touched, so the result shares its calibration with the
    input and the two remain directly comparable.
    """
    excluded = protected_class_labels(all_class_labels, changing_class_labels,
                                      additional_protected_class_labels)

    out = coefficients_df.copy()
    class_columns = coefficient_class_columns(out)
    if not class_columns:
        raise ValueError('no per-class columns found; this does not look like a coefficient table')

    # Two naming conventions are in circulation: newer tables call the row
    # '<class>_presence_constraint', older ones '<class>_constraint'. Match either, so this
    # works on a file whichever generated it.
    present = set(out.loc[out['type'] == 'multiplicative', 'spatial_regressor_name'])

    def constraint_row_name(label):
        """Newer tables name the row '<class>_presence_constraint', older ones '<class>_constraint'."""
        for candidate in (label + '_presence_constraint', label + '_constraint'):
            if candidate in present:
                return candidate
        return None

    constraint_names = [constraint_row_name(c) for c in excluded]
    missing = [c for c, name in zip(excluded, constraint_names) if name is None]
    constraint_names = [n for n in constraint_names if n]

    if missing:
        raise ValueError('no constraint row to zero for: %s. The class must be in the land-cover '
                         'correspondence so the row is generated.' % ', '.join(missing))

    # Only the CLASS-constraint rows belong to this function. A multiplicative row named for
    # something else is a scenario layer -- a protection mask, say -- carrying an assumption
    # this function knows nothing about, so resetting it would silently switch that assumption
    # off. Verified: rebuilding over a 30by30 protection row turned 0,0,0,1,1 into 1,1,1,1,1,
    # disabling the protection with no error and a perfectly plausible map.
    class_constraint_names = {n for n in (constraint_row_name(c) for c in all_class_labels) if n}
    constraint_rows = ((out['type'] == 'multiplicative')
                       & out['spatial_regressor_name'].isin(class_constraint_names))
    column_for = dict(zip(coefficient_class_labels(out), class_columns))

    out.loc[constraint_rows, class_columns] = 1.0

    for label in changing_class_labels:
        name = constraint_row_name(label)
        if name:
            out.loc[constraint_rows & (out['spatial_regressor_name'] == name),
                    column_for[label]] = 0.0

    out.loc[constraint_rows & out['spatial_regressor_name'].isin(constraint_names),
            class_columns] = 0.0

    fitted = out['type'] != 'multiplicative'
    if not out.loc[fitted, class_columns].equals(coefficients_df.loc[fitted, class_columns]):
        raise ValueError('fitted coefficients changed; only constraint rows may be modified')

    return out


def resolve_constraint_layers(coefficients_df, fine_processed_inputs_dir, lulc_src_label,
                              lulc_simplification_label, base_year):
    """Point each presence constraint at this run's own layer for the year it allocates from.

    The constraint rows carry a path to a binary raster saying where the protected class is.
    That path is written when the coefficients are produced, so it names the calibration's
    project and the year it was trained to, and a file used anywhere else then points at a
    directory that may not exist and a year that may not be the one being allocated from.

    Both matter. A layer from before the base year marks positions that have since moved; a
    layer from after it encodes land cover the run should not be able to see, which is how a
    hindcast scored against a 2020 mask appeared to gain skill it had not earned.

    Rebuilding the path here rather than editing the stored one also makes a coefficient set
    portable: the same file can be allocated from any base year, and from any project.
    """
    import os

    out = coefficients_df.copy()
    rows = out['type'] == 'multiplicative'
    suffixes = ('_presence_constraint', '_constraint')

    for i in out.index[rows]:
        name = str(out.at[i, 'spatial_regressor_name'])
        label = next((name[:-len(s)] for s in suffixes if name.endswith(s)), None)
        if label is None:
            continue
        out.at[i, 'data_location'] = os.path.join(
            fine_processed_inputs_dir, 'lulc', lulc_src_label, lulc_simplification_label,
            'binaries', str(base_year),
            'binary_%s_%s_%s_%s.tif' % (lulc_src_label, lulc_simplification_label,
                                        base_year, label))
    return out


def rebase_project_paths(coefficients_df, fine_processed_inputs_dir, base_year=None,
                         base_data_dir=None):
    """Re-root the per-project inputs a coefficient file points at.

    Most regressors are named by an absolute path written when the coefficients were fitted,
    so a file calibrated on one machine or in one project names directories that need not
    exist anywhere else. The layers themselves are not special: every project regenerates its
    own binaries and convolutions under fine_processed_inputs, so the same file is available
    locally under a different root.

    Only paths containing intermediate/fine_processed_inputs are moved. Everything else, the
    soil and climate covariates that live in base_data, is left alone, because those are
    shared rather than per-project.
    """
    import os

    marker = os.path.join('intermediate', 'fine_processed_inputs')
    out = coefficients_df.copy()

    import re

    def rebase(value):
        text = str(value)
        at = text.find(marker)
        if at < 0:
            return value
        tail = text[at + len(marker):].lstrip('/\\')
        if base_year is not None:
            # The layers are also year-stamped, in the directory and again in the filename,
            # with the year the coefficients were fitted to. A run allocating from a different
            # base year needs its own, and generates only that one.
            found = re.search(r'/(19|20)\d{2}/', '/' + tail)
            if found:
                stale = found.group(0).strip('/')
                tail = tail.replace(stale, str(base_year))
        return os.path.join(fine_processed_inputs_dir, tail)

    out['data_location'] = out['data_location'].map(rebase)

    # The shared covariates are shared in CONTENT, not in location: base_data sits at a
    # different root on every machine, so a path written on one names a directory that does
    # not exist on another. Re-root those too, keeping everything below base_data intact.
    if base_data_dir is not None:
        bd = 'base_data'

        def rebase_shared(value):
            text = str(value)
            at = text.find(bd)
            if at < 0 or marker in text:
                return value
            tail = text[at + len(bd):].lstrip('/\\')
            return os.path.join(base_data_dir, tail)

        out['data_location'] = out['data_location'].map(rebase_shared)

    return out


def coarse_years_available(source_dir):
    """The years the coarse model has a state for here, read off the directory names."""
    if not os.path.isdir(source_dir):
        return []
    return sorted(int(name) for name in os.listdir(source_dir)
                  if name.isdigit() and os.path.isdir(os.path.join(source_dir, name)))


def bracketing_coarse_years(source_dir, target_year):
    """The two available coarse years either side of target_year, or None if it needs no interpolation.

    Read from the data rather than named in a config, so a coarse model with different time steps
    needs no edit and a target year that IS a step is correctly left alone.

    Returns:
        tuple or None: (earlier, later), or None when target_year is already available or cannot be
        bracketed.
    """
    years = coarse_years_available(source_dir)
    target_year = int(target_year)
    if target_year in years:
        return None
    earlier = [y for y in years if y < target_year]
    later = [y for y in years if y > target_year]
    if not earlier or not later:
        return None
    return max(earlier), min(later)


def interpolate_coarse_state_at_year(source_dir, target_year, bracketing_years, filename_template,
                                     match_path=None, dst_dir=None, dst_filename_template=None):
    """Write a coarse state for a year the coarse model has no time step at, from the two it brackets.

    Coarse land-use models step every five years, so a fine base year taken from observed land cover
    (ESA 2022, relabelled 2023) has no matching coarse state. Differencing from the nearest step
    instead spans a longer interval than the run does -- 2020->2030 where 2023->2030 belongs -- and
    that matters because the coarse map enters allocation as an ANOMALY added to the regional total
    (covariate_sum_shift), so its amplitude is not renormalised away. A 10-year anomaly used for a
    7-year interval overstates the spatial contrast by 10/7.

    Linear between the brackets, which is the assumption the alternative makes anyway, over a five-
    year gap instead of a ten-year one.

    Args:
        source_dir (str): directory holding one subdirectory per coarse year.
        target_year (int): the year to write, e.g. 2023.
        bracketing_years (tuple): the two coarse years around it, e.g. (2020, 2025).
        filename_template (str): filename with {year} where the year appears.
        match_path (str): raster whose geotransform the output copies; defaults to the earlier bracket's.
        dst_dir (str or None): where to write, when the state belongs somewhere other than beside its
            own brackets. The first difference of every scenario starts from ONE common state in the
            baseline tree, so the reference trajectory's interpolation is written there rather than
            into the trajectory's own directory. None writes beside the brackets, as before.
        dst_filename_template (str or None): the name to write under, when the destination tree names
            its files differently from the source tree. Defaults to filename_template.

    Returns:
        list: the paths written.

    Raises:
        NameError: if target_year is not strictly between the brackets, or a bracket file is absent.
    """
    earlier, later = int(bracketing_years[0]), int(bracketing_years[1])
    target_year = int(target_year)
    if not earlier < target_year < later:
        raise NameError('%d is not strictly between the bracketing years %d and %d'
                        % (target_year, earlier, later))
    weight = (target_year - earlier) / (later - earlier)

    dst_dir = dst_dir or os.path.join(source_dir, str(target_year))
    dst_filename_template = dst_filename_template or filename_template
    hb.create_directories(dst_dir)
    earlier_path = os.path.join(source_dir, str(earlier), filename_template.format(year=earlier))
    later_path = os.path.join(source_dir, str(later), filename_template.format(year=later))
    for path in (earlier_path, later_path):
        if not hb.path_exists(path):
            raise NameError('cannot interpolate %d: %s is missing' % (target_year, path))
    dst_path = os.path.join(dst_dir, dst_filename_template.format(year=target_year))
    if hb.path_exists(dst_path):
        return []
    earlier_array = hb.as_array(earlier_path).astype('float64')
    later_array = hb.as_array(later_path).astype('float64')
    hb.save_array_as_geotiff(earlier_array + weight * (later_array - earlier_array),
                             dst_path, match_path or earlier_path)
    return [dst_path]


def reconcile_allocation(coarse_change_paths, input_lulc_path, output_lulc_path,
                         hectares_per_cell_path, class_indices, dst_path=None):
    """Per coarse cell and class, what the allocation was asked for against what it did.

    The allocator places EXPANDING classes on ranked cells and whatever occupies the chosen cell is
    displaced, so a contraction target is never executed directly. Nothing downstream records that,
    which is what this function supplies: the demanded change, the realised change, and the stock
    that was available to give. A contraction demand larger than the stock cannot be met by any
    allocation, and that is reported rather than silently absorbed by whichever class gives way.

    This is a diagnostic. It reads finished outputs and changes no allocation.

    Args:
        coarse_change_paths (dict): class label -> the demand raster the allocator was given, in
            hectares on the coarse grid.
        input_lulc_path (str): the fine LULC the allocation started from.
        output_lulc_path (str): the fine LULC it produced, on the same grid.
        hectares_per_cell_path (str): hectares per fine cell, on the same grid as the LULC.
        class_indices (dict): class label -> the integer the LULC uses for it.
        dst_path (str or None): where to write the CSV; None returns the frame without writing.

    Returns:
        pd.DataFrame: one row per coarse cell and class, with demanded_ha, realised_ha,
        available_ha and infeasible.

    Raises:
        ValueError: if the two LULC rasters or the hectares raster are not on one grid, if a demand
            raster carries a non-finite value it does not declare as nodata, if it marks a cell
            holding allocatable land as nodata, or if it does not cover the fine grid.
    """
    import numpy as np
    import pandas as pd
    import rasterio

    with rasterio.open(input_lulc_path) as a, rasterio.open(output_lulc_path) as b, \
            rasterio.open(hectares_per_cell_path) as h:
        if not (a.shape == b.shape == h.shape and a.transform == b.transform == h.transform):
            raise ValueError('input LULC, output LULC and hectares-per-cell must share one grid; got '
                             '%s, %s, %s' % (a.shape, b.shape, h.shape))
        before, after, ha, fine = a.read(1), b.read(1), h.read(1).astype(np.float64), a.transform

    rows = []
    for label, path in sorted(coarse_change_paths.items()):
        with rasterio.open(path) as c:
            demand, coarse, shape = c.read(1).astype(np.float64), c.transform, c.shape
            ndv = c.nodata
        # Nodata is only ever accepted OUTSIDE the allocation footprint. A cell holding allocatable
        # land whose demand is nodata has a MISSING demand, which is not the same as no change, and
        # reading it as zero would report that cell as satisfied. The footprint is not known until
        # the starting stock is summed, so the mask is carried down to the check below.
        if ndv is not None and np.isfinite(ndv):
            missing = demand == ndv
        elif ndv is not None:
            missing = ~np.isfinite(demand)
        else:
            missing = np.zeros(demand.shape, dtype=bool)
        demand = np.where(missing, 0.0, demand)
        nonfinite = int((~np.isfinite(demand)).sum())
        if nonfinite:
            raise ValueError('%s carries %d non-finite demand cell(s); treating them as zero would '
                             'report an unmet demand as satisfied' % (path, nonfinite))
        # Every fine cell's coarse home, from the two transforms rather than an assumed ratio.
        lat = fine.f + (np.arange(before.shape[0]) + 0.5) * fine.e
        lon = fine.c + (np.arange(before.shape[1]) + 0.5) * fine.a
        cr = ((lat - coarse.f) / coarse.e).astype(np.int64)
        cc = ((lon - coarse.c) / coarse.a).astype(np.int64)
        outside = int((cr < 0).sum() + (cr >= shape[0]).sum()
                      + (cc < 0).sum() + (cc >= shape[1]).sum())
        if outside:
            raise ValueError('%s does not cover the fine grid: %d fine row/column centre(s) fall '
                             'outside it. Clamping them into edge cells would attribute that land '
                             'to the wrong coarse cell' % (path, outside))
        flat = cr[:, None] * shape[1] + cc[None, :]
        idx = class_indices[label]
        n = shape[0] * shape[1]
        stock = np.bincount(flat.ravel(), weights=np.where(before == idx, ha, 0.0).ravel(), minlength=n)
        ended = np.bincount(flat.ravel(), weights=np.where(after == idx, ha, 0.0).ravel(), minlength=n)
        want = demand.ravel()
        inside = np.flatnonzero(missing.ravel() & ((stock > 0) | (ended > 0)))
        if inside.size:
            raise ValueError('%s marks %d cell(s) as nodata that hold allocatable land; a missing '
                             'demand is not no demand, and reading it as zero would report those '
                             'cells as satisfied. First: %s'
                             % (path, inside.size,
                                [(int(c // shape[1]), int(c % shape[1])) for c in inside[:5]]))
        # A cell is reported when anything was demanded, anything was there to take, or anything
        # ENDED there. The last is the case a demand-and-stock test misses: the output gaining a
        # class that was never demanded and never present is precisely what this should surface.
        for cell in np.flatnonzero((want != 0) | (stock > 0) | (ended != 0)):
            rows.append(dict(coarse_row=int(cell // shape[1]), coarse_col=int(cell % shape[1]),
                             class_label=label, demanded_ha=float(want[cell]),
                             realised_ha=float(ended[cell] - stock[cell]),
                             available_ha=float(stock[cell]),
                             infeasible=bool(want[cell] < 0 and -want[cell] > stock[cell])))
    out = pd.DataFrame(rows)
    if dst_path:
        out.to_csv(dst_path, index=False)
    return out

def coarse_state_dir(p, *parts):
    """Return the directory holding a coarse proportion state, interpolated states first.

    coarse_interpolated_states writes the states at fine base years that the coarse model has no
    time step for. A reader asking for such a year finds it there; every other year comes from
    coarse_simplified_proportion. Task trees that do not register coarse_interpolated_states read
    coarse_simplified_proportion only.

    Args:
        p (ProjectFlow): the project, with coarse_simplified_proportion_dir set.
        *parts (str): the scenario labels and year, as the proportion tree nests them.

    Returns:
        str: the directory to read the state from.
    """
    interpolated_dir = getattr(p, 'coarse_interpolated_states_dir', None)
    if interpolated_dir and os.path.isdir(os.path.join(interpolated_dir, *parts)):
        return os.path.join(interpolated_dir, *parts)
    return os.path.join(p.coarse_simplified_proportion_dir, *parts)

def coarse_base_state_labels(row, baseline_reference_row):
    """Return the labels of the scenario whose interpolated coarse state a row's first difference starts from.

    This applies when the fine base year is not a coarse time step, so the starting state has to be
    interpolated along some modelled trajectory. A baseline row is observed data with no trajectory
    to interpolate along, so a scenario that references one starts from its own interpolated state.
    A scenario that references another modelled row starts from that row's state instead, which is
    how a project gives several scenarios one common starting state: point their
    baseline_reference_label at the same modelled row.

    Args:
        row (pd.Series): the scenario row being differenced.
        baseline_reference_row (pd.Series): the row its baseline_reference_label names.

    Returns:
        list: exogenous, climate, model and counterfactual labels, as the scenario tree nests them.
    """
    source_row = row if baseline_reference_row['scenario_type'] == 'baseline' else baseline_reference_row
    return [str(source_row[column]) for column in ('exogenous_label', 'climate_label', 'model_label', 'counterfactual_label')]


def coarse_ha_per_class_from_fine_lulc(lulc_path, class_indices, ha_per_cell_coarse_array, coarse_rows_per_chunk=8):
    """Hectares of each class in every coarse cell, aggregated from a fine LULC raster.

    The fine raster must cover exactly the extent of ha_per_cell_coarse_array, with a whole
    number of fine cells per coarse cell. The area of a class in a coarse cell is the fraction
    of its fine cells in that class times the coarse cell's hectares. The raster is read a few
    coarse rows at a time, so a global 10-arcsecond map never has to fit in memory.

    Args:
        lulc_path (str): Fine LULC raster, already in the simplified class scheme.
        class_indices (list): Class ids to tabulate.
        ha_per_cell_coarse_array (np.ndarray): Hectares per coarse cell, shape (coarse_rows, coarse_cols).
        coarse_rows_per_chunk (int): Coarse rows read per pass.

    Returns:
        dict: class id -> float64 array of hectares, shaped like ha_per_cell_coarse_array.
    """
    coarse_n_rows, coarse_n_cols = ha_per_cell_coarse_array.shape
    fine_n_cols, fine_n_rows = hb.get_raster_info(lulc_path)['raster_size']
    upscale = fine_n_cols // coarse_n_cols
    if fine_n_cols != coarse_n_cols * upscale or fine_n_rows != coarse_n_rows * upscale:
        raise ValueError('Fine raster ' + str(lulc_path) + ' of shape ' + str((fine_n_rows, fine_n_cols)) + ' is not a whole multiple of the coarse grid ' + str((coarse_n_rows, coarse_n_cols)) + '.')

    ha_per_class = {i: np.zeros((coarse_n_rows, coarse_n_cols), dtype=np.float64) for i in class_indices}
    for coarse_row in range(0, coarse_n_rows, coarse_rows_per_chunk):
        n_rows = min(coarse_rows_per_chunk, coarse_n_rows - coarse_row)
        fine_chunk = hb.load_geotiff_chunk_by_cr_size(lulc_path, [0, coarse_row * upscale, fine_n_cols, n_rows * upscale])
        fine_chunk = fine_chunk.reshape(n_rows, upscale, coarse_n_cols, upscale)
        ha_chunk = ha_per_cell_coarse_array[coarse_row: coarse_row + n_rows]
        for i in class_indices:
            ha_per_class[i][coarse_row: coarse_row + n_rows] = (fine_chunk == i).sum(axis=(1, 3)) / float(upscale * upscale) * ha_chunk
    return ha_per_class


def check_fine_coarse_alignment(fine_path, coarse_path):
    """Return how many fine pixels span one coarse cell, after checking that the two grids nest exactly.

    Args:
        fine_path (str): raster on the fine grid.
        coarse_path (str): raster on the coarse grid.

    Returns:
        int: the number of fine pixels along one side of a coarse cell.

    Raises:
        NameError: if the CRSs differ, a grid is rotated, the cell-size ratio is not a whole number, or the two
            rasters do not cover the same extent (which includes an origin offset by part of a fine pixel).
    """
    from osgeo import osr
    fine = hb.get_raster_info_hb(fine_path)
    coarse = hb.get_raster_info_hb(coarse_path)
    fine_srs = osr.SpatialReference(wkt=fine['projection'])
    coarse_srs = osr.SpatialReference(wkt=coarse['projection'])
    if not fine_srs.IsSame(coarse_srs):
        raise NameError('Fine and coarse rasters have different CRSs: ' + fine_path + ', ' + coarse_path)
    fg, cg = fine['geotransform'], coarse['geotransform']
    if fg[2] != 0 or fg[4] != 0 or cg[2] != 0 or cg[4] != 0:
        raise NameError('Rotated grids are not supported: ' + fine_path + ', ' + coarse_path)
    ratio_x, ratio_y = cg[1] / fg[1], cg[5] / fg[5]
    ratio = int(round(ratio_x))
    if ratio < 1 or abs(ratio_x - ratio) > 1e-6 or abs(ratio_y - ratio) > 1e-6:
        raise NameError('The coarse cell is not a whole number of fine pixels (' + str(ratio_x) + ', ' + str(ratio_y) + '): ' + fine_path + ', ' + coarse_path)
    tolerance = 1e-6 * abs(fg[1])
    fine_rows, fine_cols = fine['shape']
    coarse_rows, coarse_cols = coarse['shape']
    if abs(fg[0] - cg[0]) > tolerance or abs(fg[3] - cg[3]) > tolerance or (fine_rows, fine_cols) != (coarse_rows * ratio, coarse_cols * ratio):
        raise NameError('Fine and coarse rasters do not cover the same extent: ' + fine_path + ', ' + coarse_path)
    return ratio

def describe_file_identity(path):
    """Return the size, modification time and sha256 of a file, recorded once when an output is made from it."""
    stat = os.stat(path)
    return {'path': path, 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': hb.hash_file_path(path)}

def confirm_file_identity(recorded):
    """Check a file against its recorded identity, hashing it again only when its size or time has changed.

    Args:
        recorded (dict): the output of describe_file_identity.

    Returns:
        dict or None: the record, refreshed if only the size or time changed and the bytes are the same; None if the
        bytes differ.
    """
    stat = os.stat(recorded['path'])
    if stat.st_size == recorded['size'] and stat.st_mtime_ns == recorded['mtime_ns']:
        return recorded
    current = describe_file_identity(recorded['path'])
    if current['sha256'] != recorded['sha256']:
        return None
    return current

def reuse_outputs_if_inputs_unchanged(task_name, provenance_path, output_paths, sources, settings):
    """Decide whether a task may reuse its outputs, by checking the inputs and settings recorded when they were made.

    Returns False when an output or the record is missing, so the task makes everything again. Returns True when all
    outputs exist and every recorded input still has the same bytes, after a size and time check that hashes a file
    again only if that check changes; a record refreshed that way is saved. Logs the reason and returns False when the
    settings, the set of inputs or any input's bytes differ, because the outputs then no longer describe the inputs.

    Args:
        task_name (str): named in the log message.
        provenance_path (str): the JSON record written by save_provenance.
        output_paths (list): every output the task makes.
        sources (dict): name -> path of every input file.
        settings (dict): every setting the outputs depend on, as JSON values.

    Returns:
        bool: True if the outputs can be reused.
    """
    import json
    if not (hb.path_exists(provenance_path) and all(hb.path_exists(i) for i in output_paths)):
        return False
    with open(provenance_path) as f:
        recorded = json.load(f)
    if recorded['settings'] != json.loads(json.dumps(settings)) or set(recorded['sources']) != set(sources) or any(
            recorded['sources'][key]['path'] != path for key, path in sources.items()):
        hb.log(task_name + ': recomputing because its inputs or settings differ from those recorded in ' + provenance_path)
        return False
    confirmed = {key: confirm_file_identity(record) for key, record in recorded['sources'].items()}
    changed = [record['path'] for key, record in recorded['sources'].items() if confirmed[key] is None]
    if changed:
        hb.log(task_name + ': recomputing because these inputs changed since the outputs were made: ' + ', '.join(changed))
        return False
    if confirmed != recorded['sources']:
        with open(provenance_path, 'w') as f:
            json.dump({'sources': confirmed, 'settings': recorded['settings']}, f, indent=2)
    return True

def save_provenance(provenance_path, sources, settings):
    """Record the identity of every input and the settings, after a task has written its outputs."""
    import json
    with open(provenance_path, 'w') as f:
        json.dump({'sources': {key: describe_file_identity(path) for key, path in sources.items()}, 'settings': settings}, f, indent=2)

def prepare_allocation_coefficients(p, df, calibration_parameters_path):
    """Make a read coefficient table the one this run allocates with: check its class scheme, rebuild the presence
    constraints for this scenario and point every layer at this run's own files.

    allocation_zones and the pre-allocation stock check both call this, so the two read the same constraints.

    Args:
        p (ProjectFlow): the run; reads changing_class_labels, all_class_labels, the protected-class setting,
            fine_processed_inputs_dir, key_base_year, base_data_dir and the lulc labels.
        df (pandas.DataFrame): the coefficient table as read.
        calibration_parameters_path (str): where df was read from, named in errors.

    Returns:
        pandas.DataFrame: the table to allocate with.
    """
    # Fail here rather than allocating with coefficients fitted for other classes.
    # Class ids shift between schemes, so the wrong file misassigns classes silently
    # and still produces a map.
    scheme_labels = getattr(p, 'changing_class_labels', None)
    if scheme_labels:
        check_coefficients_match_class_scheme(
            df, scheme_labels, coefficients_path=calibration_parameters_path)
    else:
        hb.log('Skipping the coefficient scheme check: changing_class_labels is not set.')

    # Rebuild the constraint block for THIS run, discarding what the file carried. We
    # train unconstrained and allocate constrained, so the block belongs to the scenario
    # rather than to the calibration that produced the file. water/other/othernat need no
    # declaration -- they are derived, having no coarse budget in this configuration.
    if scheme_labels:
        additional_protected = resolve_additional_protected_class_labels(p)
        df = apply_presence_constraints(
            df, p.all_class_labels, scheme_labels, additional_protected)
        hb.log('Rebuilt presence constraints; protected: ' + ', '.join(
            protected_class_labels(
                p.all_class_labels, scheme_labels, additional_protected)))

    # Point the presence constraints at this run's own layer for the year it allocates
    # from, rather than trusting the path baked in when the coefficients were produced.
    # That path names the calibration's project and the year it was trained to, so a
    # file used anywhere else points at a directory that need not exist and a year that
    # need not be the base year. A layer from after the base year also encodes land
    # cover the run should not see.
    # Re-root the per-project regressor layers before anything reads them. A
    # coefficient file fitted elsewhere names that project's directories, which need
    # not exist here; every project generates its own copies under the same relative
    # path. Shared covariates in base_data are untouched.
    df = rebase_project_paths(df, p.fine_processed_inputs_dir, p.key_base_year,
                              p.base_data_dir)

    df = resolve_constraint_layers(
        df, p.fine_processed_inputs_dir, p.lulc_src_label,
        p.lulc_simplification_label, p.key_base_year)

    # TODO This is bad. Fix it.
    # TODOOO, YES IT WAS A BAD IDEA YOU DUMMY.
    # TODOOO AGAIN. Indeed, still bad.
    year_replacement_dict = {}
    year_replacement_dict['2014'] = p.key_base_year
    for src, dst in year_replacement_dict.items():

        df['data_location'] = df.loc[:, 'data_location'].replace({str(src): str(dst)}, regex=True)
    return df

def projected_coarse_change_paths(p):
    """The coarse net-change raster of each changing class that allocation reads for p.year, in class order."""
    filename_end = ('_' + str(p.year) + '_' + str(p.previous_year) + '_ha_diff_' + p.exogenous_label + '_' + p.climate_label + '_'
                    + p.model_label + '_' + p.counterfactual_label + '.tif')
    return [os.path.join(p.projected_coarse_change_dir, label + filename_end) for label in p.changing_class_labels]

def resolve_calibration_parameters_path(p):
    """Find the coefficient table the way allocation_zones does, but raise instead of logging when it is missing.

    Raises:
        NameError: if calibration_parameters_source is 'calibration_task' (one table per zone, which a regional check
            cannot read before the zones exist) or names no file.
    """
    source = p.calibration_parameters_source
    if source == 'calibration_task':
        raise NameError('The stock check needs one coefficient table for the whole run; calibration_task gives one per zone.')
    path = p.get_path(str(source), leave_ref_path_if_fail=True) if hb.looks_like_path(source) else source
    for candidate in (path, os.path.join(p.input_dir, str(source)), os.path.join(p.base_data_dir, str(source))):
        if hb.path_exists(candidate):
            return candidate
    raise NameError('Coefficient table not found for calibration_parameters_source ' + str(source))

def feasibility_constraint_layers(coefficients_df, changing_class_labels):
    """The multiplicative layers that decide where each changing class may be placed.

    Args:
        coefficients_df (pandas.DataFrame): a table from prepare_allocation_coefficients.
        changing_class_labels (list): the changing classes, in allocation order.

    Returns:
        tuple: layer names (list), layer paths (list) and coefficients (numpy array, class by layer).

    Raises:
        NameError: if a multiplicative layer has different paths or coefficients in different calibration blocks,
            because a regional check cannot then know which applies to a pixel.
    """
    column_for = dict(zip(coefficient_class_labels(coefficients_df), coefficient_class_columns(coefficients_df)))
    columns = ['spatial_regressor_name', 'data_location'] + [column_for[label] for label in changing_class_labels]
    rows = coefficients_df.loc[coefficients_df['type'] == 'multiplicative', columns].drop_duplicates()
    varying = sorted(set(rows.loc[rows['spatial_regressor_name'].duplicated(), 'spatial_regressor_name']))
    if varying:
        raise NameError('Multiplicative layers differ between calibration blocks, so the stock check cannot apply them: '
                        + ', '.join(varying))
    coefficients = rows[[column_for[label] for label in changing_class_labels]].to_numpy(dtype=np.float64).T
    return list(rows['spatial_regressor_name']), list(rows['data_location']), coefficients

def placement_allowed(layers_3d, coefficients):
    """Where each class may be placed under the multiplicative layers, as the allocator computes it.

    The allocator multiplies a class's suitability by 1 - (1 - coefficient) * layer for every multiplicative layer and
    never places the class where the product is 0.

    Args:
        layers_3d (numpy.ndarray): layer by row by column.
        coefficients (numpy.ndarray): class by layer.

    Returns:
        numpy.ndarray: boolean, class by row by column.
    """
    allowed = np.ones((coefficients.shape[0],) + layers_3d.shape[1:], dtype=np.float64)
    for k in range(coefficients.shape[0]):
        for j in range(layers_3d.shape[0]):
            allowed[k] *= 1.0 - (1.0 - coefficients[k, j]) * layers_3d[j]
    return allowed != 0

def regional_stock_capacity(lulc_array, valid_array, locked_array, ha_array, region_array, class_ids, allowed_3d,
                            gaining_by_region, n_regions):
    """Hectares, per region, of each class's stock and of the land each gaining class may take.

    A pixel counts in exactly one class, so the class stocks never share hectares. Gains do share land, so their
    joint capacity is counted once per pixel in any_receivable.

    Args:
        lulc_array (numpy.ndarray): the map the year allocates from.
        valid_array (numpy.ndarray): bool, pixels the allocator may consider at all (base-year land cover not nodata).
        locked_array (numpy.ndarray): bool, pixels already changed in an earlier year, which may not change again.
        ha_array (numpy.ndarray): hectares of each pixel.
        region_array (numpy.ndarray): region id of each pixel, 0 outside every region.
        class_ids (list): changing class ids, in allocation order.
        allowed_3d (numpy.ndarray): bool, class by row by column, from placement_allowed.
        gaining_by_region (numpy.ndarray): bool, class by region id, True where the region asks the class to gain.
        n_regions (int): the largest region id.

    Returns:
        dict: 'physical', 'eligible', 'losable' and 'receivable' (class by region id) and 'any_receivable' (region id).
            physical is the valid stock, eligible leaves out locked pixels, losable keeps only eligible pixels some
            gaining class of the region may take, receivable is the land each class may gain on.
    """
    region = region_array.astype(np.int64)
    size = n_regions + 1

    def total(mask):
        return np.bincount(region.ravel(), weights=np.where(mask & (region > 0), ha_array, 0.0).ravel(), minlength=size)[:size]

    open_land = valid_array & ~locked_array
    receivable = [open_land & allowed_3d[k] & (lulc_array != class_ids[k]) & gaining_by_region[k][region] for k in range(len(class_ids))]
    any_receivable = np.logical_or.reduce(receivable)
    out = {'physical': [], 'eligible': [], 'losable': [], 'receivable': []}
    for k, class_id in enumerate(class_ids):
        is_class = lulc_array == class_id
        out['physical'].append(total(is_class & valid_array))
        out['eligible'].append(total(is_class & open_land))
        out['losable'].append(total(is_class & any_receivable))
        out['receivable'].append(total(receivable[k]))
    out = {key: np.array(value) for key, value in out.items()}
    out['any_receivable'] = total(any_receivable)
    return out

def evaluate_regional_feasibility(demand, capacity, class_labels, tolerance_ha=1e-6):
    """Compare each region's net change with the land it can supply, before anything is allocated.

    Necessary, not sufficient: a region that passes can still miss its targets, because the allocator places gains
    within each coarse cell, rounds to whole pixels and takes losses by rank without reading negative targets.

    Args:
        demand (numpy.ndarray): net change in hectares, class by region id.
        capacity (dict): from regional_stock_capacity.
        class_labels (list): changing class labels, in allocation order.
        tolerance_ha (float): floating-point slack only, in hectares.

    Returns:
        pandas.DataFrame: one row per region and class with a non-zero change, plus one 'all_gains' row per region with
            any gain. A loss is compared with the losable stock, a gain with the land the class may take, and the
            region's gains together with the land any of them may take. shortfall_ha > 0 marks a failure.
    """
    rows = []
    for region_id in np.nonzero(np.any(demand != 0, axis=0))[0]:
        for k, label in enumerate(class_labels):
            change = float(demand[k, region_id])
            if change == 0:
                continue
            check, available = ('loss', capacity['losable'][k, region_id]) if change < 0 else ('gain', capacity['receivable'][k, region_id])
            rows.append({'region_id': int(region_id), 'class_label': label, 'check': check, 'change_ha': change,
                         'requested_ha': abs(change), 'available_ha': float(available),
                         'physical_stock_ha': float(capacity['physical'][k, region_id]),
                         'eligible_stock_ha': float(capacity['eligible'][k, region_id])})
        gains = float(np.maximum(demand[:, region_id], 0).sum())
        if gains > 0:
            rows.append({'region_id': int(region_id), 'class_label': 'all_gains', 'check': 'all_gains', 'change_ha': gains,
                         'requested_ha': gains, 'available_ha': float(capacity['any_receivable'][region_id]),
                         'physical_stock_ha': np.nan, 'eligible_stock_ha': np.nan})
    table = pd.DataFrame(rows, columns=['region_id', 'class_label', 'check', 'change_ha', 'requested_ha', 'available_ha',
                                        'physical_stock_ha', 'eligible_stock_ha'])
    table['shortfall_ha'] = np.where(table['requested_ha'] - table['available_ha'] > tolerance_ha,
                                     table['requested_ha'] - table['available_ha'], 0.0)
    return table

def read_aligned_window(path, grid_geotransform, col, row, n_cols, n_rows):
    """Read the block of a raster that covers columns and rows of another grid with the same cell size.

    Raises:
        NameError: if the cell sizes differ, the grids are offset by part of a cell, or the raster does not cover the block.
    """
    info = hb.get_raster_info_hb(path)
    g = info['geotransform']
    if abs(g[1] - grid_geotransform[1]) > 1e-9 * abs(g[1]) or abs(g[5] - grid_geotransform[5]) > 1e-9 * abs(g[5]):
        raise NameError('Cell size of ' + path + ' differs from the allocation grid.')
    col_offset = (grid_geotransform[0] + col * grid_geotransform[1] - g[0]) / g[1]
    row_offset = (grid_geotransform[3] + row * grid_geotransform[5] - g[3]) / g[5]
    c, r = int(round(col_offset)), int(round(row_offset))
    if abs(col_offset - c) > 1e-6 or abs(row_offset - r) > 1e-6:
        raise NameError(path + ' is offset from the allocation grid by part of a cell.')
    width, height = info['raster_size']
    if c < 0 or r < 0 or c + n_cols > width or r + n_rows > height:
        raise NameError(path + ' does not cover the allocation grid.')
    return hb.load_geotiff_chunk_by_cr_size(path, [c, r, n_cols, n_rows])

def earlier_allocation_tiles(scenario_dir, tile_name, map_years, mask_year):
    """The allocation tiles that make up an accepted map, as allocation finds them.

    Args:
        scenario_dir (str): allocations/<exogenous>/<climate>/<model>/<counterfactual>.
        tile_name (str): the tile's lulc file name with '{}' for the year.
        map_years (list): years whose lulc tiles to lay over the base map, oldest first, so a tile that did not change
            in a later year keeps its last map.
        mask_year (int): year whose cumulative_change_happened tiles mark locked pixels, or None for none.

    Returns:
        list: (kind, path) pairs, kind 'lulc' or 'locked', in the order they are to be applied.
    """
    def found(year, file_name):
        return sorted(glob.glob(os.path.join(scenario_dir, str(year), 'allocation_zones', '*', 'allocation', file_name)))
    tiles = [('lulc', path) for year in map_years for path in found(year, tile_name.format(year))]
    if mask_year is not None:
        tiles += [('locked', path) for path in found(mask_year, 'cumulative_change_happened.tif')]
    return tiles

def allocation_tile_windows(tiles, grid_geotransform):
    """Where each tile sits on the allocation grid: (kind, path, row, col, n_rows, n_cols)."""
    windows = []
    for kind, path in tiles:
        g = hb.get_raster_info_hb(path)
        col = int(round((g['geotransform'][0] - grid_geotransform[0]) / grid_geotransform[1]))
        row = int(round((g['geotransform'][3] - grid_geotransform[3]) / grid_geotransform[5]))
        windows.append((kind, path, row, col) + tuple(g['raster_size'][::-1]))
    return windows

def paste_allocation_tiles(lulc, locked, row, windows):
    """Write the tiles that overlap a stripe of rows into it, in order: lulc tiles into lulc, locked tiles into locked.

    Each grid pixel is written, not added, so overlapping tiles cannot count a pixel twice.

    Args:
        lulc (numpy.ndarray): the stripe of the map, changed in place.
        locked (numpy.ndarray): bool, the stripe of locked pixels, changed in place; may be None if no tile is 'locked'.
        row (int): grid row of the stripe's first row.
        windows (list): from allocation_tile_windows.
    """
    height, n_cols = lulc.shape
    for kind, path, tile_row, tile_col, tile_rows, tile_cols in windows:
        r0, r1 = max(row, tile_row), min(row + height, tile_row + tile_rows)
        if r0 >= r1:
            continue
        c0, c1 = max(0, tile_col), min(n_cols, tile_col + tile_cols)
        block = hb.as_array(path)[r0 - tile_row:r1 - tile_row, c0 - tile_col:c1 - tile_col]
        if kind == 'lulc':
            lulc[r0 - row:r1 - row, c0:c1] = block
        else:
            locked[r0 - row:r1 - row, c0:c1] = block == 1

def regional_transition_hectares(before, after, ha, region, valid, class_ids, n_regions):
    """Hectares that move from each class to each class, per region, summed from fine pixel areas.

    The diagonal is the stock that did not change. Pixels outside every region (id 0) and pixels that are not valid
    are left out.

    Args:
        before (numpy.ndarray): the accepted map the year starts from.
        after (numpy.ndarray): the map the year produced, same shape.
        ha (numpy.ndarray): hectares of each pixel.
        region (numpy.ndarray): region id of each pixel, 0 outside every region.
        valid (numpy.ndarray): bool, pixels to count (data in both maps).
        class_ids (list): every class id the maps may hold, in report order.
        n_regions (int): the largest region id.

    Returns:
        numpy.ndarray: region id by source class by destination class, in hectares.

    Raises:
        NameError: if a counted pixel holds a class id not in class_ids.
    """
    counted = valid & (region > 0)
    for name, array in (('before', before), ('after', after)):
        unknown = sorted(set(np.unique(array[counted]).tolist()) - set(int(i) for i in class_ids))
        if unknown:
            raise NameError('The %s map holds class ids with no label: %s' % (name, unknown))
    n = len(class_ids)
    lookup = np.zeros(int(max(max(class_ids), before[counted].max(initial=0), after[counted].max(initial=0))) + 1, dtype=np.int64)
    lookup[np.asarray(class_ids, dtype=np.int64)] = np.arange(n)
    index = (region[counted].astype(np.int64) * n + lookup[before[counted]]) * n + lookup[after[counted]]
    return np.bincount(index, weights=ha[counted], minlength=(n_regions + 1) * n * n).reshape(n_regions + 1, n, n)

def read_allowed_transitions(path, class_labels):
    """Read explicit donor rules: a CSV of from_class_label, to_class_label, allowed (1 or 0).

    Every pair of different classes must be listed, so no rule is assumed.

    Returns:
        numpy.ndarray: bool, source class by destination class; the diagonal is True.

    Raises:
        NameError: if a pair is missing or listed twice, or a label is unknown.
    """
    df = pd.read_csv(path, encoding='utf-8-sig')
    unknown = sorted((set(df['from_class_label']) | set(df['to_class_label'])) - set(class_labels))
    if unknown:
        raise NameError('Donor rules name unknown classes: %s' % unknown)
    if df.duplicated(['from_class_label', 'to_class_label']).any():
        raise NameError('Donor rules list a pair twice: ' + path)
    position = {label: k for k, label in enumerate(class_labels)}
    allowed = np.eye(len(class_labels), dtype=bool)
    listed = np.eye(len(class_labels), dtype=bool)
    for r in df.itertuples():
        allowed[position[r.from_class_label], position[r.to_class_label]] = bool(int(r.allowed))
        listed[position[r.from_class_label], position[r.to_class_label]] = True
    missing = [(class_labels[i], class_labels[j]) for i, j in zip(*np.nonzero(~listed))]
    if missing:
        raise NameError('Donor rules do not cover every pair of classes; missing %s' % missing[:10])
    return allowed

def summarize_regional_transitions(matrix, stock_before, stock_after, intended, class_labels, allowed=None, tolerance_ha=1e-6,
                                   relative_tolerance=1e-9):
    """Physical accounting of one year's transitions per region and class, checked against the maps' own stocks.

    Args:
        matrix (numpy.ndarray): from regional_transition_hectares.
        stock_before (numpy.ndarray): region id by class, hectares on the starting map, counted independently.
        stock_after (numpy.ndarray): region id by class, hectares on the produced map.
        intended (dict): class label -> array by region id of the intended net change (the GTAP totals as fed to the
            allocator); classes without a target are left out.
        class_labels (list): labels in the matrix order.
        allowed (numpy.ndarray): bool donor rules, source by destination, or None when none were supplied.
        tolerance_ha (float): floating-point slack only.
        relative_tolerance (float): floating-point slack as a share of the stock, for sums over large regions.

    Returns:
        tuple: (transitions, summary) DataFrames. transitions has one row per region, source and destination with
            hectares > 0, diagonal included. summary has one row per region and class.

    Raises:
        NameError: if a row sum differs from the starting stock or a column sum from the produced stock, which would
            mean the matrix lost or invented land.
    """
    rows_sum, columns_sum = matrix.sum(axis=2), matrix.sum(axis=1)
    if not (np.allclose(rows_sum, stock_before, rtol=relative_tolerance, atol=tolerance_ha)
            and np.allclose(columns_sum, stock_after, rtol=relative_tolerance, atol=tolerance_ha)):
        raise NameError('Transition matrix does not reproduce the map stocks: max row error %.6g ha, max column error %.6g ha'
                        % (np.abs(rows_sum - stock_before).max(initial=0), np.abs(columns_sum - stock_after).max(initial=0)))
    diagonal = np.einsum('rkk->rk', matrix)
    regions = [int(r) for r in np.nonzero(stock_before.sum(axis=1) + stock_after.sum(axis=1))[0] if r > 0]
    transitions, summary = [], []
    for r in regions:
        for i, j in zip(*np.nonzero(matrix[r] > 0)):
            transitions.append({'region_id': r, 'from_class_label': class_labels[i], 'to_class_label': class_labels[j],
                                'ha': float(matrix[r, i, j]), 'unchanged': bool(i == j),
                                'allowed': (bool(allowed[i, j]) if allowed is not None else np.nan)})
        for k, label in enumerate(class_labels):
            gain, loss = float(columns_sum[r, k] - diagonal[r, k]), float(rows_sum[r, k] - diagonal[r, k])
            target = float(intended[label][r]) if label in intended else np.nan
            violation = float(matrix[r, :, k][~allowed[:, k]].sum()) if allowed is not None else np.nan
            summary.append({'region_id': r, 'class_label': label, 'before_ha': float(stock_before[r, k]), 'after_ha': float(stock_after[r, k]),
                            'unchanged_ha': float(diagonal[r, k]), 'gross_gain_ha': gain, 'gross_loss_ha': loss, 'realized_net_ha': gain - loss,
                            'intended_net_ha': target, 'realized_minus_intended_ha': gain - loss - target,
                            'disallowed_gain_ha': violation})
    return pd.DataFrame(transitions), pd.DataFrame(summary)
