import logging, os, math, sys
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
    # TODO CRITICAL, eliminate the duplication of this code across seals, gtappy, gtap_invest etc.
    # srtip()
    # Rules:
    # First check if is numeric
    # Then check if has extension, is path
    for attribute_name, attribute_value in list(zip(input_row.index, input_row.values)):
        if attribute_name == 'coarse_correspondence_path':
            pass
        
        
        try:
            float(attribute_value)
            is_floatable = True
        except:
            is_floatable = False
        try:
            int(attribute_value)
            is_intable = True
        except:
            is_intable = False

        if attribute_name == 'regional_projections_input_path':
            pass
        
        
        
        # Check if attribute name or attribute_value have cat_ears.
        has_cat_ears = hb.has_cat_ears(attribute_name) or hb.has_cat_ears(attribute_value)
        if has_cat_ears:
            replace_dict = hb.replace_cat_ears_with_object_attributes
            attribute_value = hb.replace_cat_ears_with_object_attributes(attribute_value, input_object)
            

            
        

            # build the replace dict
        #     # ce_list = hb.parse_to_ce_list(attribute_value) # BORKED
        #     # while 1:
                
        #     # TODO HACK Only parses the first catear entry
        #     first_ce_entry = attribute_value.split('<^', 1)[1].split('^>', 1)[0]
        #     replace_dict = {}
        #     if getattr(input_object, first_ce_entry, None) is not None:
        #         replace_dict[i] = getattr(input_object, i)
        #     # replace the cat_ears
        #     attribute_value = hb.replace_in_string_via_dict(attribute_value, replace_dict)
            
            
        #     hb.replace_cat_ears_with_variables(p.regional_projections_input_path, year)
        
        
        # CRITICAL TODO, make this consistent across multiple projects that implement it.
        # NOTE Clever use of p.get_path() here.
        if '.' in str(attribute_value) and not is_floatable: # Might be a path        
            if has_cat_ears:
                path = input_object.get_path(attribute_value, leave_ref_path_if_fail=True)
            else:    
                path = input_object.get_path(attribute_value)
            setattr(input_object, attribute_name, path)        
        
        
        
        
        
        # # NOTE Clever use of p.get_path() here.
        # if '.' in str(attribute_value) and not is_floatable: # Might be a path
        #     path = input_object.get_path(attribute_value)
        #     setattr(input_object, attribute_name, path)
            
            
            
            
            
            
            

        elif 'year' in attribute_name:
            if ' ' in str(attribute_value):
                new_attribute_value = []
                for i in attribute_value.split(' '):
                    try:
                        new_attribute_value.append(int(i))
                    except:
                        new_attribute_value.append(str(i))
                attribute_value = new_attribute_value

                # attribute_value = [int(i) if 'nan' not in str(i) and intable else None for i in attribute_value.split(' ')]
            elif is_intable:
                if attribute_name == 'key_base_year':
                    attribute_value = int(attribute_value)
                else:
                    attribute_value = [int(attribute_value)]
            elif 'lulc' in attribute_name: #
                attribute_value = str(attribute_value)
            else:
                if 'nan' not in str(attribute_value):
                    try:
                        attribute_value = [int(attribute_value)]
                    except:
                        attribute_value = [str(attribute_value)]
                else:
                    attribute_value = None
            setattr(input_object, attribute_name, attribute_value)

        elif 'dimensions' in attribute_name:
            if ' ' in str(attribute_value):
                attribute_value = [str(i) if 'nan' not in str(i) else None for i in attribute_value.split(' ')]
            else:
                if 'nan' not in str(attribute_value):
                    attribute_value = [str(attribute_value)]
                else:
                    attribute_value = None

            setattr(input_object, attribute_name, attribute_value)
        else:
            if str(attribute_value).lower() == 'nan':
                attribute_value = None
                setattr(input_object, attribute_name, attribute_value)
            else:
                # Check if the t string
                setattr(input_object, attribute_name, attribute_value)

    model_spec = {}
    model_spec['regional_projections_input_path'] = ''
    assign_defaults_from_model_spec(input_object, model_spec)

def set_derived_attributes(p):

    # Resolutions come from the fine and coarse maps
    p.fine_resolution = hb.get_cell_size_from_path(p.base_year_lulc_path)
    p.fine_resolution_arcseconds = hb.pyramid_compatible_resolution_to_arcseconds[p.fine_resolution]
    
    if hb.path_exists(p.coarse_projections_input_path):
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
    for year_c, year in enumerate(years):
        allocate_per_zone_dict = {}
        for column in columns_to_process:
            output_path = os.path.join(output_dir, column + output_filename_end)
            
            if not hb.path_exists(output_path):
                hb.log('Processing ' + column + ' for ' + scenario_label + ',  writing to ' + output_path)
                regions_column_id = regions_column_id.replace('label', 'id')
                

                for i, change in merged[column].items():
                    zone_id = int(merged[regions_column_id][i])
                    
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


