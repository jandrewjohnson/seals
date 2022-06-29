import logging, os, math
from osgeo import gdal
import numpy as np
import scipy
import scipy.stats as st
import scipy.ndimage
import hazelbean as hb
# from hazelbean.ui import model, inputs
from collections import OrderedDict
from matplotlib import pyplot as plt
# import geoecon as ge
import matplotlib
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


logging.basicConfig(level=logging.WARNING)
# hb.ui.model.LOGGER.setLevel(logging.WARNING)
# hb.ui.inputs.LOGGER.setLevel(logging.WARNING)

L = hb.get_logger('seals_utils')
L.setLevel(logging.INFO)

logging.getLogger('Fiona').setLevel(logging.WARNING)
logging.getLogger('fiona.collection').setLevel(logging.WARNING)

np.seterr(divide='ignore', invalid='ignore')

dev_mode = True


def show_lulc_class_change_difference(baseline_array, observed_array, projected_array, lulc_class, similarity_array, change_array, annotation_text, output_path, **kwargs):
    fig, axes = plt.subplots(2, 1)

    classes = np.zeros(observed_array.shape)
    # TOGGLING the next line determines if the baseline nonchanging show up, which can be overload
    classes = np.where((baseline_array == lulc_class), 1, classes)
    classes = np.where((observed_array == lulc_class) & (projected_array != lulc_class) & (baseline_array != lulc_class), 2, classes)
    classes = np.where((projected_array == lulc_class) & (observed_array != lulc_class) & (baseline_array != lulc_class), 3, classes)
    classes = np.where((projected_array == lulc_class) & (observed_array == lulc_class) & (baseline_array != lulc_class), 4, classes)

    axes[0].annotate(annotation_text,
                     xy=(.05, .75), xycoords='figure fraction',
                     horizontalalignment='left', verticalalignment='top',
                     fontsize=6)

    for ax in axes:
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    # import geoecon as ge
    cmap = hb.generate_custom_colorbar(classes, vmin=0, vmax=4, color_scheme='bold_spectral_white_left', transparent_at_cbar_step=0)
    im_top_0 = axes[0].imshow(classes, cmap=cmap, vmin=0, vmax=4)

    bounds = np.linspace(1, 3, 3)
    bounds = [.5, 1.5, 2.5, 3.5, 4.5]

    # ticks = np.linspace(1, 2, 2)
    ticks = [1, 2, 3, 4]
    cbar0 = plt.colorbar(im_top_0, ax=axes[0], orientation='vertical', aspect=20, shrink=1.0, ticks=ticks, boundaries=bounds)  # , format='%1i', spacing='proportional', norm=norm,

    # cbar_tick_locations = [0, 1, 2, 3, 4]
    tick_labels = [
        'Baseline',
        'Only Observed',
        'Only Projected',
        'Observed and projected'
    ]
    cbar0.set_ticklabels(tick_labels)
    cbar0.ax.tick_params(labelsize=6)

    similarity_cmap = hb.generate_custom_colorbar(similarity_array, vmin=0, vmax=1, color_scheme='bold_spectral_white_left', transparent_at_cbar_step=0)
    multiplication_factor = int(similarity_array.shape[0] / change_array.shape[0])
    change_array_r = hb.naive_upsample(change_array.astype(np.float64), multiplication_factor)

    # Make symmetric vmin-vmax to ensure zero in center
    vmin = np.min(change_array_r)
    vmax = np.max(change_array_r)
    if abs(vmin) > vmax:
        vmax = -vmin
    else:
        vmin = -vmax
    im1 = axes[1].imshow(change_array_r, vmin=vmin, vmax=vmax, cmap='BrBG')
    im2 = axes[1].imshow(similarity_array, cmap=similarity_cmap)

    cbar1 = plt.colorbar(im1, ax=axes[1], orientation='vertical')
    cbar1.set_label('Net hectare change', size=9)
    cbar1.ax.tick_params(labelsize=6)

    axes[0].set_title('Class ' + str(lulc_class) + ' observed vs. projected expansions')
    axes[1].set_title('Coarse change and difference score')

    axes[0].title.set_fontsize(10)
    axes[1].title.set_fontsize(10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=600)
    plt.close()


# Helper
def show_overall_lulc_fit(lulc_baseline_array, lulc_observed_array, projected_lulc_array, difference_metric, output_path, **kwargs):
    fig, axes = plt.subplots(2, 2)

    for ax in fig.get_axes():
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    lulc_cmap = hb.generate_custom_colorbar(lulc_baseline_array, vmin=0, vmax=5, color_scheme='seals_simplified', transparent_at_cbar_step=0)
    score_cmap = hb.generate_custom_colorbar(difference_metric, color_scheme='bold_spectral_white_left', transparent_at_cbar_step=0)

    im_00 = axes[0, 0].imshow(lulc_baseline_array, vmin=0, vmax=10, alpha=1, cmap=lulc_cmap)
    im_01 = axes[0, 1].imshow(lulc_observed_array, vmin=0, vmax=10, alpha=1, cmap=lulc_cmap)
    im_10 = axes[1, 0].imshow(projected_lulc_array, vmin=0, vmax=10, alpha=1, cmap=lulc_cmap)
    im_11 = axes[1, 1].imshow(difference_metric, vmin=0, vmax=1, alpha=1, cmap=score_cmap)

    cbar_tick_locations = [0, 1, 2, 3, 4]
    tick_labels = [
        '',
        'Baseline',
        'Only Observed',
        'Only Projected',
        'Observed and projected'
    ]

    # cbar0 = plt.colorbar(im_10, ax=axes[1, 0], orientation='horizontal', aspect=33, shrink=0.7)
    # cbar0.set_ticks(cbar_tick_locations)
    # cbar0.set_ticklabels(tick_labels)
    # cbar0.ax.tick_params(rotation=-30)
    # cbar0.ax.tick_params(labelsize=6)
    #
    # cbar1 = plt.colorbar(im_11, ax=axes[1, 1], orientation='horizontal', aspect=33, shrink=0.7)
    # cbar1.set_label('Difference score', size=9)
    # cbar1.ax.tick_params(labelsize=6)

    axes[0, 0].set_title('Baseline')
    axes[0, 1].set_title('Observed future')
    axes[1, 0].set_title('Projected future')
    axes[1, 1].set_title('Difference score')

    axes[0, 0].title.set_fontsize(10)
    axes[0, 1].title.set_fontsize(10)
    axes[1, 0].title.set_fontsize(10)
    axes[1, 1].title.set_fontsize(10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, )
    plt.close()


# # Helper
# def show_class_expansions_vs_change(lulc_baseline_array, projected_lulc_array, class_id, change_array, output_path, **kwargs):
#     fig, ax = plt.subplots(1, 1)
#
#     for ax in fig.get_axes():
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['bottom'].set_visible(False)
#         ax.spines['left'].set_visible(False)
#
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#
#     lulc_cmap = ge.generate_custom_colorbar(projected_lulc_array, color_scheme='spectral_bold_white_left', transparent_at_cbar_step=0)
#     current_class_expansions = np.where((projected_lulc_array == class_id) & (lulc_baseline_array != class_id), 1, 0)
#     current_class_contractions = np.where((projected_lulc_array != class_id) & (lulc_baseline_array == class_id), 2, 0)
#     combined = current_class_expansions + current_class_contractions
#
#     multiplication_factor = int(projected_lulc_array.shape[0] / change_array.shape[0])
#     change_array_r = hb.naive_upsample(change_array.astype(np.float64), multiplication_factor)
#
#     # Make symmetric vmin-vmax to ensure zero in center
#     vmin = np.min(change_array_r)
#     vmax = np.max(change_array_r)
#     if abs(vmin) > vmax:
#         vmax = -vmin
#     else:
#         vmin = -vmax
#
#     im1 = ax.imshow(change_array_r, vmin=vmin, vmax=vmax, cmap='BrBG')
#     im2 = ax.imshow(combined, cmap=lulc_cmap)
#
#     bounds = np.linspace(1, 3, 3)
#     bounds = [i - .5 for i in bounds]
#     # norm = matplotlib.colors.BoundaryNorm(bounds, lulc_cmap.N)
#
#     ticks = np.linspace(1, 2, 2)
#     cbar0 = plt.colorbar(im2, ax=ax, orientation='vertical', aspect=20, shrink=0.5, cmap=lulc_cmap, ticks=ticks, boundaries=bounds)  # , format='%1i', spacing='proportional', norm=norm,
#
#     tick_labels = [
#         'Expansion',
#         'Contraction',
#     ]
#     cbar0.set_ticklabels(tick_labels)
#     cbar0.ax.tick_params(labelsize=6)
#
#     if kwargs.get('title'):
#         ax.set_title(kwargs['title'])
#         ax.title.set_fontsize(10)
#
#     fig.tight_layout()
#     fig.savefig(output_path, dpi=600, )
#     plt.close()


## HYBRID FUNCTION
def plot_generation(p):
    # LEARNING POINT: I had assigned these as p.projected_lulc_af, which because they were project level, means they couldn't be deleted as intermediates.
    projected_lulc_path = os.path.join(p.cur_dir, 'projected_lulc.tif')
    projected_lulc_af = hb.ArrayFrame(projected_lulc_path)
    overall_similarity_plot_af = hb.ArrayFrame(os.path.join(p.cur_dir, 'overall_similarity_plot.tif'))
    lulc_baseline_af = hb.ArrayFrame(p.lulc_baseline_path)
    # p.lulc_observed_af = hb.ArrayFrame(p.lulc_observed_path)
    lulc_observed_af = hb.ArrayFrame(p.lulc_t2_path)


    coarse_change_paths = hb.list_filtered_paths_nonrecursively(p.coarse_change_dir, include_extensions='.tif')
    scaled_proportion_to_allocate_paths = []
    for path in coarse_change_paths:
        scaled_proportion_to_allocate_paths.append(os.path.join(p.coarse_change_dir, os.path.split(path)[1]))

    overall_similarity_sum = np.sum(overall_similarity_plot_af.data)
    for i in p.change_class_labels:
        difference_metric_path = os.path.join(p.cur_dir, 'class_' + str(i - 1) + '_similarity_plots.tif')
        difference_metric = hb.as_array(difference_metric_path)


        change_array = hb.as_array(scaled_proportion_to_allocate_paths[i - 1])

        annotation_text = """Class 
similarity:

""" + str(round(np.sum(difference_metric))) + """
.

Weighted
class
similarity:

""" + str(round(np.sum(difference_metric) / np.count_nonzero(np.where((projected_lulc_af.data == i) & (lulc_baseline_af.data != i), 1, 0)), 3)) + """


Overall
similarity
sum:

""" + str(round(np.sum(overall_similarity_sum), 3)) + """
"""

        # hb.pp(hb.enumerate_array_as_odict(p.lulc_baseline_af.data))
        # hb.pp(hb.enumerate_array_as_odict(p.lulc_observed_af.data))
        # hb.pp(hb.enumerate_array_as_odict(p.projected_lulc_af.data))

        output_path = os.path.join(p.cur_dir, 'class_' + str(i) + '_observed_vs_projected.png')
        show_lulc_class_change_difference(lulc_baseline_af.data, lulc_observed_af.data, projected_lulc_af.data, i, difference_metric,
                                          change_array, annotation_text, output_path)

        output_path = os.path.join(p.cur_dir, 'class_' + str(i) + '_projected_expansion_and_contraction.png')
        show_class_expansions_vs_change(lulc_baseline_af.data, projected_lulc_af.data, i, change_array, output_path, title='Class ' + str(i) + ' projected expansion and contraction on coarse change')

    output_path = os.path.join(p.cur_dir, 'lulc_comparison_and_scores.png')
    show_overall_lulc_fit(lulc_baseline_af.data, lulc_observed_af.data, projected_lulc_af.data, overall_similarity_plot_af.data, output_path, title='Overall LULC and fit')

    overall_similarity_plot_af = None

def show_class_expansions_vs_change(lulc_baseline_array, projected_lulc_array, class_id, change_array, output_path, **kwargs):
    """Change array is the COARSE net change of class_id"""

    # GridSpec lets me say that 5/6ths of the plot should be the imshow and the bottom should be the axes.
    fig = plt.figure()
    gs = fig.add_gridspec(6, 6)
    top_ax = fig.add_subplot(gs[0:5, :])
    bottom_left_ax = fig.add_subplot(gs[5, 0:3])
    bottom_right_ax = fig.add_subplot(gs[5, 3:6])

    # Remove all spines and lines.
    for ax in fig.get_axes():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


    # Calculate the net change array so that 1 = expansion, 2 = current, and 3 = contraction, and then make 0 transparent.
    n_plot_values = 3
    current_class_expansions = np.where((projected_lulc_array == class_id) & (lulc_baseline_array != class_id), 1, 0).astype(np.float32)
    current_class_presence = np.where((projected_lulc_array == class_id) & (lulc_baseline_array == class_id), 2, 0).astype(np.float32)
    current_class_contractions = np.where((projected_lulc_array != class_id) & (lulc_baseline_array == class_id), 3, 0).astype(np.float32)
    combined = current_class_expansions + current_class_presence + current_class_contractions
    combined[combined == 0] = np.nan # This makes it transparent so you can see behind.

    # Need to upsample the coarse resolution to the fine reslution so that they can be plotted on the same axis.
    multiplication_factor = int(projected_lulc_array.shape[0] / change_array.shape[0]) # Scales up so that it equals hectares still.
    change_array_r = hb.naive_upsample(change_array.astype(np.float64), multiplication_factor)

    # Make symmetric vmin-vmax to ensure zero in center
    vmin = np.min(change_array_r[change_array_r != -9999.0])
    vmax = np.max(change_array_r[change_array_r != -9999.0])

    if abs(vmin) > vmax:
        vmax = -vmin
    else:
        vmin = -vmax

    # Spread the axis a but so it doesn't look so saturated.
    vmin *= 1.5
    vmax *= 1.5

    im1 = top_ax.imshow(change_array_r, vmin=vmin, vmax=vmax, cmap='BrBG')
    im2 = top_ax.imshow(combined, vmin=1, vmax=3, cmap='RdYlBu_r')

    bounds = np.linspace(1, n_plot_values + 1, n_plot_values + 1)
    bounds = [i - .5 for i in bounds]
    ticks = np.linspace(1, n_plot_values, n_plot_values)

    cbar1 = plt.colorbar(im1, ax=bottom_left_ax, orientation='horizontal', aspect=20, shrink=1)  # , format='%1i', spacing='proportional', norm=norm,
    cbar1.ax.tick_params(labelsize=6)
    cbar1.set_label('Coarse resolution: net change', fontsize=7)

    cbar2 = plt.colorbar(im2, ax=bottom_right_ax, orientation='horizontal', aspect=20, shrink=1, ticks=ticks, boundaries=bounds)  # , format='%1i', spacing='proportional', norm=norm,
    # cbar = plt.colorbar(im2, ax=ax, orientation='vertical', aspect=20, shrink=0.5, cmap=lulc_cmap, ticks=ticks, boundaries=bounds)  # , format='%1i', spacing='proportional', norm=norm,
    tick_labels = ['Expansion', 'Current', 'Contraction']
    cbar2.set_ticklabels(tick_labels)
    cbar2.ax.tick_params(labelsize=6)
    cbar2.set_label('Fine resolution: specific changes', fontsize=7)



    if kwargs.get('title'):
        top_ax.set_title(kwargs['title'])
        top_ax.title.set_fontsize(10)
    #
    fig.tight_layout()

    # print('fig', fig)
    # print('output_path', output_path, hb.path_exists(output_path))
    fig.savefig('test.png', dpi=600, )
    fig.savefig(output_path, dpi=600)
    plt.close()


def plot_coefficients(output_dir, spatial_layer_coefficients_2d):
    fig, ax = plt.subplots(1, 1)

    for ax in fig.get_axes():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    for c, ax in enumerate(fig.get_axes()):
        if c < spatial_layer_coefficients_2d.shape[0]:
            im = ax.imshow(spatial_layer_coefficients_2d.T, vmin=-1, vmax=1, cmap='BrBG')
            ax.set_title('Coefficients')
            ax.title.set_fontsize(8)

    output_path = hb.ruri(os.path.join(output_dir, 'Coefficients.png'))
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, )
    plt.close()


def plot_coarse_change_3d(output_dir, coarse_change_3d):
    plot_n_r, plot_n_c = int(math.ceil(float(coarse_change_3d.shape[0]) ** .5)), int(math.floor(float(coarse_change_3d.shape[0] ** .5)))
    fig, ax = plt.subplots(plot_n_c, plot_n_r)

    for ax in fig.get_axes():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    for c, ax in enumerate(fig.get_axes()):
        if c < coarse_change_3d.shape[0]:
            vmin = np.min(coarse_change_3d)
            vmax = np.max(coarse_change_3d)
            if abs(vmin) < abs(vmax):
                vmin = vmax * -1

            if abs(vmin) > abs(vmax):
                vmax = vmin * -1
            im = ax.imshow(coarse_change_3d[c], vmin=vmin, vmax=vmax, cmap='BrBG')
            ax.set_title('Class ' + str(c) + ' change')
            ax.title.set_fontsize(8)

    output_path = os.path.join(output_dir, 'coarse_change.png')
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, )
    plt.close()


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
        p.L.debug('Calculating similarity for class ' + str(class_id))

        observed_changes_to = np.where((observed_array == class_id) & (baseline_array != class_id), 1, 0)
        observed_changes_from = np.where((observed_array != class_id) & (baseline_array == class_id), 1, 0)

        projected_changes_to = np.where((projected_array == class_id) & (baseline_array != class_id), 1, 0)
        projected_changes_from = np.where((projected_array != class_id) & (baseline_array == class_id), 1, 0)

        sum_metric, n_predictions, observed_not_projected_to_metric, projected_not_observed_to_metric = calc_similarity_of_two_arrays(observed_changes_to, projected_changes_to)

        if n_predictions:
            avg = sum_metric / n_predictions
        else:
            avg = 0

        p.L.debug('TO sum_metric: ' + str(sum_metric) + ' n_predictions: ' + str(n_predictions) + ' avg: ' + str(avg))

        sum_metrics_to.append(sum_metric)
        n_predictions_to.append(n_predictions)

        sum_metric, n_predictions, observed_not_projected_from_metric, projected_not_observed_from_metric = calc_similarity_of_two_arrays(observed_changes_from,
                                                                                                                                          projected_changes_from)

        if n_predictions:
            avg = sum_metric / n_predictions
        else:
            avg = 0

        p.L.debug('FROM sum_metric: ' + str(sum_metric) + ' n_predictions: ' + str(n_predictions) + ' avg: ' + str(avg))

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
            p.L.debug('Projected zero change, but there was observed change. It is unclear to assess the similarity of these metrics.')
            a1_not_a2_distance = scipy.ndimage.morphology.distance_transform_edt(a1_not_a2_flipped_array.astype(np.float64)) ** 2
            a1_not_a2_metric = np.where(a2 == 1, a1_not_a2_distance, 0).astype(np.float64)
            a2_not_a1_metric = np.zeros(a1_has_values.shape)
    else:
        if a2_has_values:
            p.L.debug('Observed zero change, but there was projected change. It is unclear to assess the similarity of these metrics.')
            a1_not_a2_metric = np.zeros(a1_has_values.shape).astype(np.float64)
            a2_not_a1_distance = scipy.ndimage.morphology.distance_transform_edt(a2_not_a1_flipped_array.astype(np.float64)) ** 2
            a2_not_a1_metric = np.where(a1 == 1, a2_not_a1_distance, 0).astype(np.float64)
        else:
            p.L.debug('Projected and observed zero change. It is unclear to assess the similarity of these metrics.')
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
        p.L.warning('WARNING NOTHING came up as above the blurring threshold!')

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


def fft_gaussian(signal_path, kernel_path, target_path, target_nodata=-9999.0, compress=False, n_threads=1):
    """
    Blur the input with a guassian using sigma (higher sigma means more blur). In areas of the blur above the threshold,
    return 1 - blurred so thtat values near zero indicate strong presence. In areas below the threshold, return


    The positive attribute of this func is it gives an s-curve relationship between 0 and 1 with a smoothed discontinuity
    around the threshold while never converging too close to 1 even at extreme distances without requiring slow calculation
    of a large footrint convolution.
    """
    print('signal_path, kernel_path, target_path', signal_path, kernel_path, target_path)
    signal_path_band = (signal_path, 1)
    kernel_path_band = (kernel_path, 1)

    if compress:
        gtiff_creation_options = hb.DEFAULT_GTIFF_CREATION_OPTIONS
    else:
        gtiff_creation_options = hb.DEFAULT_GTIFF_CREATION_OPTIONS

    raster_driver_creation_tuple = ('GTiff', gtiff_creation_options)
    hb.convolve_2d(
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


def sigmoidal_curve(x, left_value, inflection_point, steepness, magnitude, e=hb.e):
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

