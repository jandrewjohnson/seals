import os

import hazelbean as hb
import numpy as np
import pandas as pd
from matplotlib import colors as colors
from matplotlib import pyplot as plt
import numpy as np
import hazelbean as hb
import os
from seals import seals_utils
import pandas as pd


def show_lulc_class_change_difference(baseline_array, observed_array, projected_array, lulc_class, similarity_array, change_array, annotation_text, output_path, **kwargs):
    fig, axes = plt.subplots(2, 1)

    classes = np.zeros(observed_array.shape)
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

    # TODO Extract from geoecon
    cmap = hb.generate_custom_colorbar(classes, color_scheme='bold_spectral_white_left', transparent_at_cbar_step=0)
    im_top_0 = axes[0].imshow(classes, cmap=cmap)

    bounds = np.linspace(1, 3, 3)
    bounds = [.5, 1.5, 2.5, 3.5, 4.5]

    # ticks = np.linspace(1, 2, 2)
    ticks = [1, 2, 3, 4]
    cbar0 = plt.colorbar(im_top_0, ax=axes[0], orientation='vertical', aspect=20, shrink=1.0, cmap=cmap, ticks=ticks, boundaries=bounds)  # , format='%1i', spacing='proportional', norm=norm,

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
    im1 = axes[1].imshow(change_array_r,  vmin=vmin, vmax=vmax, cmap='BrBG')
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

def show_overall_lulc_fit(baseline_lulc_array, observed_lulc_array, projected_lulc_array, difference_metric, output_path, indices_to_labels_dict, **kwargs):
    fig, axes = plt.subplots(2, 2)

    for ax in fig.get_axes():
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    lulc_cmap = hb.generate_custom_colorbar(baseline_lulc_array, vmin=0, vmax=5, color_scheme='seals_simplified', transparent_at_cbar_step=0)
    score_cmap = hb.generate_custom_colorbar(difference_metric, color_scheme='bold_spectral_white_left', transparent_at_cbar_step=0)

    im_00 = axes[0, 0].imshow(baseline_lulc_array, vmin=0, vmax=10, alpha=1, cmap=lulc_cmap)
    im_01 = axes[0, 1].imshow(observed_lulc_array, vmin=0, vmax=10, alpha=1, cmap=lulc_cmap)
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

def plot_array_as_seals7_lulc(input_array, output_path, title, indices_to_labels_dict, dpi=900):
    from hazelbean.visualization import generate_custom_colorbar
    fig, ax = plt.subplots()
    cmap = generate_custom_colorbar(input_array, color_scheme='seals_simplified')
    im = ax.imshow(input_array, cmap=cmap, vmin=0, vmax=10, interpolation='nearest')

    max_cbar_category = 7
    min_cbar_category = 1
    n_categories = 7

    bin_size = (max_cbar_category - min_cbar_category) / (n_categories - 1)
    bounds = np.linspace(min_cbar_category, max_cbar_category + 1, n_categories + 1)
    bounds = [i - bin_size / 2.0 for i in bounds]

    ticks = np.linspace(min_cbar_category, max_cbar_category, n_categories)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    orientation = 'vertical'
    aspect = 20
    shrink = .50
    # cbar = plt.colorbar(im, ax=ax, cmap=cmap, orientation=orientation, )  # , format='%1i', spacing='proportional', norm=norm,
    # cbar = plt.colorbar(im, ax=ax, cmap=cmap, ticks=ticks, boundaries=bounds, orientation=orientation, )  # , format='%1i', spacing='proportional', norm=norm,
    cbar = plt.colorbar(im, ax=ax, orientation=orientation, aspect=aspect, shrink=shrink, cmap=cmap, ticks=ticks, boundaries=bounds)  # , format='%1i', spacing='proportional', norm=norm,


    ticklabels = [i.title() for i in list([i for i in indices_to_labels_dict.values() if 'ndv' not in i.lower()])]
    # ticklabels = [i.title() for i in list(indices_to_labels_dict.values())[1:]]
    cbar.set_ticklabels(ticklabels)

    tick_labelsize = 8
    cbar.ax.tick_params(labelsize=tick_labelsize)

    ax.set_title(title)

    ax.title.set_fontsize(12)

    fig.tight_layout()

    fig.savefig(output_path, dpi=dpi)
    plt.close()


def show_class_expansions_vs_change_underneath(lulc_baseline_array, projected_lulc_array, class_id, change_array, output_path, **kwargs):
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

    import hazelbean.calculation_core.aspect_ratio_array_functions
    change_array_r = hazelbean.calculation_core.aspect_ratio_array_functions.naive_downscale(change_array.astype(np.float64), multiplication_factor)

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

    fig.tight_layout()


    fig.savefig('test.png', dpi=600, )
    fig.savefig(output_path, dpi=600)
    plt.close()


def show_class_expansions_vs_change(lulc_baseline_array, projected_lulc_array, class_id, change_array, output_path, **kwargs):
    """Change array is the COARSE net change of class_id"""

    # GridSpec lets me say that 5/6ths of the plot should be the imshow and the bottom should be the axes.
    fig = plt.figure()
    gs = fig.add_gridspec(7, 6)
    title_ax = fig.add_subplot(gs[0, 0:6])
    top_left_ax = fig.add_subplot(gs[1:6, 0:3])
    top_right_ax = fig.add_subplot(gs[1:6, 3:6])
    bottom_left_ax = fig.add_subplot(gs[6, 0:3])
    bottom_right_ax = fig.add_subplot(gs[6, 3:6])

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

    import hazelbean.calculation_core.aspect_ratio_array_functions
    change_array_r = hazelbean.calculation_core.aspect_ratio_array_functions.naive_downscale(change_array.astype(np.float64), multiplication_factor)

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

    im1 = top_left_ax.imshow(change_array_r, vmin=vmin, vmax=vmax, cmap='BrBG')
    im2 = top_right_ax.imshow(combined, vmin=1, vmax=3, cmap='RdYlBu_r')

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
        title_ax.set_title(kwargs['title'])
        title_ax.title.set_fontsize(10)

    fig.tight_layout()

    fig.savefig(output_path, dpi=600)
    plt.close()

def show_class_expansions_vs_change_with_numeric_report(lulc_baseline_array, projected_lulc_array, class_id, change_array, ha_per_cell_coarse_array, ha_per_cell_fine_array, output_path, **kwargs):
    """Change array is the COARSE net change of class_id"""

    # GridSpec lets me say that 5/6ths of the plot should be the imshow and the bottom should be the axes.
    fig = plt.figure()
    gs = fig.add_gridspec(7, 8)
    title_ax = fig.add_subplot(gs[0, 0:6])
    top_left_ax = fig.add_subplot(gs[1:6, 0:3])
    top_right_ax = fig.add_subplot(gs[1:6, 3:6])
    bottom_left_ax = fig.add_subplot(gs[6, 0:3])
    bottom_right_ax = fig.add_subplot(gs[6, 3:6])
    report_ax = fig.add_subplot(gs[1:6, 6:8])

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

    import hazelbean.calculation_core.aspect_ratio_array_functions
    change_array_r = hazelbean.calculation_core.aspect_ratio_array_functions.naive_downscale(change_array.astype(np.float64), multiplication_factor)

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

    im1 = top_left_ax.imshow(change_array_r, vmin=vmin, vmax=vmax, cmap='BrBG')
    im2 = top_right_ax.imshow(combined, vmin=1, vmax=3, cmap='RdYlBu_r')

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
        title_ax.set_title(kwargs['title'])
        title_ax.title.set_fontsize(10)

    fig.tight_layout()

    fine_cells_per_coarse_cell = np.mean(ha_per_cell_coarse_array) / np.mean(ha_per_cell_fine_array)

    n_digits = 3

    report_string = ''
    report_string += 'n cells before: ' + str(np.count_nonzero(lulc_baseline_array == class_id)) + '\n'
    report_string += 'n cells after: ' + str(np.count_nonzero(projected_lulc_array == class_id)) + '\n'

    expansions = np.count_nonzero(np.where((projected_lulc_array == class_id) & (lulc_baseline_array != class_id), 1, 0))
    contractions = np.count_nonzero(np.where((projected_lulc_array != class_id) & (lulc_baseline_array == class_id), 1, 0))

    sum_expansions = np.sum(expansions)
    sum_contractions = np.sum(contractions)

    report_string += 'expansions: ' + str(sum_expansions) + '\n'
    report_string += 'contractions: ' + str(sum_contractions) + '\n'

    net_alloc = sum_expansions - sum_contractions
    report_string += 'net alloc: ' + str(net_alloc) + '\n'


    sum_change_array = np.sum(change_array)
    mean_ha_per_cell_fine = np.mean(ha_per_cell_fine_array)
    requested = sum_change_array / mean_ha_per_cell_fine
    report_string += 'requested: ' + str(hb.round_significant_n(requested, n_digits)) + '\n'

    diff = net_alloc - requested
    report_string += 'diff: ' + str(diff) + '\n'
    # report_string += 'hectares per coarse: ' + str(hb.round_significant_n(np.mean(ha_per_cell_coarse_array), n_digits)) + '\n'
    # lulc_baseline_array, projected_lulc_array, class_id, change_array


    report_ax.annotate(report_string, xy=(0.5, 0.5), xytext=(0.5, 0.5), fontsize=7, ha='center', va='center')

    fig.savefig(output_path, dpi=600)
    plt.close()


def show_specific_class_expansions_vs_change_with_numeric_report_and_validation(lulc_baseline_array, projected_lulc_array, class_id, class_label, change_array, ha_per_cell_coarse_array, ha_per_cell_fine_array, source_dir, output_path, **kwargs):
    """Change array is the COARSE net change of class_id"""

    # GridSpec lets me say that 5/6ths of the plot should be the imshow and the bottom should be the axes.
    fig = plt.figure()
    gs = fig.add_gridspec(7, 11)
    title_ax = fig.add_subplot(gs[0, 0:6])
    top_left_ax = fig.add_subplot(gs[1:6, 0:3])
    top_right_ax = fig.add_subplot(gs[1:6, 3:6])
    bottom_left_ax = fig.add_subplot(gs[6, 0:3])
    bottom_right_ax = fig.add_subplot(gs[6, 3:6])
    report_ax = fig.add_subplot(gs[1:6, 6:8])
    top_right_right_ax = fig.add_subplot(gs[1:6, 8:11])
    bottom_right_right_ax = fig.add_subplot(gs[6, 8:11])

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

    import hazelbean.calculation_core.aspect_ratio_array_functions
    change_array_r = hazelbean.calculation_core.aspect_ratio_array_functions.naive_downscale(change_array.astype(np.float64), multiplication_factor)

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

    validation_dir = os.path.join(source_dir, 'validation')
    validation_path = os.path.join(validation_dir, class_label + '_allocated_prop.tif')

    validation_array = hb.as_array(validation_path)

    im1 = top_left_ax.imshow(change_array_r, vmin=vmin, vmax=vmax, cmap='BrBG')
    im2 = top_right_ax.imshow(combined, vmin=1, vmax=3, cmap='RdYlBu_r')
    im3 = top_right_right_ax.imshow(validation_array, vmin=vmin, vmax=vmax, cmap='BrBG')

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

    cbar3 = plt.colorbar(im3, ax=bottom_right_right_ax, orientation='horizontal', aspect=20, shrink=1)  # , format='%1i', spacing='proportional', norm=norm,
    cbar3.ax.tick_params(labelsize=6)
    cbar3.set_label('Allocated change', fontsize=7)



    if kwargs.get('title'):
        title_ax.set_title(kwargs['title'])
        title_ax.title.set_fontsize(10)

    fig.tight_layout()

    fine_cells_per_coarse_cell = np.mean(ha_per_cell_coarse_array) / np.mean(ha_per_cell_fine_array)

    n_digits = 3

    report_string = ''
    report_string += 'n cells before: ' + str(np.count_nonzero(lulc_baseline_array == class_id)) + '\n'
    report_string += 'n cells after: ' + str(np.count_nonzero(projected_lulc_array == class_id)) + '\n'

    expansions = np.count_nonzero(np.where((projected_lulc_array == class_id) & (lulc_baseline_array != class_id), 1, 0))
    contractions = np.count_nonzero(np.where((projected_lulc_array != class_id) & (lulc_baseline_array == class_id), 1, 0))

    sum_expansions = np.sum(expansions)
    sum_contractions = np.sum(contractions)

    report_string += 'expansions: ' + str(sum_expansions) + '\n'
    report_string += 'contractions: ' + str(sum_contractions) + '\n'

    net_alloc = sum_expansions - sum_contractions
    report_string += 'net alloc: ' + str(net_alloc) + '\n'


    sum_change_array = np.sum(change_array)
    mean_ha_per_cell_fine = np.mean(ha_per_cell_fine_array)
    requested = sum_change_array / mean_ha_per_cell_fine
    report_string += 'requested: ' + str(hb.round_significant_n(requested, n_digits)) + '\n'

    diff = net_alloc - requested
    report_string += 'diff: ' + str(diff) + '\n'
    # report_string += 'hectares per coarse: ' + str(hb.round_significant_n(np.mean(ha_per_cell_coarse_array), n_digits)) + '\n'
    # lulc_baseline_array, projected_lulc_array, class_id, change_array


    report_ax.annotate(report_string, xy=(0.5, 0.5), xytext=(0.5, 0.5), fontsize=7, ha='center', va='center')



    fig.savefig(output_path, dpi=600)
    plt.close()


def show_all_class_expansions_vs_change_with_numeric_report_and_validation(lulc_baseline_array, projected_lulc_array, class_ids, class_labels, change_array_paths, ha_per_cell_coarse_array, ha_per_cell_fine_array, source_dir, output_path, **kwargs):
    """Change array is the COARSE net change of class_id"""

    # GridSpec lets me say that 5/6ths of the plot should be the imshow and the bottom should be the axes.
    fig = plt.figure(figsize=(6, 7.5))
    n_rows = len(class_ids) * 3 + 1 # * 3 so that the plots are bigger than the cbars,+ 2 for title and cbars
    plt.margins(0,0)

    gs = fig.add_gridspec(n_rows + 1, 12, wspace=0.01, hspace=0.01)
    # title_ax = fig.add_subplot(gs[0, 0:12])

    axes_grid = []
    for c, class_id in enumerate(class_ids):
        left_col = 0 + c * 3
        right_col = 3 + c * 3
        axes_grid.append([])

        axes_grid[c].append(fig.add_subplot(gs[left_col:right_col, 0:3]))
        axes_grid[c].append(fig.add_subplot(gs[left_col:right_col, 3:6]))
        axes_grid[c].append(fig.add_subplot(gs[left_col:right_col, 6:9]))
        axes_grid[c].append(fig.add_subplot(gs[left_col:right_col, 9:12]))

    bottom_left_ax = fig.add_subplot(gs[right_col, 0:3])
    bottom_right_ax = fig.add_subplot(gs[right_col, 3:6])
    bottom_right_right_ax = fig.add_subplot(gs[right_col, 6:9])

    # Remove all spines and lines.
    for ax in fig.get_axes():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


    for c, class_id in enumerate(class_ids):
        class_label = class_labels[c]
        change_array = hb.as_array(change_array_paths[c])

        # Calculate the net change array so that 1 = expansion, 2 = current, and 3 = contraction, and then make 0 transparent.
        n_plot_values = 3
        current_class_expansions = np.where((projected_lulc_array == class_id) & (lulc_baseline_array != class_id), 1, 0).astype(np.float32)
        current_class_presence = np.where((projected_lulc_array == class_id) & (lulc_baseline_array == class_id), 2, 0).astype(np.float32)
        current_class_contractions = np.where((projected_lulc_array != class_id) & (lulc_baseline_array == class_id), 3, 0).astype(np.float32)
        combined = current_class_expansions + current_class_presence + current_class_contractions
        combined[combined == 0] = np.nan # This makes it transparent so you can see behind.

        # Need to upsample the coarse resolution to the fine reslution so that they can be plotted on the same axis.
        multiplication_factor = int(projected_lulc_array.shape[0] / change_array.shape[0]) # Scales up so that it equals hectares still.

        import hazelbean.calculation_core.aspect_ratio_array_functions
        change_array_r = hazelbean.calculation_core.aspect_ratio_array_functions.naive_downscale(change_array.astype(np.float64), multiplication_factor)

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

        validation_dir = os.path.join(source_dir, 'validation')
        validation_path = os.path.join(validation_dir, class_label + '_allocated_prop.tif')

        validation_array = hb.as_array(validation_path)

        validation_array_r = hazelbean.calculation_core.aspect_ratio_array_functions.naive_downscale(validation_array.astype(np.float64), multiplication_factor)

        # if class_label == 'grassland':
        #     print(validation_array_r)

        im1 = axes_grid[c][0].imshow(change_array_r, vmin=vmin, vmax=vmax, cmap='BrBG')
        im2 = axes_grid[c][1].imshow(combined, vmin=1, vmax=3, cmap='Spectral')
        im3 = axes_grid[c][2].imshow(validation_array_r, vmin=vmin, vmax=vmax, cmap='BrBG')

        ###------------ Report column now
        n_digits = 3

        report_string = class_label.title() + '\n\n'
        report_string += 'n cells before: ' + str(np.count_nonzero(lulc_baseline_array == class_id)) + '\n'
        report_string += 'n cells after: ' + str(np.count_nonzero(projected_lulc_array == class_id)) + '\n'

        expansions = np.count_nonzero(np.where((projected_lulc_array == class_id) & (lulc_baseline_array != class_id), 1, 0))
        contractions = np.count_nonzero(np.where((projected_lulc_array != class_id) & (lulc_baseline_array == class_id), 1, 0))

        sum_expansions = np.sum(expansions)
        sum_contractions = np.sum(contractions)

        report_string += 'expansions: ' + str(sum_expansions) + '\n'
        report_string += 'contractions: ' + str(sum_contractions) + '\n'

        net_alloc = sum_expansions - sum_contractions
        report_string += 'net alloc: ' + str(net_alloc) + '\n'

        sum_change_array = np.sum(change_array)
        mean_ha_per_cell_fine = np.mean(ha_per_cell_fine_array)
        requested = sum_change_array / mean_ha_per_cell_fine
        report_string += 'requested: ' + str(requested).split('.')[0] + '\n'

        diff = net_alloc - requested
        report_string += 'diff: ' + str(diff).split('.')[0] + '\n'
        # report_string += 'hectares per coarse: ' + str(hb.round_significant_n(np.mean(ha_per_cell_coarse_array), n_digits)) + '\n'
        # lulc_baseline_array, projected_lulc_array, class_id, change_array


        axes_grid[c][3].annotate(report_string, xy=(0.1, 0.9), xytext=(0.1, 0.9), fontsize=5, ha='left', va='top')



        if c == len(class_ids) - 1: # on the last plot, add the combined cbars etc
            bounds = np.linspace(1, n_plot_values + 1, n_plot_values + 1)
            bounds = [i - .5 for i in bounds]
            ticks = np.linspace(1, n_plot_values, n_plot_values)

            cbar1 = plt.colorbar(im1, ax=bottom_left_ax, orientation='horizontal', aspect=20, shrink=1)  # , format='%1i', spacing='proportional', norm=norm,
            cbar1.ax.tick_params(labelsize=6)
            cbar1.set_label('Requested change', fontsize=7)

            cbar2 = plt.colorbar(im2, ax=bottom_right_ax, orientation='horizontal', aspect=20, shrink=1, ticks=ticks, boundaries=bounds)  # , format='%1i', spacing='proportional', norm=norm,
            # cbar = plt.colorbar(im2, ax=ax, orientation='vertical', aspect=20, shrink=0.5, cmap=lulc_cmap, ticks=ticks, boundaries=bounds)  # , format='%1i', spacing='proportional', norm=norm,
            tick_labels = ['Expand', 'Current', 'Contract']
            cbar2.set_ticklabels(tick_labels)
            cbar2.ax.tick_params(labelsize=5)
            cbar2.set_label('Actual change', fontsize=7)

            cbar3 = plt.colorbar(im3, ax=bottom_right_right_ax, orientation='horizontal', aspect=20, shrink=1)  # , format='%1i', spacing='proportional', norm=norm,
            cbar3.ax.tick_params(labelsize=6)
            cbar3.set_label('Net change', fontsize=7)



    # if kwargs.get('title'):
    #     title_ax.set_title(kwargs['title'])
    #     title_ax.title.set_fontsize(8)

    fig.tight_layout()



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
    import math
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
