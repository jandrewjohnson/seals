

from matplotlib import colors as colors
from matplotlib import pyplot as plt
import numpy as np

def general_plots():
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8)
    full_change_matrix_no_diagonal = 'DO'
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

def change_pngs(passed_p=None):
    if passed_p is None:
        global p
    else:
        p = passed_p

    if p.run_this:

        for scenario_name in [p.scenario_label]:
            p.lulc_baseline_af = hb.ArrayFrame(p.lulc_baseline_path)
            lulc_projected_path = os.path.join(p.allocation_dir, 'lulc_' + scenario_name + '.tif')
            p.lulc_projected_af = hb.ArrayFrame(lulc_projected_path)


            for c, path in enumerate(p.projected_current_coarse_change_input_paths):
                scaled_proportion_to_allocate_path = os.path.join(p.allocation_dir, os.path.split(path)[1])
                change_array = hb.as_array(scaled_proportion_to_allocate_path)
                output_path = os.path.join(p.cur_dir, scenario_name + '_class_' + str(c + 1) + '_projected_expansion_and_contraction.png')

                seals_utils.show_class_expansions_vs_change(p.lulc_baseline_af.data, p.lulc_projected_af.data, c + 1, change_array, output_path,
                                                title='Class ' + str(c + 1) + ' projected expansion and contraction on coarse change')


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

    cmap = ge.generate_custom_colorbar(classes, color_scheme='bold_spectral_white_left', transparent_at_cbar_step=0)
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

    similarity_cmap = ge.generate_custom_colorbar(similarity_array, vmin=0, vmax=1, color_scheme='bold_spectral_white_left', transparent_at_cbar_step=0)
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

def show_overall_lulc_fit(baseline_lulc_array, observed_lulc_array, projected_lulc_array, difference_metric, output_path, **kwargs):
    fig, axes = plt.subplots(2, 2)

    for ax in fig.get_axes():
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    lulc_cmap = ge.generate_custom_colorbar(baseline_lulc_array, vmin=0, vmax=5, color_scheme='seals_simplified', transparent_at_cbar_step=0)
    score_cmap = ge.generate_custom_colorbar(difference_metric, color_scheme='bold_spectral_white_left', transparent_at_cbar_step=0)

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

# Helper
def show_class_expansions_vs_change(baseline_lulc_array, projected_lulc_array, class_id, change_array, output_path, **kwargs):
    fig, ax  = plt.subplots(1, 1)

    for ax in fig.get_axes():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    lulc_cmap = ge.generate_custom_colorbar(projected_lulc_array, color_scheme='spectral_bold_white_left', transparent_at_cbar_step=0)
    current_class_expansions = np.where((projected_lulc_array == class_id) & (baseline_lulc_array != class_id), 1, 0)
    current_class_contractions = np.where((projected_lulc_array != class_id) & (baseline_lulc_array == class_id), 2, 0)
    combined = current_class_expansions + current_class_contractions

    multiplication_factor = int(projected_lulc_array.shape[0] / change_array.shape[0])
    change_array_r = hb.naive_upsample(change_array.astype(np.float64), multiplication_factor)

    # Make symmetric vmin-vmax to ensure zero in center
    vmin = np.min(change_array_r)
    vmax = np.max(change_array_r)
    if abs(vmin) > vmax:
        vmax = -vmin
    else:
        vmin = -vmax

    im1 = ax.imshow(change_array_r, vmin=vmin, vmax=vmax, cmap='BrBG')
    im2 = ax.imshow(combined, cmap=lulc_cmap)

    bounds = np.linspace(1, 3, 3)
    bounds = [i - .5 for i in bounds]
    # norm = matplotlib.colors.BoundaryNorm(bounds, lulc_cmap.N)

    ticks = np.linspace(1, 2, 2)
    cbar0 = plt.colorbar(im2, ax=ax, orientation='vertical', aspect=20, shrink=0.5, cmap=lulc_cmap, ticks=ticks, boundaries=bounds) # , format='%1i', spacing='proportional', norm=norm,

    tick_labels = [
        'Expansion',
        'Contraction',
    ]
    cbar0.set_ticklabels(tick_labels)
    cbar0.ax.tick_params(labelsize=6)

    if kwargs.get('title'):
        ax.set_title(kwargs['title'])
        ax.title.set_fontsize(10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, )
    plt.close()


## HYBRID FUNCTION
def plot_generation(p, generation_id):
    projected_lulc_path = os.path.join(p.optimized_seals_run_dir, 'gen' + str(generation_id).zfill(6) + '_predicted_lulc.tif')
    p.projected_lulc_af = hb.ArrayFrame(projected_lulc_path)
    p.overall_similarity_plot_af = hb.ArrayFrame(os.path.join(p.optimized_seals_run_dir, 'gen' + str(generation_id).zfill(6) + '_overall_similarity_plot.tif'))

    overall_similarity_sum = np.sum(p.overall_similarity_plot_af.data)
    for i in p.change_class_labels:
        difference_metric_path = os.path.join(p.optimized_seals_run_dir, 'gen' + str(generation_id).zfill(6) + '_class_' + str(i - 1) + '_similarity.tif')
        difference_metric = hb.as_array(difference_metric_path)
        change_array = hb.as_array(p.coarse_change_paths[i - 1])

        annotation_text = """Class 
similarity:

""" + str(round(np.sum(difference_metric))) + """


Weighted
class
similarity:

""" + str(round(np.sum(difference_metric) / np.count_nonzero(np.where((p.projected_lulc_af.data == i) & (p.baseline_lulc_af.data != i), 1, 0)), 3)) + """


Overall
similarity
sum:

""" + str(round(np.sum(overall_similarity_sum), 3)) + """
"""

        output_path = os.path.join(p.cur_dir, 'gen' + str(generation_id).zfill(6) + '_class_' + str(i) + '_observed_vs_projected.png')
        show_lulc_class_change_difference(p.baseline_lulc_af.data, p.observed_lulc_af.data, p.projected_lulc_af.data, i, difference_metric,
                                          change_array, annotation_text, output_path)

        output_path = os.path.join(p.cur_dir, 'gen' + str(generation_id).zfill(6) + '_class_' + str(i) + '_projected_expansion_and_contraction.png')
        show_class_expansions_vs_change(p.baseline_lulc_af.data, p.projected_lulc_af.data, i, change_array, output_path, title='Class ' + str(i) + ' projected expansion and contraction on coarse change')

    output_path = os.path.join(p.cur_dir, 'gen' + str(generation_id).zfill(6) + '_lulc_comparison_and_scores.png')
    show_overall_lulc_fit(p.baseline_lulc_af.data, p.observed_lulc_af.data, p.projected_lulc_af.data, p.overall_similarity_plot_af.data, output_path, title='Overall LULC and fit')


def plot_final_run():
    global p
    if p.run_this:
        if not getattr(p, 'final_generation_id', None):
            p.final_generation_id = 0
        plot_generation(p, p.final_generation_id)

