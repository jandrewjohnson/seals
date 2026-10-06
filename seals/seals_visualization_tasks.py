from matplotlib import colors as colors
from matplotlib import pyplot as plt
import numpy as np
import hazelbean as hb
import os
import pandas as pd
import geopandas as gpd
from matplotlib.patches import Patch
# import contextily as ctx
import rasterio
from rasterio.plot import show
from html import escape as _esc
from datetime import datetime
import getpass, platform

# import seals
from seals import seals_utils
from seals.seals_visualization_functions import *

def visualization(p):
    # Make folder for all visualizations.
    pass

def coarse_change_with_class_change_underneath(passed_p=None):
    p = passed_p

    if p.run_this:


        if p.scenario_definitions_path is not None:
            p.scenarios_df = pd.read_csv(p.scenario_definitions_path)
            for index, row in p.scenarios_df.iterrows():
                seals_utils.assign_df_row_to_object_attributes(p, row)
                seals_utils.set_derived_attributes(p)

                if p.scenario_type !=  'baseline':
                    max_plotting_size = 200000

                    # By default, this will select 4 zones from different parts of the list to plot full change matrices. This is slow.
                    # You can override this to plot all here:
                    # zones_to_plot = 'first' # one of first, all, or four
                    zones_to_plot = 'random_four'

                    for year_c, year in enumerate(p.years):
                        target_allocation_zones_dir = os.path.join(p.allocations_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year), 'allocation_zones')
                        seals_utils.load_blocks_list(p, target_allocation_zones_dir)

                        if year_c == 0:
                            previous_year = p.key_base_year
                        else:
                            previous_year = p.years[year_c - 1]

                        if zones_to_plot == 'all':
                            target_zones = p.global_processing_blocks_list
                        elif zones_to_plot == 'four':
                            target_zones = [p.global_processing_blocks_list[0], p.global_processing_blocks_list[int(len(p.global_processing_blocks_list)/4)], p.global_processing_blocks_list[int(len(p.global_processing_blocks_list)/2)], p.global_processing_blocks_list[int(len(p.global_processing_blocks_list)*3/4)]]
                        elif zones_to_plot == 'first':
                            target_zones = [p.global_processing_blocks_list[0]]
                        elif zones_to_plot == 'random_four':
                            n_processing_blocks = int(len(p.global_processing_blocks_list))
                            if n_processing_blocks < 4:
                                raise ValueError('There are not enough processing blocks to select four random ones. Please use a different zones_to_plot option.')
                            starting_tile_pos = [
                                        int(n_processing_blocks*(1/8)), #
                                        int(n_processing_blocks*(3/8)),
                                        int(n_processing_blocks*(5/8)),
                                        int(n_processing_blocks*(7/8)),
                                        ]
                            
                            # Iterate through the selected starting positions and check if there is an lulc_file actually existing.                            
                            target_zones = []
                            target_blocks = []
                            target_coarse_blocks = []
                            for target_pos in starting_tile_pos:
                                for i in range(int(n_processing_blocks / 4)):
                                    
                                    target_block = p.global_processing_blocks_list[target_pos + i]
                                    target_coarse_block = p.global_coarse_blocks_list[target_pos + i]
                                    target_zone = str(target_block[0] + '_' + target_block[1])
                                    ha_diff_from_previous_year_dir_to_plot = os.path.join(p.coarse_simplified_ha_difference_from_previous_year_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year))
                                    allocation_dir_to_plot = os.path.join(p.intermediate_dir, 'allocations', p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year), 'allocation_zones', str(target_zone), 'allocation')
                                    lulc_projected_path= os.path.join(allocation_dir_to_plot, 'lulc_' + p.lulc_src_label + '_'  + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(year) + '.tif')
                                    print('lulc_projected_path', lulc_projected_path)
                                    if hb.path_exists(lulc_projected_path):
                                        target_zones.append(target_zone)
                                        target_blocks.append(target_block)
                                        target_coarse_blocks.append(target_coarse_block)
                                        break
                                    if len(target_zones) >= 4:
                                        break                                
                            if len(target_zones) < 4:
                                raise ValueError('NONE OF THE BLOCKS CHECKED have lulc maps present to select four random ones. Please use a different zones_to_plot option.')

                        else:
                            raise ValueError('zones_to_plot must be one of first, all, or four')

                        # # Make sure the target zones are in the right format
                        # for c, row in enumerate(target_zones):
                        #         target_zones[c] = str(row[0] + '_' + row[1])

                        for c, target_zone in enumerate(target_zones):
                            ha_diff_from_previous_year_dir_to_plot = os.path.join(p.coarse_simplified_ha_difference_from_previous_year_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year))
                            allocation_dir_to_plot = os.path.join(p.intermediate_dir, 'allocations', p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year), 'allocation_zones', target_zone, 'allocation')
                            lulc_projected_path= os.path.join(allocation_dir_to_plot, 'lulc_' + p.lulc_src_label + '_'  + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(year) + '.tif')
                            lulc_projected_array = None

                            if previous_year == p.key_base_year:
                                lulc_previous_year_path = os.path.join(allocation_dir_to_plot, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.model_label + '_' + str(previous_year) + '.tif')

                                lulc_previous_year_array = None     # For deffered loading
                            else:
                                previous_allocation_dir_to_plot = allocation_dir_to_plot.replace('\\','/').replace('/' + str(year) + '/', '/' + str(previous_year) + '/')
                                lulc_previous_year_path = os.path.join(previous_allocation_dir_to_plot, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(previous_year) + '.tif')
                                lulc_previous_year_array = None

                            for class_id, class_label in zip(p.lulc_correspondence_dict['dst_ids'], p.lulc_correspondence_dict['dst_labels']):

                                filename = class_label + '_' + str(year) + '_' + str(previous_year) + '_ha_diff_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '.tif'
                                scaled_proportion_to_allocate_path = os.path.join(ha_diff_from_previous_year_dir_to_plot, filename)
                                output_path = os.path.join(p.cur_dir, str(year) + '_' + target_zone + '_' + class_label + '_projected_expansion_and_contraction.png')
                                
                                if hb.path_exists(scaled_proportion_to_allocate_path) and not hb.path_exists(output_path):
                                    hb.log('Plotting ' + output_path)
                                    if lulc_projected_array is None:
                                        lulc_projected_array = hb.as_array_resampled_to_size(lulc_projected_path, max_plotting_size)


                                    if lulc_previous_year_array is None:
                                        lulc_previous_year_array = hb.as_array_resampled_to_size(lulc_previous_year_path, max_plotting_size)
                                    
                                    p.global_coarse_blocks_list
                                    cr_size1 =[target_zone.split('_')[0], target_zone.split('_')[1], int(p.processing_resolution), int(p.processing_resolution)] # This is the coarse resolution size of the zone to plot.
                                    cr_size = target_coarse_blocks[c] # This is the coarse resolution size of the zone to plot.
                                    
                                    # cr_size = p.global_processing_blocks_list[0] # This is the coarse resolution size of the zone to plot.
                                    change_array = hb.load_geotiff_chunk_by_cr_size(scaled_proportion_to_allocate_path, cr_size)
                                    
                                    a = hb.enumerate_array_as_odict(change_array)
                                    print('a', a)
                                    a
                                    # hb.enumerate_raster_path
                                    # change_array = hb.as_array(scaled_proportion_to_allocate_path)



                                    show_class_expansions_vs_change_underneath(lulc_previous_year_array, lulc_projected_array, class_id, change_array, output_path,
                                                                    title='Class ' + class_label + ' projected expansion and contraction on coarse change')


def coarse_change_with_class_change(passed_p=None):
    # For each class, plot the coarse and fine data

    p = passed_p

    if p.run_this:


        if p.scenario_definitions_path is not None:
            p.scenarios_df = pd.read_csv(p.scenario_definitions_path)
            for index, row in p.scenarios_df.iterrows():
                seals_utils.assign_df_row_to_object_attributes(p, row)
                seals_utils.set_derived_attributes(p)

                if p.scenario_type !=  'baseline':
                    max_plotting_size = 200000

                    # By default, this will select 4 zones from different parts of the list to plot full change matrices. This is slow.
                    # You can override this to plot all here:
                    zones_to_plot = 'random_four' # one of first, all, or four

                    for year_c, year in enumerate(p.years):
                        target_allocation_zones_dir = os.path.join(p.allocations_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year), 'allocation_zones')
                        seals_utils.load_blocks_list(p, target_allocation_zones_dir)

                        if year_c == 0:
                            previous_year = p.key_base_year
                        else:
                            previous_year = p.years[year_c - 1]

                        if zones_to_plot == 'all':
                            target_zones = p.global_processing_blocks_list
                        elif zones_to_plot == 'four':
                            target_zones = [p.global_processing_blocks_list[0], p.global_processing_blocks_list[int(len(p.global_processing_blocks_list)/4)], p.global_processing_blocks_list[int(len(p.global_processing_blocks_list)/2)], p.global_processing_blocks_list[int(len(p.global_processing_blocks_list)*3/4)]]
                        elif zones_to_plot == 'first':
                            target_zones = [p.global_processing_blocks_list[0]]
                        elif zones_to_plot == 'random_four':
                            n_processing_blocks = int(len(p.global_processing_blocks_list))
                            if n_processing_blocks < 4:
                                raise ValueError('There are not enough processing blocks to select four random ones. Please use a different zones_to_plot option.')
                            starting_tile_pos = [
                                        int(n_processing_blocks*(1/8)), #
                                        int(n_processing_blocks*(3/8)),
                                        int(n_processing_blocks*(5/8)),
                                        int(n_processing_blocks*(7/8)),
                                        ]
                            
                            # Iterate through the selected starting positions and check if there is an lulc_file actually existing.                            
                            target_zones = []
                            target_blocks = []
                            target_coarse_blocks = []
                            for target_pos in starting_tile_pos:
                                for i in range(int(n_processing_blocks / 4)):
                                    
                                    target_block = p.global_processing_blocks_list[target_pos + i]
                                    target_coarse_block = p.global_coarse_blocks_list[target_pos + i]
                                    target_zone = str(target_block[0] + '_' + target_block[1])
                                    ha_diff_from_previous_year_dir_to_plot = os.path.join(p.coarse_simplified_ha_difference_from_previous_year_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year))
                                    allocation_dir_to_plot = os.path.join(p.intermediate_dir, 'allocations', p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year), 'allocation_zones', str(target_zone), 'allocation')
                                    lulc_projected_path= os.path.join(allocation_dir_to_plot, 'lulc_' + p.lulc_src_label + '_'  + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(year) + '.tif')
                                    print('lulc_projected_path', lulc_projected_path)
                                    if hb.path_exists(lulc_projected_path):
                                        target_zones.append(target_zone)
                                        target_blocks.append(target_block)
                                        target_coarse_blocks.append(target_coarse_block)
                                        break
                                    if len(target_zones) >= 4:
                                        break                                
                            if len(target_zones) < 4:
                                raise ValueError('NONE OF THE BLOCKS CHECKED have lulc maps present to select four random ones. Please use a different zones_to_plot option.')

                            
                        else:
                            raise ValueError('zones_to_plot must be one of first, all, or four')

                        # # Make sure the target zones are in the right format
                        # for c, row in enumerate(target_zones):
                        #         target_zones[c] = str(row[0] + '_' + row[1])

                        for target_zone in target_zones:
                            ha_diff_from_previous_year_dir_to_plot = os.path.join(p.coarse_simplified_ha_difference_from_previous_year_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year))
                            allocation_dir_to_plot = os.path.join(p.intermediate_dir, 'allocations', p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year), 'allocation_zones', target_zone, 'allocation')
                            lulc_projected_path= os.path.join(allocation_dir_to_plot, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(year) + '.tif')
                            lulc_projected_array = None

                            if previous_year == p.key_base_year:
                                lulc_previous_year_path = os.path.join(allocation_dir_to_plot, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.model_label + '_' + str(previous_year) + '.tif')

                                lulc_previous_year_array = None     # For deffered loading
                            else:
                                previous_allocation_dir_to_plot = allocation_dir_to_plot.replace('\\','/').replace('/' + str(year) + '/', '/' + str(previous_year) + '/')
                                lulc_previous_year_path = os.path.join(previous_allocation_dir_to_plot, 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(previous_year) + '.tif')
                                lulc_previous_year_array = None

                            for class_id, class_label in zip(p.lulc_correspondence_dict['dst_ids'], p.lulc_correspondence_dict['dst_labels']):

                                filename = class_label + '_' + str(year) + '_' + str(previous_year) + '_ha_diff_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '.tif'
                                scaled_proportion_to_allocate_path = os.path.join(allocation_dir_to_plot, filename)
                                output_path = os.path.join(p.cur_dir, str(year) + '_' + target_zone + '_' + class_label + '_projected_expansion_and_contraction.png')

                                if hb.path_exists(scaled_proportion_to_allocate_path) and not hb.path_exists(output_path):
                                    hb.log('Plotting ' + output_path)
                                    if lulc_projected_array is None:
                                        lulc_projected_array = hb.as_array_resampled_to_size(lulc_projected_path, max_plotting_size)


                                    if lulc_previous_year_array is None:
                                        lulc_previous_year_array = hb.as_array_resampled_to_size(lulc_previous_year_path, max_plotting_size)
                                    change_array = hb.as_array(scaled_proportion_to_allocate_path)

                                    show_class_expansions_vs_change(lulc_previous_year_array, lulc_projected_array, class_id, change_array, output_path,
                                                                    title='Class ' + class_label + ' projected expansion and contraction on coarse change')

def target_zones_matrices_pngs(passed_p=None):
    p = passed_p

    # TODOOO: Document how i separates the chnage matrices and change matrices pngs into content/visualization. Then
    # add a a simple LULC plot. This might involve pulling in geoecon code.

    if p.run_this:
        if p.scenario_definitions_path is not None:
            p.scenarios_df = pd.read_csv(p.scenario_definitions_path)

            for index, row in p.scenarios_df.iterrows():
                seals_utils.assign_df_row_to_object_attributes(p, row)
                seals_utils.set_derived_attributes(p)

                classes_that_might_change = p.changing_class_indices
                if p.scenario_type !=  'baseline':
                    for c, year in enumerate(p.years):
                        full_change_matrix_no_diagonal_path = os.path.join(p.full_change_matrices_dir, str(year), 'full_change_matrix_no_diagonal.tif')
                        full_change_matrix_no_diagonal_auto_png_path = os.path.join(p.cur_dir, str(year) + '_full_change_matrix_no_diagonal_auto.png')
                        if not hb.path_exists(full_change_matrix_no_diagonal_auto_png_path) or not hb.path_exists(full_change_matrix_no_diagonal_path):
                            n_classes = len(classes_that_might_change)

                            fig, ax = plt.subplots()
                            fig.set_size_inches(10, 8)

                            # Get the CR_width_height of the zone(s) we want to plut


                            zones_to_plot = 'random_four' # one of first, all, or four
                            if zones_to_plot == 'all':
                                target_zones = p.global_processing_blocks_list
                                offsets = [p.coarse_blocks_list[0]]
                                offsets = [[int(i) for i in j] for j in target_zones]

                            elif zones_to_plot == 'four':
                                target_zones = [p.global_processing_blocks_list[0], p.global_processing_blocks_list[int(len(p.global_processing_blocks_list)/4)], p.global_processing_blocks_list[int(len(p.global_processing_blocks_list)/2)], p.global_processing_blocks_list[int(len(p.global_processing_blocks_list)*3/4)]]
                                offsets = [p.coarse_blocks_list[0]]
                                offsets = [[int(i) for i in j] for j in target_zones]



                            elif zones_to_plot == 'first':
                                target_zones = [p.global_processing_blocks_list[0]]
                                offsets = [p.coarse_blocks_list[0]]
                                offsets = [[int(i) for i in offsets[0]]]
                            else:
                                raise ValueError('zones_to_plot must be one of first, all, or four')


                            # full_change_matrix_no_diagonal = hb.as_array(full_change_matrix_no_diagonal_path)

                            for offset in offsets:
                                full_change_matrix_no_diagonal = hb.load_geotiff_chunk_by_cr_size(full_change_matrix_no_diagonal_path, offset)

                                for year in p.years:

                                    current_lulc_filename = 'change_matrix_' + str(year) + '.tif'
                                    current_change_dir = os.path.join(p.cur_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label)
                                    # title = 'LULC ' + p.exogenous_label + ' ' + p.climate_label + ' ' + p.model_label + ' ' + p.counterfactual_label + ' ' + str(year)
                                    # title = title.title()

                                    hb.create_directories(current_change_dir)

                                    full_change_matrix_no_diagonal_png_path = os.path.join(current_change_dir, current_lulc_filename)

                            if np.sum(full_change_matrix_no_diagonal) > 0:
                                # Plot the heatmap
                                vmin = np.min(full_change_matrix_no_diagonal)
                                vmax = np.max(full_change_matrix_no_diagonal)
                                im = ax.imshow(full_change_matrix_no_diagonal, cmap='YlGnBu', norm=colors.LogNorm(vmin=vmin + 1, vmax=vmax))

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

                                coarse_match_n_rows = hb.get_shape_from_dataset_path(p.aoi_ha_per_cell_coarse_path)[0]
                                coarse_match_n_cols = hb.get_shape_from_dataset_path(p.aoi_ha_per_cell_coarse_path)[1]
                                for i in range(n_classes * (coarse_match_n_rows)):
                                    class_id = i % n_classes
                                    row_labels.append(str(class_id))

                                for i in range(n_classes * (coarse_match_n_cols)):
                                    class_id = i % n_classes
                                    col_labels.append(str(class_id))

                                trans = ax.get_xaxis_transform()  # x in data untis, y in axes fraction

                                for i in range(coarse_match_n_rows):
                                    ann = ax.annotate('Zone ' + str(i + 1), xy=(-3.5, i / coarse_match_n_rows + .5 / coarse_match_n_rows), xycoords=trans)
                                    # ann = ax.annotate('Class ' + str(i + 1), xy=(-2.5, i / p.coarse_match.n_rows + .5 / p.coarse_match.n_rows), xycoords=trans)
                                    ann = ax.annotate('Zone ' + str(i + 1), xy=(i * (coarse_match_n_rows + 1) + .25 * coarse_match_n_rows, 1.05), xycoords=trans)  #
                                    # ann = ax.annotate('MgII', xy=(-2, 1 / (i * n_classes + n_classes / 2)), xycoords=trans)
                                    # plt.annotate('This is awesome!',
                                    #              xy=(-.1, i * n_classes + n_classes / 2),
                                    #              xycoords='data',
                                    #              textcoords='offset points',
                                    #              arrowprops=dict(arrowstyle="->"))

                                # ax.set_xticklabels(col_labels)
                                # ax.set_yticklabels(row_labels)

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



                                # texts = annotate_heatmap(im, valfmt="{x:.1f} t")

                                major_gridline = False
                                for i in range(n_classes * coarse_match_n_rows + 1):
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


                                from hazelbean.visualization import \
                                    full_show_array
                                full_show_array(full_change_matrix_no_diagonal, output_path=full_change_matrix_no_diagonal_auto_png_path, cbar_label='Number of changes from class R to class C per tile', title='Change matrix mosaic',
                                                num_cbar_ticks=2, vmin=0, vmid=vmax / 10.0, vmax=vmax, color_scheme='ylgnbu')

def full_change_matrices_pngs(passed_p=None):
    p = passed_p

    # TODOOO: Document how i separates the chnage matrices and change matrices pngs into content/visualization. Then
    # add a a simple LULC plot. This might involve pulling in geoecon code.

    if p.run_this:
        if p.scenario_definitions_path is not None:
            p.scenarios_df = pd.read_csv(p.scenario_definitions_path)

            for index, row in p.scenarios_df.iterrows():
                seals_utils.assign_df_row_to_object_attributes(p, row)
                seals_utils.set_derived_attributes(p)

                classes_that_might_change = p.changing_class_indices
                if p.scenario_type !=  'baseline':
                    for c, year in enumerate(p.years):
                        full_change_matrix_no_diagonal_path = os.path.join(p.full_change_matrices_dir, str(year), 'full_change_matrix_no_diagonal.tif')
                        full_change_matrix_no_diagonal_auto_png_path = os.path.join(p.cur_dir, str(year) + '_full_change_matrix_no_diagonal_auto.png')
                        if not hb.path_exists(full_change_matrix_no_diagonal_auto_png_path) or not hb.path_exists(full_change_matrix_no_diagonal_path):
                            n_classes = len(classes_that_might_change)

                            fig, ax = plt.subplots()
                            fig.set_size_inches(10, 8)

                            full_change_matrix_no_diagonal = hb.as_array(full_change_matrix_no_diagonal_path)
                            if np.sum(full_change_matrix_no_diagonal) > 0:
                                # Plot the heatmap
                                vmin = np.min(full_change_matrix_no_diagonal)
                                vmax = np.max(full_change_matrix_no_diagonal)
                                im = ax.imshow(full_change_matrix_no_diagonal, cmap='YlGnBu', norm=colors.LogNorm(vmin=vmin + 1, vmax=vmax))

                                # Create colorbar
                                cbar = ax.figure.colorbar(im, ax=ax, shrink=.6)
                                cbar.set_label('Number of cells changed from class ROW to class COL', size=10)
                                # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

                                # We want to show all ticks...
                                # ax.set_xticks(np.arange(full_change_matrix_no_diagonal.shape[1]))
                                # ax.set_xticks(np.arange(full_change_matrix_no_diagonal.shape[1]))
                                ax.set_xticks([])
                                ax.set_yticks([])
                                # ... and label them with the respective list entries.

                                # row_labels = []
                                # col_labels = []

                                coarse_match_n_rows = hb.get_shape_from_dataset_path(p.aoi_ha_per_cell_coarse_path)[0]
                                coarse_match_n_cols = hb.get_shape_from_dataset_path(p.aoi_ha_per_cell_coarse_path)[1]
                                # for i in range(n_classes * (coarse_match_n_rows)):
                                #     class_id = i % n_classes
                                #     # row_labels.append(str(class_id))

                                # for i in range(n_classes * (coarse_match_n_cols)):
                                #     class_id = i % n_classes
                                #     # col_labels.append(str(class_id))

                                # trans = ax.get_xaxis_transform()  # x in data untis, y in axes fraction

                                # plot_zone_labels = False # Must manually build locations
                                # if plot_zone_labels:
                                #     for i in range(coarse_match_n_rows):
                                #         ann = ax.annotate('Zone ' + str(i + 1), xy=(-3.5, i / coarse_match_n_rows + .5 / coarse_match_n_rows), xycoords=trans)
                                #         # ann = ax.annotate('Class ' + str(i + 1), xy=(-2.5, i / p.coarse_match.n_rows + .5 / p.coarse_match.n_rows), xycoords=trans)
                                #     for i in range(coarse_match_n_cols):
                                #         ann = ax.annotate('Zone ' + str(i + 1), xy=(i * (coarse_match_n_cols + 1) + .25 * coarse_match_n_cols, 1.05), xycoords=trans)  #
                                #         # ann = ax.annotate('MgII', xy=(-2, 1 / (i * n_classes + n_classes / 2)), xycoords=trans)
                                #         # plt.annotate('This is awesome!',
                                #         #              xy=(-.1, i * n_classes + n_classes / 2),
                                #         #              xycoords='data',
                                #         #              textcoords='offset points',
                                #         #              arrowprops=dict(arrowstyle="->"))
                                # plot_ticks = 1
                                # if plot_ticks:
                                #     # ax.set_xticklabels(col_labels)
                                #     # ax.set_yticklabels(row_labels)

                                #     # Let the horizontal axes labeling appear on top.
                                #     ax.tick_params(top=False, bottom=False, labeltop=False, labelbottom=False)

                                #     # Rotate the tick labels and set their alignment.
                                #     # plt.setp(ax.get_xticklabels(), rotation=90, ha="center", rotation_mode="anchor")

                                #     # ax.set_xticks(np.arange(full_change_matrix_no_diagonal.shape[1] + 1) - .5, minor=True)
                                #     # ax.set_yticks(np.arange(full_change_matrix_no_diagonal.shape[0] + 1) - .5, minor=True)
                                #     # ax.tick_params(which="minor", bottom=False, left=False)

                                # im, cbar = heatmap(full_change_matrix_no_diagonal, row_labels, col_labels, ax=ax,
                                #                    cmap="YlGn", cbarlabel="harvest [t/year]")
                                # Turn spines off and create white grid.
                                for edge, spine in ax.spines.items():
                                    spine.set_visible(False)

                                ax.grid(which="minor", color="w", linestyle='-', linewidth=1)


                                full_change_matrix_no_diagonal_png_path = os.path.join(p.cur_dir, str(year) + '_full_change_matrix_no_diagonal.png')
                                # texts = annotate_heatmap(im, valfmt="{x:.1f} t")

                                major_gridline = False
                                for i in range(n_classes * coarse_match_n_rows + 1):
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
                                        # ax.axvline(x=xloc, color='grey')
                                        ax.axhline(y=yloc, color='grey')

                                major_gridline = False
                                for i in range(n_classes * coarse_match_n_cols + 1):
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
                                        # ax.axhline(y=yloc, color='grey')


                                # # plt.title('Cross-zone change matrix')
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


                                from hazelbean.visualization import \
                                    full_show_array
                                full_show_array(full_change_matrix_no_diagonal, output_path=full_change_matrix_no_diagonal_auto_png_path, cbar_label='Number of changes from class R to class C per tile', title='Change matrix mosaic',
                                                num_cbar_ticks=2, vmin=0, vmid=vmax / 10.0, vmax=vmax, color_scheme='ylgnbu')


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
        from seals import seals_visualization_functions
        output_path = os.path.join(p.cur_dir, 'gen' + str(generation_id).zfill(6) + '_class_' + str(i) + '_observed_vs_projected.png')
        seals_visualization_functions.show_lulc_class_change_difference(p.baseline_lulc_af.data, p.observed_lulc_af.data, p.projected_lulc_af.data, i, difference_metric,
                                          change_array, annotation_text, output_path)

        output_path = os.path.join(p.cur_dir, 'gen' + str(generation_id).zfill(6) + '_class_' + str(i) + '_projected_expansion_and_contraction.png')
        seals_visualization_functions.show_class_expansions_vs_change(p.baseline_lulc_af.data, p.projected_lulc_af.data, i, change_array, output_path, title='Class ' + str(i) + ' projected expansion and contraction on coarse change')

    output_path = os.path.join(p.cur_dir, 'gen' + str(generation_id).zfill(6) + '_lulc_comparison_and_scores.png')
    seals_visualization_functions.show_overall_lulc_fit(p.baseline_lulc_af.data, p.observed_lulc_af.data, p.projected_lulc_af.data, p.overall_similarity_plot_af.data, output_path, title='Overall LULC and fit')




def lulc_pngs(p):
    # Simple plot of the PNGs.

    if p.run_this:
        if p.scenario_definitions_path is not None:
            p.scenarios_df = pd.read_csv(p.scenario_definitions_path)


            for index, row in p.scenarios_df.iterrows():
                seals_utils.assign_df_row_to_object_attributes(p, row)
                seals_utils.set_derived_attributes(p)

                # Build a dict for the lulc labels
                labels_dict = dict(zip(p.all_class_indices, p.all_class_labels))

                # this acountcs for the fact that the way the correspondence is loaded is not necessarily in the numerical order
                indices_to_labels_dict = dict(sorted(labels_dict.items()))

                for year in p.years:
                    if p.scenario_type ==  'baseline':
                        current_lulc_filename = 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.model_label + '_' + str(p.key_base_year) + '.tif'
                        title = 'LULC ' + p.exogenous_label + ' ' + p.model_label + ' ' + str(p.key_base_year)
                        title = title.title()
                    else:
                        current_lulc_filename = 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(year) + '.tif'
                        title = 'LULC ' + p.exogenous_label + ' ' + p.climate_label + ' ' + p.model_label + ' ' + p.counterfactual_label + ' ' + str(year)
                        title = title.title()


                    max_plotting_size = 10000000

                    current_lulc_path = os.path.join(p.stitched_lulc_simplified_scenarios_dir, current_lulc_filename)
                    current_output_path = os.path.join(p.cur_dir, current_lulc_filename.replace('.tif', '.png'))
                    if not hb.path_exists(current_output_path):
                        if hb.path_exists(current_lulc_path):
                            current_lulc_array = hb.as_array_resampled_to_size(current_lulc_path, max_plotting_size)

                            plot_array_as_seals7_lulc(current_lulc_array, output_path=current_output_path, title=title, indices_to_labels_dict=indices_to_labels_dict)



def coarse_fine_with_report(passed_p=None):
    p = passed_p

    if p.run_this:


        if p.scenario_definitions_path is not None:
            p.scenarios_df = pd.read_csv(p.scenario_definitions_path)
            for index, row in p.scenarios_df.iterrows():
                seals_utils.assign_df_row_to_object_attributes(p, row)
                seals_utils.set_derived_attributes(p)

                if p.scenario_type !=  'baseline':
                    max_plotting_size = 200000

                    # By default, this will select 4 zones from different parts of the list to plot full change matrices. This is slow.
                    # You can override this to plot all here:
                    zones_to_plot = 'random_four' # one of first, all, or four

                    for year_c, year in enumerate(p.years):
                        target_allocation_zones_dir = os.path.join(p.allocations_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year), 'allocation_zones')
                        seals_utils.load_blocks_list(p, target_allocation_zones_dir)

                        if year_c == 0:
                            previous_year = p.key_base_year
                        else:
                            previous_year = p.years[year_c - 1]

                        # if p.plotting_level >= 0:
                        #     zones_to_plot = 'all'
                        # elif p.plotting_level >= 30:
                        #     zones_to_plot = 'four'
                        # elif p.plotting_level >= 20:
                        #     zones_to_plot = 'first'
                        # else:
                        #     zones_to_plot = 'none'

                        if zones_to_plot == 'all':
                            target_zones = p.global_processing_blocks_list
                        elif zones_to_plot == 'four':
                            target_zones = [p.global_processing_blocks_list[0], p.global_processing_blocks_list[int(len(p.global_processing_blocks_list)/4)], p.global_processing_blocks_list[int(len(p.global_processing_blocks_list)/2)], p.global_processing_blocks_list[int(len(p.global_processing_blocks_list)*3/4)]]
                        elif zones_to_plot == 'first':
                            target_zones = [p.global_processing_blocks_list[0]]
                        elif zones_to_plot == 'random_four':
                            n_processing_blocks = int(len(p.global_processing_blocks_list))
                            if n_processing_blocks < 4:
                                raise ValueError('There are not enough processing blocks to select four random ones. Please use a different zones_to_plot option.')
                            starting_tile_pos = [
                                        int(n_processing_blocks*(1/8)), #
                                        int(n_processing_blocks*(3/8)),
                                        int(n_processing_blocks*(5/8)),
                                        int(n_processing_blocks*(7/8)),
                                        ]
                            
                            # Iterate through the selected starting positions and check if there is an lulc_file actually existing.                            
                            target_zones = []
                            target_blocks = []
                            target_coarse_blocks = []
                            target_fine_blocks = []
                            for target_pos in starting_tile_pos:
                                for i in range(int(n_processing_blocks / 4)):
                                    
                                    target_block = p.global_processing_blocks_list[target_pos + i]
                                    target_coarse_block = p.global_coarse_blocks_list[target_pos + i]
                                    target_fine_block = p.global_fine_blocks_list[target_pos + i]
                                    target_zone = str(target_block[0] + '_' + target_block[1])
                                    ha_diff_from_previous_year_dir_to_plot = os.path.join(p.coarse_simplified_ha_difference_from_previous_year_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year))
                                    allocation_dir_to_plot = os.path.join(p.intermediate_dir, 'allocations', p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year), 'allocation_zones', str(target_zone), 'allocation')
                                    lulc_projected_path= os.path.join(allocation_dir_to_plot, 'lulc_' + p.lulc_src_label + '_'  + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(year) + '.tif')
                                    print('lulc_projected_path', lulc_projected_path)
                                    if hb.path_exists(lulc_projected_path):
                                        target_zones.append(target_zone)
                                        target_blocks.append(target_block)
                                        target_coarse_blocks.append(target_coarse_block)
                                        target_fine_blocks.append(target_fine_block)
                                        break
                                    if len(target_zones) >= 4:
                                        break                                
                            if len(target_zones) < 4:
                                raise ValueError('NONE OF THE BLOCKS CHECKED have lulc maps present to select four random ones. Please use a different zones_to_plot option.')

                        else:
                            target_zones = []
                            hb.debug('No zones to plot.')

                        # Make sure the target zones are in the right format
                        # for c, row in enumerate(target_zones):
                        #         target_zones[c] = str(row[0] + '_' + row[1])

                        for c, target_zone in enumerate(target_zones):
                            allocation_dir_to_plot = os.path.join(p.intermediate_dir, 'allocations', p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year), 'allocation_zones', target_zone, 'allocation')
                            lulc_projected_path= os.path.join(allocation_dir_to_plot, 'lulc_' + p.lulc_src_label + '_'  + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(year) + '.tif')
                            lulc_projected_array = None
                            ha_per_cell_coarse_path = os.path.join(allocation_dir_to_plot, 'block_ha_per_cell_coarse.tif')
                            ha_per_cell_fine_path = os.path.join(allocation_dir_to_plot, 'block_ha_per_cell_fine.tif')


                            # lulc_baseline_path = os.path.join(p.cur_dir, 'lulc_' + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.model_label + '_' + str(p.year) + '.tif')
                            # lulc_baseline_path = os.path.join(p.cur_dir, 'lulc_' + p.lulc_simplification_label + '_baseline_' + p.model_label + '_' + str(p.year) + '.tif')

                            if previous_year == p.key_base_year:
                                lulc_previous_year_path = os.path.join(allocation_dir_to_plot, 'lulc_' + p.lulc_src_label + '_'  + p.lulc_simplification_label + '_' + p.model_label + '_' + str(previous_year) + '.tif')

                                lulc_previous_year_array = None     # For deffered loading
                            else:
                                previous_allocation_dir_to_plot = allocation_dir_to_plot.replace('\\','/').replace('/' + str(year) + '/', '/' + str(previous_year) + '/')
                                lulc_previous_year_path = os.path.join(previous_allocation_dir_to_plot, 'lulc_' + p.lulc_src_label + '_'  + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(previous_year) + '.tif')
                                lulc_previous_year_array = None

                            if p.plotting_level >= 0:
                                do_class_specific_plots = True
                            else:
                                do_class_specific_plots = False

                            if do_class_specific_plots:
                                for class_id, class_label in zip(p.changing_class_indices, p.changing_class_labels):

                                    filename = class_label + '_' + str(year) + '_' + str(previous_year) + '_ha_diff_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '.tif'
                                    scaled_proportion_to_allocate_path = os.path.join(allocation_dir_to_plot, filename)
                                    output_path = os.path.join(p.cur_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year), str(target_zone), class_label + '_projected_expansion_and_contraction.png')

                                    if hb.path_exists(scaled_proportion_to_allocate_path) and not hb.path_exists(output_path):
                                        hb.log('Plotting ' + output_path)
                                        hb.create_directories(output_path)
                                        if lulc_projected_array is None:
                                            lulc_projected_array = hb.as_array_resampled_to_size(lulc_projected_path, max_plotting_size)


                                        if lulc_previous_year_array is None:
                                            lulc_previous_year_array = hb.as_array_resampled_to_size(lulc_previous_year_path, max_plotting_size)
                                        change_array = hb.as_array(scaled_proportion_to_allocate_path)

                                        ha_per_cell_coarse_path = os.path.join(allocation_dir_to_plot, 'block_ha_per_cell_coarse.tif')
                                        ha_per_cell_fine_path = os.path.join(allocation_dir_to_plot, 'block_ha_per_cell_fine.tif')

                                        ha_per_cell_coarse_array = hb.as_array(ha_per_cell_coarse_path)
                                        ha_per_cell_fine_array = hb.as_array(ha_per_cell_fine_path)

                                        show_specific_class_expansions_vs_change_with_numeric_report_and_validation(lulc_previous_year_array, lulc_projected_array, class_id, class_label, change_array, ha_per_cell_coarse_array,
                                                                                            ha_per_cell_fine_array, allocation_dir_to_plot, output_path,
                                                                                            title='Class ' + class_label + ' projected expansion and contraction on coarse change')

                            if p.plotting_level >= 0:
                                do_all_class_plots = True
                            else:
                                do_all_class_plots = False

                            if do_all_class_plots:
                                change_array_paths = []
                                for class_id, class_label in zip(p.changing_class_indices, p.changing_class_labels):
                                    filename = class_label + '_' + str(year) + '_' + str(previous_year) + '_ha_diff_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '.tif'
                                    scaled_proportion_to_allocate_path = os.path.join(allocation_dir_to_plot, filename)
                                    change_array_paths.append(scaled_proportion_to_allocate_path)

                                if lulc_projected_array is None:
                                    lulc_projected_array = hb.as_array_resampled_to_size(lulc_projected_path, max_plotting_size)


                                if lulc_previous_year_array is None:
                                    lulc_previous_year_array = hb.as_array_resampled_to_size(lulc_previous_year_path, max_plotting_size)
                                # change_array = hb.as_array(scaled_proportion_to_allocate_path)

                                output_path = os.path.join(p.cur_dir, p.exogenous_label, p.climate_label, p.model_label, p.counterfactual_label, str(year), str(target_zone), 'all_classes_projected_expansion_and_contraction.png')
                                if not hb.path_exists(output_path):
                                    hb.create_directories(output_path)

                                    ha_per_cell_coarse_array = hb.as_array(ha_per_cell_coarse_path)
                                    
                                    if not hb.path_exists(ha_per_cell_fine_path):
                                        target_coarse_blocks
                                        hectares_per_grid_cell = hb.load_geotiff_chunk_by_cr_size(p.aoi_ha_per_cell_fine_path, target_fine_blocks[c], output_path=ha_per_cell_fine_path).astype(np.float64)
                                    
                                    ha_per_cell_fine_array = hb.as_array(ha_per_cell_fine_path)

                                    show_all_class_expansions_vs_change_with_numeric_report_and_validation(lulc_previous_year_array, lulc_projected_array, p.changing_class_indices,
                                                                                                    p.changing_class_labels, change_array_paths, ha_per_cell_coarse_array,
                                                                                                    ha_per_cell_fine_array, allocation_dir_to_plot, output_path,
                                                                                                    title='Projected expansion and contraction on coarse change')



## HYBRID FUNCTION
def simpler_plot_generation(p):
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

# Temoporarily disable HTML report generation until we rebuild the release system to include contextily
# def html_report(p):
#     # Generate a HTML report with the generated PNGs.
#     if p.run_this:
#         if p.scenario_definitions_path is not None:
#             p.scenarios_df = pd.read_csv(p.scenario_definitions_path)

#             report_assets_dir = os.path.join(p.cur_dir, 'assets')
#             os.makedirs(report_assets_dir, exist_ok=True)
#             report_output_path = os.path.join(p.cur_dir, 'seals_visualization_report.html')

#             # Add summary table
#             scenarios_table_rows = []
#             for index, row in p.scenarios_df.iterrows():
#                 seals_utils.assign_df_row_to_object_attributes(p, row)
#                 seals_utils.set_derived_attributes(p)
#                 aoi_val = getattr(p, 'aoi', '') or getattr(p, 'aoi_path', '')
#                 years_val = ','.join([str(y) for y in getattr(p, 'years', [])]) if getattr(p, 'years', None) is not None else ''
#                 scenarios_table_rows.append({
#                     'index': index,
#                     'scenario_label': getattr(p, 'scenario_label', '') or f"{getattr(p,'exogenous_label','')}_{getattr(p,'climate_label','')}_{getattr(p,'model_label','')}_{getattr(p,'counterfactual_label','')}",
#                     'scenario_type': getattr(p, 'scenario_type', ''),
#                     'aoi': aoi_val,
#                     'exogenous_label': getattr(p, 'exogenous_label', ''),
#                     'climate_label': getattr(p, 'climate_label', ''),
#                     'model_label': getattr(p, 'model_label', ''),
#                     'counterfactual_label': getattr(p, 'counterfactual_label', ''),
#                     'years': years_val
#                 })

#             # render table HTML (simple, small)
#             scenarios_table_html = ['<div class="description"><h3>Scenarios</h3><table style="width:100%;border-collapse:collapse;">']
#             # header
#             headers = ['#', 'scenario_label', 'scenario_type', 'aoi', 'exogenous', 'climate', 'model', 'counterfactual', 'years']
#             ths = ''.join([f'<th style="text-align:left;padding:6px;border-bottom:1px solid #ddd;">{_esc(h)}</th>' for h in headers])
#             scenarios_table_html.append(f'<tr>{ths}</tr>')
#             # rows
#             for r in scenarios_table_rows:
#                 tds = ''.join([
#                     f'<td style="padding:6px;border-bottom:1px solid #f0f0f0;">{_esc(str(r.get(k, "")))}</td>'
#                     for k in ['index', 'scenario_label', 'scenario_type', 'aoi', 'exogenous_label', 'climate_label', 'model_label', 'counterfactual_label', 'years']
#                 ])
#                 scenarios_table_html.append(f'<tr>{tds}</tr>')
#             scenarios_table_html.append('</table></div>')
#             scenarios_table_html = '\n'.join(scenarios_table_html)

#             # === AOI SECTION ===
#             fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#             for ax in axes.flat:
#                 ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
#                                labelbottom=False, labelleft=False)
#             # Plot 1: Vector AOI
#             aoi = gpd.read_file(p.aoi_path)
#             aoi.plot(ax=axes[0], facecolor='lightblue', edgecolor='black', linewidth=0.5, alpha=0.5)
#             axes[0].set_title('AOI Boundary')
#             # Zoom to AOI bounds before adding basemap
#             minx, miny, maxx, maxy = p.bb
#             axes[0].set_xlim(minx, maxx)
#             axes[0].set_ylim(miny, maxy)
#             ctx.add_basemap(axes[0], source=ctx.providers.OpenStreetMap.Mapnik, crs='EPSG:4326', alpha=0.7, attribution=False)
#             # Plot 2: Fine resolution raster
#             with rasterio.open(p.aoi_ha_per_cell_fine_path) as src:
#                 show(src, ax=axes[1], cmap='viridis')
#                 axes[1].set_title('Fine Resolution (ha per cell)')
#             # Plot 3: Coarse resolution raster
#             with rasterio.open(p.aoi_ha_per_cell_coarse_path) as src:
#                 show(src, ax=axes[2], cmap='viridis')
#                 axes[2].set_title('Coarse Resolution (ha per cell)')
#             plt.tight_layout()
#             plt.savefig(os.path.join(report_assets_dir, 'aoi_plots.png'))
#             plt.close()
            
#             # === FINE PROCESSED INPUTS SECTION ===
#             # LULC colors (including 0 = nodata at top)
#             lulc_colors = [
#                 (1, 1, 1),                 # 0 = nodata
#                 (212/255,106/255,110/255), # 1 urban
#                 (227/255,167/255,92/255),  # 2 cropland
#                 (232/255,232/255,106/255), # 3 grassland
#                 (79/255,169/255,90/255),   # 4 forest
#                 (159/255,218/255,143/255), # 5 othernat
#                 (144/255,174/255,224/255), # 6 water
#                 (209/255,209/255,209/255), # 7 other
#             ]
#             lulc_cmap = colors.ListedColormap(lulc_colors)
            
#             # Forces clean discrete boundaries
#             bounds = [i - 0.5 for i in range(9)]  # -0.5 → 7.5
#             lulc_norm = colors.BoundaryNorm(bounds, lulc_cmap.N)
            
#             fig, axes = plt.subplots(2, 2, figsize=(12, 10))
#             for ax in axes.flat:
#                 ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
#                                labelbottom=False, labelleft=False)
            
#             # Plot 1: Simplified LULC
#             lulc_path = os.path.join(p.fine_processed_inputs_dir, 'lulc', p.lulc_src_label, p.lulc_simplification_label, f'lulc_{p.lulc_src_label}_{p.lulc_simplification_label}_{p.base_years[0]}.tif')
#             with rasterio.open(lulc_path) as src:
#                 show(
#                     src,
#                     ax=axes[0,0],
#                     cmap=lulc_cmap,
#                     norm=lulc_norm
#                 )
#                 axes[0,0].set_title('SEALS7 Classification (Base Year)')
#                 legend_labels = [
#                     'urban',
#                     'cropland',
#                     'grassland',
#                     'forest',
#                     'othernat',
#                     'water',
#                     'other'
#                 ]
#                 # 1–7 only
#                 patches = [
#                     plt.Rectangle((0,0), 1, 1, fc=lulc_colors[i])
#                     for i in range(1, 8)
#                 ]
#                 axes[0,0].legend(
#                     patches,
#                     legend_labels,
#                     loc='lower left',
#                     fontsize=8,
#                     title="LULC Classes"
#                 )
            
#             # Plot 2: Binary cropland
#             binary_path = os.path.join(p.fine_processed_inputs_dir, 'lulc', p.lulc_src_label, p.lulc_simplification_label, 'binaries', str(p.base_years[0]), f'binary_{p.lulc_src_label}_{p.lulc_simplification_label}_{p.base_years[0]}_cropland.tif')
#             with rasterio.open(binary_path) as src:
#                 show(src, ax=axes[0,1], cmap='YlGn')
#                 axes[0,1].set_title('Binary Cropland Mask')
            
#             # Plot 3: Gaussian 1 convolution
#             conv1_path = os.path.join(p.fine_processed_inputs_dir, 'lulc', p.lulc_src_label, p.lulc_simplification_label, 'convolutions', str(p.base_years[0]), f'convolution_{p.lulc_src_label}_{p.lulc_simplification_label}_{p.base_years[0]}_cropland_gaussian_1.tif')
#             with rasterio.open(conv1_path) as src:
#                 show(src, ax=axes[1,0], cmap='YlGn')
#                 axes[1,0].set_title('Cropland Convolution (Gaussian σ=1)')
            
#             # Plot 4: Gaussian 5 convolution
#             conv5_path = os.path.join(p.fine_processed_inputs_dir, 'lulc', p.lulc_src_label, p.lulc_simplification_label, 'convolutions', str(p.base_years[0]), f'convolution_{p.lulc_src_label}_{p.lulc_simplification_label}_{p.base_years[0]}_cropland_gaussian_5.tif')
#             with rasterio.open(conv5_path) as src:
#                 show(src, ax=axes[1,1], cmap='YlGn')
#                 axes[1,1].set_title('Cropland Convolution (Gaussian σ=5)')
            
#             plt.tight_layout()
#             plt.savefig(os.path.join(report_assets_dir, 'fine_processed_inputs_plots.png'))
#             plt.close()

#             # Plot region IDs (once, before scenarios)
#             region_ids_path = os.path.join(p.regional_change_dir, 'region_ids.tif')
#             if os.path.exists(region_ids_path):
#                 fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                
#                 with rasterio.open(region_ids_path) as src:
#                     data = src.read(1)
#                     # Mask out nodata values
#                     if src.nodata is not None:
#                         data_masked = np.ma.masked_equal(data, src.nodata)
#                     else:
#                         data_masked = np.ma.masked_equal(data, 0)
                    
#                     # Get unique region IDs (excluding masked values)
#                     unique_ids = np.unique(data_masked.compressed())
#                     n_regions = len(unique_ids)
                    
#                     # Create discrete colormap for actual regions
#                     cmap = plt.cm.get_cmap('tab20', n_regions)
                    
#                     # Get extent for basemap
#                     bounds = src.bounds
#                     extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
                    
#                     # Add basemap first
#                     ax.set_xlim(bounds.left, bounds.right)
#                     ax.set_ylim(bounds.bottom, bounds.top)
#                     ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, 
#                                    crs='EPSG:4326', alpha=0.7, attribution=False)
                    
#                     # Plot region IDs on top with transparency
#                     im = ax.imshow(data_masked, cmap=cmap, interpolation='nearest',
#                                   extent=extent, alpha=0.5, vmin=unique_ids.min(), 
#                                   vmax=unique_ids.max())
                    
#                     ax.set_title('Region IDs', fontsize=12)
#                     ax.axis('off')
                    
#                     # Create categorical legend 
#                     colors_list = [cmap(i) for i in range(n_regions)]
#                     patches = [Patch(facecolor=colors_list[i % len(colors_list)], edgecolor='k', label=str(int(uid)))
#                                for i, uid in enumerate(unique_ids)]
#                     ax.legend(handles=patches, title='Region ID', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                
#                 plt.tight_layout()
#                 plt.savefig(os.path.join(report_assets_dir, 'region_ids.png'), dpi=150, bbox_inches='tight')
#                 plt.close()
#                 region_ids_html = """
#     <h3>Region IDs</h3>
#     <div class="image-container">
#         <img src="assets/region_ids.png" alt="Region IDs for regional analysis">
#     </div>
# """
#             else:
#                 region_ids_html = ""

#             # Prepare collections for coarse and regional sections
#             coarse_scenario_tabs = []
#             regional_scenario_tabs = []
#             stitched_scenario_tabs = []
#             scenario_list = []
            
#             # Define LULC classes to plot
#             lulc_classes = ['cropland', 'urban', 'grassland', 'forest', 'othernat']
            
#             # Loop through scenarios to generate plots
#             for index, row in p.scenarios_df.iterrows():
#                 seals_utils.assign_df_row_to_object_attributes(p, row)
#                 seals_utils.set_derived_attributes(p)

#                 if p.scenario_type == 'baseline':
#                     continue  # Skip baseline for this report
                
#                 scenario_name = f"{p.exogenous_label}_{p.climate_label}_{p.model_label}_{p.counterfactual_label}"
#                 scenario_id = f"scenario_{index}"
#                 scenario_list.append((scenario_id, scenario_name))
                
#                 # === COARSE CHANGE SECTION ===
#                 coarse_content = ""
                
#                 for yi, year in enumerate(p.years):
#                     # for the first year in p.years use the configured base year (p.base_years[0])
#                     # otherwise use the previous entry in p.years
#                     if yi == 0:
#                         prev_year = p.base_years[0]
#                     else:
#                         prev_year = p.years[yi - 1]
                    
#                     # Create figure for this year's coarse changes
#                     n_classes = len(lulc_classes)
#                     fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4))
#                     fig.suptitle(f'Coarse Change: {year} (from {prev_year})', fontsize=14)

#                     for idx, lulc_class in enumerate(lulc_classes):
#                         # Build path to coarse change file
#                         change_fname = f'{lulc_class}_{year}_{prev_year}_ha_diff_{p.exogenous_label}_{p.climate_label}_{p.model_label}_{p.counterfactual_label}.tif'
#                         change_path = os.path.join(
#                             p.coarse_change_dir, 
#                             'coarse_simplified_ha_difference_from_previous_year',
#                             p.exogenous_label, 
#                             p.climate_label, 
#                             p.model_label, 
#                             p.counterfactual_label, 
#                             str(year),
#                             change_fname
#                         )
                        
#                         # Plot if file exists
#                         if os.path.exists(change_path):
#                             with rasterio.open(change_path) as src:
#                                 data = src.read(1)
#                                 # Use diverging colormap for change (red=loss, green=gain)
#                                 vmax = np.nanmax(np.abs(data))
#                                 im = axes[idx].imshow(data, cmap='RdYlGn', vmin=-vmax, vmax=vmax)
#                                 axes[idx].set_title(f'{lulc_class.capitalize()}', fontsize=10)
#                                 axes[idx].axis('off')
#                                 plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04, label='ha change')
#                         else:
#                             axes[idx].text(0.5, 0.5, f'{lulc_class}\nNot Found', 
#                                          ha='center', va='center', transform=axes[idx].transAxes)
#                             axes[idx].axis('off')
                    
#                     plt.tight_layout()
                    
#                     # Save figure
#                     output_filename = f'coarse_change_{scenario_name}_{year}.png'
#                     plt.savefig(os.path.join(report_assets_dir, output_filename), dpi=150, bbox_inches='tight')
#                     plt.close()
                    
#                     # Add to coarse content
#                     coarse_content += f"""
#         <h4>Year {year}</h4>
#         <div class="image-container">
#             <img src="assets/{output_filename}" alt="Coarse change for {year}">
#         </div>
# """
                
#                 coarse_scenario_tabs.append((scenario_id, coarse_content))
                
#                 # === REGIONAL CHANGE SECTION ===
#                 # Region ID plotted earlier
#                 regional_content = ""
                
#                 # Check if regional change exists for this scenario
#                 has_regional_change = False
#                 test_year = p.years[0]
#                 test_class = lulc_classes[0]
#                 test_path = os.path.join(
#                     p.regional_change_dir,
#                     p.exogenous_label, 
#                     p.climate_label, 
#                     p.model_label, 
#                     p.counterfactual_label,
#                     str(test_year),
#                     f'{test_class}_{test_year}_{p.base_years[0]}_ha_diff_{p.exogenous_label}_{p.climate_label}_{p.model_label}_{p.counterfactual_label}_regional_coarsified.tif'
#                 )
#                 has_regional_change = os.path.exists(test_path)
                
#                 if not has_regional_change:
#                     regional_content = """
#         <p style="color: #888; font-style: italic; padding: 20px;">No regional change data available for this scenario.</p>
# """
#                 else:
#                     for yi, year in enumerate(p.years):
#                         if yi == 0:
#                             prev_year = p.base_years[0]
#                         else:
#                             prev_year = p.years[yi - 1]
                        
#                         # Create 5-panel figure for regional coarsified
#                         n_classes = len(lulc_classes)
#                         fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4))
#                         fig.suptitle(f'Regional Coarsified: {year} (from {prev_year})', fontsize=14)
                        
#                         for idx, lulc_class in enumerate(lulc_classes):
#                             regional_coarsified_path = os.path.join(
#                                 p.regional_change_dir,
#                                 p.exogenous_label, 
#                                 p.climate_label, 
#                                 p.model_label, 
#                                 p.counterfactual_label,
#                                 str(year),
#                                 f'{lulc_class}_{year}_{prev_year}_ha_diff_{p.exogenous_label}_{p.climate_label}_{p.model_label}_{p.counterfactual_label}_regional_coarsified.tif'
#                             )
                            
#                             if os.path.exists(regional_coarsified_path):
#                                 with rasterio.open(regional_coarsified_path) as src:
#                                     data = src.read(1)
#                                     vmax = np.nanmax(np.abs(data))
#                                     im = axes[idx].imshow(data, cmap='RdYlGn', vmin=-vmax, vmax=vmax)
#                                     axes[idx].set_title(f'{lulc_class.capitalize()}', fontsize=10)
#                                     axes[idx].axis('off')
#                                     plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04, label='ha change')
#                             else:
#                                 axes[idx].text(0.5, 0.5, f'{lulc_class}\nNot Found', 
#                                              ha='center', va='center', transform=axes[idx].transAxes)
#                                 axes[idx].axis('off')
                        
#                         plt.tight_layout()
#                         output_filename = f'regional_coarsified_{scenario_name}_{year}.png'
#                         plt.savefig(os.path.join(report_assets_dir, output_filename), dpi=150, bbox_inches='tight')
#                         plt.close()
                        
#                         regional_content += f"""
#         <h4>Year {year} - Regional Coarsified</h4>
#         <div class="image-container">
#             <img src="assets/{output_filename}" alt="Regional coarsified for {year}">
#         </div>
# """
                        
#                         # Create 5-panel figure for covariate sum shift
#                         fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4))
#                         fig.suptitle(f'Covariate Sum Shift: {year} (from {prev_year})', fontsize=14)
                        
#                         for idx, lulc_class in enumerate(lulc_classes):
#                             covariate_shift_path = os.path.join(
#                                 p.regional_change_dir,
#                                 p.exogenous_label, 
#                                 p.climate_label, 
#                                 p.model_label, 
#                                 p.counterfactual_label,
#                                 str(year),
#                                 f'{lulc_class}_{year}_{prev_year}_ha_diff_{p.exogenous_label}_{p.climate_label}_{p.model_label}_{p.counterfactual_label}_covariate_sum_shift.tif'
#                             )
                            
#                             if os.path.exists(covariate_shift_path):
#                                 with rasterio.open(covariate_shift_path) as src:
#                                     data = src.read(1)
#                                     vmax = np.nanmax(np.abs(data))
#                                     im = axes[idx].imshow(data, cmap='RdYlGn', vmin=-vmax, vmax=vmax)
#                                     axes[idx].set_title(f'{lulc_class.capitalize()}', fontsize=10)
#                                     axes[idx].axis('off')
#                                     plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04, label='ha change')
#                             else:
#                                 axes[idx].text(0.5, 0.5, f'{lulc_class}\nNot Found', 
#                                              ha='center', va='center', transform=axes[idx].transAxes)
#                                 axes[idx].axis('off')
                        
#                         plt.tight_layout()
#                         output_filename = f'covariate_shift_{scenario_name}_{year}.png'
#                         plt.savefig(os.path.join(report_assets_dir, output_filename), dpi=150, bbox_inches='tight')
#                         plt.close()
                        
#                         regional_content += f"""
#         <h4>Year {year} - Covariate Sum Shift</h4>
#         <div class="image-container">
#             <img src="assets/{output_filename}" alt="Covariate sum shift for {year}">
#         </div>
# """
                
#                 regional_scenario_tabs.append((scenario_id, regional_content))
                
#                 # === STITCHED LULC SECTION ===
#                 stitched_content = ""
                
#                 # Use the same LULC colormap as before
#                 lulc_colors_stitched = [
#                     (1, 1, 1),                 # 0 = nodata
#                     (212/255,106/255,110/255), # 1 urban
#                     (227/255,167/255,92/255),  # 2 cropland
#                     (232/255,232/255,106/255), # 3 grassland
#                     (79/255,169/255,90/255),   # 4 forest
#                     (159/255,218/255,143/255), # 5 othernat
#                     (144/255,174/255,224/255), # 6 water
#                     (209/255,209/255,209/255), # 7 other
#                 ]
#                 lulc_cmap_stitched = colors.ListedColormap(lulc_colors_stitched)
#                 bounds_stitched = [i - 0.5 for i in range(9)]
#                 lulc_norm_stitched = colors.BoundaryNorm(bounds_stitched, lulc_cmap_stitched.N)
                
#                 for year in p.years:
#                     # Create 2-panel figure (full and clipped)
#                     fig, axes = plt.subplots(1, 2, figsize=(16, 7))
                    
#                     # Full stitched LULC
#                     stitched_path = os.path.join(
#                         p.stitched_lulc_simplified_scenarios_dir,
#                         f'lulc_{p.lulc_src_label}_{p.lulc_simplification_label}_{p.exogenous_label}_{p.climate_label}_{p.model_label}_{p.counterfactual_label}_{year}.tif'
#                     )
                    
#                     if os.path.exists(stitched_path):
#                         with rasterio.open(stitched_path) as src:
#                             show(src, ax=axes[0], cmap=lulc_cmap_stitched, norm=lulc_norm_stitched)
#                             axes[0].set_title(f'Stitched LULC {year}', fontsize=12)
#                             axes[0].axis('off')
#                     else:
#                         axes[0].text(0.5, 0.5, f'Stitched LULC\n{year}\nNot Found', 
#                                    ha='center', va='center', transform=axes[0].transAxes)
#                         axes[0].axis('off')
                    
#                     # Clipped stitched LULC with basemap
#                     clipped_path = os.path.join(
#                         p.stitched_lulc_simplified_scenarios_dir,
#                         f'lulc_{p.lulc_src_label}_{p.lulc_simplification_label}_{p.exogenous_label}_{p.climate_label}_{p.model_label}_{p.counterfactual_label}_{year}_clipped.tif'
#                     )
                    
#                     if os.path.exists(clipped_path):
#                         with rasterio.open(clipped_path) as src:
#                             # Get extent for basemap
#                             bounds = src.bounds
#                             extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
                            
#                             # Add basemap first
#                             axes[1].set_xlim(bounds.left, bounds.right)
#                             axes[1].set_ylim(bounds.bottom, bounds.top)
#                             ctx.add_basemap(axes[1], source=ctx.providers.OpenStreetMap.Mapnik, 
#                                           crs='EPSG:4326', alpha=0.7, attribution=False)
                            
#                             # Read and plot LULC data with transparency
#                             data = src.read(1)
#                             # Mask nodata
#                             if src.nodata is not None:
#                                 data_masked = np.ma.masked_equal(data, src.nodata)
#                             else:
#                                 data_masked = np.ma.masked_equal(data, 0)
                            
#                             axes[1].imshow(data_masked, cmap=lulc_cmap_stitched, norm=lulc_norm_stitched,
#                                          extent=extent, alpha=0.7, interpolation='nearest')
#                             axes[1].set_title(f'Clipped LULC {year}', fontsize=12)
#                             axes[1].axis('off')
#                     else:
#                         axes[1].text(0.5, 0.5, f'Clipped LULC\n{year}\nNot Found', 
#                                    ha='center', va='center', transform=axes[1].transAxes)
#                         axes[1].axis('off')
                    
#                     # Add legend to the clipped plot
#                     legend_labels = ['urban', 'cropland', 'grassland', 'forest', 'othernat', 'water', 'other']
#                     patches = [plt.Rectangle((0,0), 1, 1, fc=lulc_colors_stitched[i]) for i in range(1, 8)]
#                     axes[1].legend(patches, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, title="LULC Classes")
                    
#                     plt.tight_layout()
                    
#                     # Save figure
#                     output_filename = f'stitched_lulc_{scenario_name}_{year}.png'
#                     plt.savefig(os.path.join(report_assets_dir, output_filename), dpi=150, bbox_inches='tight')
#                     plt.close()
                    
#                     # Add to stitched content
#                     stitched_content += f"""
#         <h4>Year {year}</h4>
#         <div class="image-container">
#             <img src="assets/{output_filename}" alt="Stitched LULC for {year}">
#         </div>
# """
                
#                 stitched_scenario_tabs.append((scenario_id, stitched_content))

#             # Build tab buttons and content for coarse section
#             coarse_tabs_buttons = ""
#             coarse_tabs_content = ""
#             for i, (scenario_id, scenario_name) in enumerate(scenario_list):
#                 active_class = "active" if i == 0 else ""
#                 coarse_tabs_buttons += f'<button class="tab-button {active_class}" onclick="openTab(event, \'coarse_{scenario_id}\')">{scenario_name}</button>\n'
                
#                 display_style = "block" if i == 0 else "none"
#                 coarse_tabs_content += f"""
#     <div id="coarse_{scenario_id}" class="tab-content" style="display: {display_style};">
# {coarse_scenario_tabs[i][1]}
#     </div>
# """
            
#             # Build tab buttons and content for regional section
#             regional_tabs_buttons = ""
#             regional_tabs_content = ""
#             for i, (scenario_id, scenario_name) in enumerate(scenario_list):
#                 active_class = "active" if i == 0 else ""
#                 regional_tabs_buttons += f'<button class="tab-button {active_class}" onclick="openTab(event, \'regional_{scenario_id}\')">{scenario_name}</button>\n'
                
#                 display_style = "block" if i == 0 else "none"
#                 regional_tabs_content += f"""
#     <div id="regional_{scenario_id}" class="tab-content" style="display: {display_style};">
# {regional_scenario_tabs[i][1]}
#     </div>
# """
            
#             # Build tab buttons and content for stitched LULC section
#             stitched_tabs_buttons = ""
#             stitched_tabs_content = ""
#             for i, (scenario_id, scenario_name) in enumerate(scenario_list):
#                 active_class = "active" if i == 0 else ""
#                 stitched_tabs_buttons += f'<button class="tab-button {active_class}" onclick="openTab(event, \'stitched_{scenario_id}\')">{scenario_name}</button>\n'
                
#                 display_style = "block" if i == 0 else "none"
#                 stitched_tabs_content += f"""
#     <div id="stitched_{scenario_id}" class="tab-content" style="display: {display_style};">
# {stitched_scenario_tabs[i][1]}
#     </div>
# """

#             # Function to convert image to base64
#             import base64
#             def image_to_base64(image_path):
#                 """Convert image file to base64 data URI"""
#                 if os.path.exists(image_path):
#                     with open(image_path, 'rb') as img_file:
#                         img_data = base64.b64encode(img_file.read()).decode('utf-8')
#                         return f'data:image/png;base64,{img_data}'
#                 return ''
            
#             # Collect all image paths and convert to base64
#             image_embeds = {}
            
#             # Static images
#             for img_name in ['aoi_plots.png', 'fine_processed_inputs_plots.png', 'region_ids.png']:
#                 img_path = os.path.join(report_assets_dir, img_name)
#                 image_embeds[img_name] = image_to_base64(img_path)
            
#             # Dynamic scenario images
#             for filename in os.listdir(report_assets_dir):
#                 if filename.endswith('.png') and filename not in image_embeds:
#                     img_path = os.path.join(report_assets_dir, filename)
#                     image_embeds[filename] = image_to_base64(img_path)
            
#             # Replace all asset references with base64 data
#             def embed_images_in_html(html_content, image_dict):
#                 """Replace src="assets/..." with base64 data URIs"""
#                 for filename, base64_data in image_dict.items():
#                     if base64_data:  # Only replace if image exists
#                         html_content = html_content.replace(f'src="assets/{filename}"', f'src="{base64_data}"')
#                 return html_content

#             # Generate HTML report
#             html_template = """<!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>SEALS Visualization Report</title>
#     <style>
#         body {{
#             font-family: Arial, sans-serif;
#             max-width: 1400px;
#             margin: 0 auto;
#             padding: 20px;
#             background-color: #f5f5f5;
#         }}
#         h1 {{
#             color: #333;
#             border-bottom: 3px solid #4CAF50;
#             padding-bottom: 10px;
#         }}
#         h2 {{
#             color: #555;
#             margin-top: 40px;
#             border-bottom: 2px solid #ddd;
#             padding-bottom: 5px;
#         }}
#         h3 {{
#             color: #666;
#             margin-top: 25px;
#         }}
#         h4 {{
#             color: #777;
#             margin-top: 15px;
#             font-size: 1em;
#         }}
#         .image-container {{
#             background: white;
#             padding: 20px;
#             margin: 20px 0;
#             border-radius: 8px;
#             box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#         }}
#         img {{
#             max-width: 100%;
#             height: auto;
#             display: block;
#             margin: 0 auto;
#         }}
#         .date {{
#             color: #888;
#             font-size: 0.9em;
#         }}
#         p {{
#             line-height: 1.6;
#         }}
#         .description {{
#             background: #fff;
#             padding: 15px;
#             margin: 20px 0;
#             border-radius: 5px;
#             border-left: 4px solid #4CAF50;
#         }}
#         .description strong {{
#             color: #4CAF50;
#         }}
#         code {{
#             background: #f4f4f4;
#             padding: 2px 6px;
#             border-radius: 3px;
#             font-family: 'Courier New', monospace;
#             font-size: 0.9em;
#         }}
#         pre {{
#             background: #f4f4f4;
#             padding: 10px;
#             border-radius: 5px;
#             overflow-x: auto;
#             font-size: 0.85em;
#         }}
        
#         /* Tab styles */
#         .tab-container {{
#             margin: 20px 0;
#         }}
#         .tab-buttons {{
#             overflow: hidden;
#             border-bottom: 2px solid #4CAF50;
#             margin-bottom: 20px;
#         }}
#         .tab-button {{
#             background-color: #f1f1f1;
#             border: none;
#             outline: none;
#             cursor: pointer;
#             padding: 12px 20px;
#             transition: 0.3s;
#             font-size: 14px;
#             margin-right: 2px;
#             border-top-left-radius: 5px;
#             border-top-right-radius: 5px;
#         }}
#         .tab-button:hover {{
#             background-color: #ddd;
#         }}
#         .tab-button.active {{
#             background-color: #4CAF50;
#             color: white;
#         }}
#         .tab-content {{
#             padding: 10px;
#             border-radius: 5px;
#         }}
#     </style>
#     <script>
#         function openTab(evt, tabId) {{
#             // Hide all tab contents in the same section
#             var section = tabId.split('_')[0]; // 'coarse' or 'regional'
#             var allTabs = document.querySelectorAll('[id^="' + section + '_"]');
#             allTabs.forEach(function(tab) {{
#                 tab.style.display = 'none';
#             }});
            
#             // Remove active class from all buttons in the same section
#             var buttons = evt.currentTarget.parentElement.getElementsByClassName('tab-button');
#             for (var i = 0; i < buttons.length; i++) {{
#                 buttons[i].className = buttons[i].className.replace(' active', '');
#             }}
            
#             // Show the selected tab and mark button as active
#             document.getElementById(tabId).style.display = 'block';
#             evt.currentTarget.className += ' active';
#         }}
#     </script>
# </head>
# <body>
#     <h1>SEALS Visualization Report</h1>
#     <p class="date">Generated: {date}, User: {user}, OS: {system}</p>
#     {scenarios_table}
    
#     <h2>Area of Interest (AOI)</h2>
#     <div class="description">
#         <p><strong>Purpose:</strong></p>
#         <p>Defines the geographic boundary for the analysis and creates resolution pyramids for efficient processing at multiple scales.</p>
#         <ul>
#             <li>Loads the area of interest boundary from the input shapefile</li>
#             <li>Creates a vector geopackage (<code>.gpkg</code>) file for the AOI</li>
#             <li>Generates raster "pyramids" at different resolutions:
#                 <ul>
#                     <li>Fine resolution: Matches the base LULC data (typically 300m)</li>
#                     <li>Coarse resolution: Matches the coarse projection data (typically 0.25°)</li>
#                 </ul>
#             </li>
#         </ul>
#         <p><strong>Key outputs:</strong></p>
#         <pre>intermediate/project_aoi/
# ├── aoi_RWA.gpkg                           # Vector boundary
# └── pyramids/
#     ├── aoi_ha_per_cell_coarse.tif    # Coarse resolution
#     └── aoi_ha_per_cell_fine.tif      # Fine resolution</pre>
#     </div>
#     <div class="image-container">
#         <img src="assets/aoi_plots.png" alt="AOI plots showing boundary, fine resolution, and coarse resolution">
#     </div>
    
#     <h2>Fine Processed Inputs</h2>
#     <div class="description">
#         <p><strong>Purpose:</strong></p>
#         <p>Prepares the baseline land-use/land-cover (LULC) data for allocation by simplifying classifications, creating binary masks, and generating spatial convolutions.Binary masks and convolutions inform the allocation model by capturing spatial patterns and neighborhood characteristics.</p>
#         <ul>
#             <li><strong>Generated Kernels:</strong> Creates Gaussian kernels for spatial smoothing</li>
#             <li><strong>LULC Simplifications:</strong> Reclassifies detailed ESA categories into 7 SEALS classes:
#                 <ul>
#                     <li>Cropland, Forest, Grassland, Urban, Water, Other natural, Other</li>
#                 </ul>
#             </li>
#             <li><strong>LULC Binaries:</strong> Creates separate binary masks for each class (1 = present, 0 = absent)</li>
#             <li><strong>LULC Convolutions:</strong> Applies Gaussian smoothing to capture neighborhood characteristics</li>
#         </ul>
#         <p><strong>Key outputs:</strong></p>
#         <pre>intermediate/fine_processed_inputs/lulc/esa/seals7/
# ├── lulc_esa_seals7_2017.tif              # Simplified 7-class LULC
# ├── binaries/2017/                         # Binary masks by class
# │   ├── binary_esa_seals7_2017_cropland.tif
# │   └── ...
# └── convolutions/2017/                     # Smoothed neighborhood data
#     ├── convolution_esa_seals7_2017_cropland_gaussian_1.tif
#     └── ...</pre>
#     </div>
#     <div class="image-container">
#         <img src="assets/fine_processed_inputs_plots.png" alt="Fine processed inputs showing LULC classification and convolutions">
#     </div>
    
#     <h2>Coarse Change</h2>
#     <div class="description">
#         <p><strong>Purpose:</strong></p>
#         <p>Extracts and processes land-use change projections from global coarse-resolution models (e.g., LUH2) for the study region. Links exogenous global scenarios (SSP, RCP) to local allocation by quantifying how much of each land class needs to change.</p>
#         <ul>
#             <li><strong>Coarse Extraction:</strong> Clips global projection data to the region's bounding box</li>
#             <li><strong>Coarse Simplified Proportion:</strong> Calculates proportions of each SEALS7 class</li>
#             <li><strong>Coarse Simplified Ha:</strong> Converts proportions to hectares</li>
#             <li><strong>Ha Difference:</strong> Calculates change in hectares between time periods (e.g., 2017 → 2030 → 2050)</li>
#         </ul>
#         <p><strong>Key outputs:</strong></p>
#         <pre>intermediate/coarse_change/coarse_simplified_ha_difference_from_previous_year/
# └── ssp2/rcp45/luh2-message/bau/2030/
#     ├── cropland_2030_2017_ha_diff_ssp2_rcp45_luh2-message_bau.tif
#     ├── forest_2030_2017_ha_diff_ssp2_rcp45_luh2-message_bau.tif
#     └── ...</pre>
#     </div>
#     <div class="tab-container">
#         <div class="tab-buttons">
# {coarse_tabs_buttons}
#         </div>
# {coarse_tabs_content}
#     </div>
    
#     <h2>Regional Change</h2>
#     <div class="description">
#         <p><strong>Purpose:</strong></p>
#         <p>Adjusts coarse-resolution changes using regional covariates to account for local constraints and opportunities.</p>
#         <ul>
#             <li>Creates a region ID map dividing AOI into separately processed zones</li>
#             <li>Applies algorithms (e.g., proportional, covariate sum shift) to downscale from coarse to regional resolution</li>
#             <li><strong>Regional Coarsified:</strong> Aggregated regional change</li>
#             <li><strong>Covariate Sum Shift:</strong> Change adjusted by spatial covariates (infrastructure, topography, current land use)</li>
#         </ul>
#         <p><strong>Key outputs:</strong></p>
#         <pre>intermediate/regional_change/
# ├── region_ids.tif                          # Regional zone identifiers
# └── ssp2/rcp45/luh2-message/bau/2030/
#     ├── cropland_2030_2017_ha_diff_..._regional_coarsified.tif
#     └── cropland_2030_2017_ha_diff_..._covariate_sum_shift.tif</pre>
#     </div>
#     {region_ids_section}
#     <div class="tab-container">
#         <div class="tab-buttons">
# {regional_tabs_buttons}
#         </div>
# {regional_tabs_content}
#     </div>
    
#     <h2>Stitched LULC Scenarios</h2>
#     <div class="description">
#         <p><strong>Purpose:</strong></p>
#         <p>Combines all allocation zones back into complete region-wide LULC maps for each scenario.</p>
#         <ul>
#             <li>Stitches together individual allocation blocks</li>
#             <li>Clips to exact boundary</li>
#             <li>Generates outputs for all scenario combinations (bau, policy scenarios, etc.)</li>
#             <li>Creates both full extent and clipped versions</li>
#         </ul>
#         <p><strong>Key outputs:</strong></p>
#         <pre>intermediate/stitched_lulc_simplified_scenarios/
# ├── lulc_esa_seals7_ssp2_rcp45_luh2-message_bau_2030.tif
# ├── lulc_esa_seals7_ssp2_rcp45_luh2-message_bau_2050.tif
# └── lulc_esa_seals7_ssp2_rcp45_luh2-message_bau_2030_clipped.tif</pre>
#     </div>
#     <div class="tab-container">
#         <div class="tab-buttons">
# {stitched_tabs_buttons}
#         </div>
# {stitched_tabs_content}
#     </div>
    
# </body>
# </html>"""

#             # Format the HTML with current date and dynamic sections
#             html_content = html_template.format(
#                 date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                 user=getpass.getuser(),
#                 system=platform.system(),
#                 scenarios_table=scenarios_table_html,
#                 region_ids_section=region_ids_html,
#                 coarse_tabs_buttons=coarse_tabs_buttons,
#                 coarse_tabs_content=coarse_tabs_content,
#                 regional_tabs_buttons=regional_tabs_buttons,
#                 regional_tabs_content=regional_tabs_content,
#                 stitched_tabs_buttons=stitched_tabs_buttons,
#                 stitched_tabs_content=stitched_tabs_content
#             )

#             # Embed all images as base64
#             html_content = embed_images_in_html(html_content, image_embeds)
            
#             # Write HTML file
#             with open(report_output_path, 'w') as f:
#                 f.write(html_content)
            