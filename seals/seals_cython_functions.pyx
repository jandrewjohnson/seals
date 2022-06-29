# cython: cdivision=True
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#cython: boundscheck=False, wraparound=False
from libc.math cimport log
import hazelbean as hb
import os
import time
from collections import OrderedDict
from cython.parallel cimport prange
import scipy.ndimage
import cython
cimport cython
import numpy as np  # NOTE, both imports are required. cimport adds extra information to the pyd while the import actually defines numppy
cimport numpy as np
from numpy cimport ndarray
from libc.math cimport sin
from libc.math cimport fabs
import math, time

@cython.cdivision(False)
@cython.boundscheck(True)
@cython.wraparound(True)
def seals_allocation_vector_input(ndarray[np.float64_t, ndim=1] change_vector_1d not None,
                                   ndarray[np.int64_t, ndim=2] input_lulc not None,
                                   ndarray[np.float64_t, ndim=3] spatial_layers_3d not None,
                                   ndarray[np.float64_t, ndim=2] spatial_layer_coefficients_2d not None,
                                   ndarray[np.int64_t, ndim=1] spatial_layer_function_types_1d not None,
                                   ndarray[np.int64_t, ndim=2] valid_mask_array not None,
                                   ndarray[np.int64_t, ndim=1] change_class_ids not None,
                                   ndarray[np.float64_t, ndim=2] hectares_per_grid_cell not None,
                                   str output_dir,
                                   double reporting_level,
                                   str call_string,
                                   ):

    cdef size_t n_fine_rows = input_lulc.shape[0]
    cdef size_t n_fine_cols = input_lulc.shape[1]

    cdef np.int64_t coarse_r, coarse_c, fine_r, fine_c, chunk_r, chunk_c, class_i, class_j, i, j, current_fine_starting_r, current_fine_starting_c, interior_allocation_step, k, regressor_k

    cdef long long counter = 1
    cdef long long while_counter = 1

    # # NOTE, before, resolution was oddly used as a word to indicate the coarse_cell size, which wasn't necessarily the same as the iterator chunk.
    # cdef np.int64_t resolution = input_lulc.shape[1] / coarse_change_3d[0].shape[1]
    # cdef np.int64_t other_resolution = input_lulc.shape[0] / coarse_change_3d[0].shape[0]
    # if not resolution == other_resolution:
    #     print ('WARNING, resolutions not        amicable.', resolution, other_resolution)

    cdef size_t n_allocation_classes = len(change_vector_1d)

    cdef np.int64_t n_fine_cells = n_fine_rows * n_fine_cols

    cdef np.ndarray[np.int64_t, ndim=2] raveled = np.zeros((n_allocation_classes, n_fine_cells), dtype=np.int64)

    cdef np.ndarray[np.int64_t, ndim=2] current_ranked_rows = np.zeros((n_allocation_classes, n_fine_cells), dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=2] current_ranked_cols = np.zeros((n_allocation_classes, n_fine_cells), dtype=np.int64)

    cdef int write_fine_r = 0
    cdef int write_fine_c = 0

    cdef np.ndarray[np.float64_t, ndim=3] to_rank_arrays = np.zeros((n_allocation_classes, n_fine_rows, n_fine_cols), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=3] rank_arrays = np.zeros((n_allocation_classes, n_fine_rows, n_fine_cols), dtype=np.float64)

    cdef np.ndarray[np.float64_t, ndim=1] current_goals = np.zeros(n_allocation_classes, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] current_goal_left = np.zeros(n_allocation_classes, dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim=1] current_positions = np.zeros(n_allocation_classes, dtype=np.int64)

    cdef np.ndarray[np.float64_t, ndim=3] output_to_rank_arrays = np.zeros((n_allocation_classes, n_fine_rows, n_fine_cols), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=3] output_rank_arrays = np.zeros((n_allocation_classes, n_fine_rows, n_fine_cols), dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim=3] output_change_arrays = np.zeros((n_allocation_classes, n_fine_rows, n_fine_cols), dtype=np.int64)

    cdef np.ndarray[np.int64_t, ndim=2] projected_lulc = np.copy(input_lulc).astype(np.int64)

    if call_string is not '':
        pass
        # print ('Cython call_string: ' + call_string)

    # current_fine_starting_r = coarse_r * resolution
    # current_fine_starting_c = coarse_c * resolution

    ## Create the arrays to rank

    # Add in all within-cell regressors
    for class_j in range(n_allocation_classes):
        # NOTE: we iterate through regressors first for additive, then for multiplicative because the formula really is (a + b + c + d +...) * e * f

        for regressor_k in range(len(spatial_layers_3d)):
            if spatial_layer_function_types_1d[regressor_k] == 2 and spatial_layer_coefficients_2d[class_j, regressor_k] != 0:  # Additive
                to_rank_arrays[class_j] += (spatial_layer_coefficients_2d[class_j, regressor_k] *
                                                    spatial_layers_3d[regressor_k])
        for regressor_k in range(len(spatial_layers_3d)):
            if spatial_layer_function_types_1d[regressor_k] == 1:  # Multiplicative
                to_rank_arrays[class_j] *= 1.0 - (
                        ((spatial_layer_coefficients_2d[class_j, regressor_k] - 1) * -1.0) *
                        spatial_layers_3d[regressor_k]
                )

        # Also mask invalid now
        to_rank_arrays[class_j] *= valid_mask_array

        # Set locations that already have class_j as its type to zero
        to_rank_arrays[class_j] *= np.where(input_lulc == change_class_ids[class_j], 0, 1)

        # Flip array so that we start from highest value
        to_rank_arrays[class_j] *= -1.0

        # Set zeros to high value for ranking purposes
        to_rank_arrays[class_j][to_rank_arrays[class_j] == 0] = 9.e9  # NOTE LOGIC, low values ranked first, but areas with ZERO are used as no data, so we don't want them in the rank between pos and neg values.

        # Values very close to zero no longer have smooth ranking surfaces due to floating point errors.
        to_rank_arrays[class_j][(to_rank_arrays[class_j] < 1.e-14) & (to_rank_arrays[class_j] > -1.e-14)] = 9.e9  # NOTE LOGIC, low values ranked first, but areas with ZERO are used as no data, so we don't want them in the rank between pos and neg values.
        to_rank_arrays[class_j][to_rank_arrays[class_j] < -9999999999999999] = 9.e9  # NOTE, necessary to fix what I think was an underflow error. was getting a lot of 1e-35 numbers.

        # For visualization, also save to full-extent array
        output_to_rank_arrays[class_j] = to_rank_arrays[class_j]

    # Do the ranking
    for class_j in range(n_allocation_classes):
        # Sort each class_j array
        raveled[class_j] = to_rank_arrays[class_j].argsort(axis=None)
        output_rank_arrays[class_j] = raveled[class_j]
        # if reporting_level >= 5:  # Write rank arrays
        #     counter = 0
        #     for i in range(n_fine_cells):
        #         if output_to_rank_arrays[class_j, current_fine_starting_r + raveled[class_j, i] / resolution, current_fine_starting_c + <int> (raveled[class_j, i] % resolution)] <= 999999:
        #             rank_arrays[class_j, raveled[class_j, i] / resolution, <int> (raveled[class_j, i] % resolution)] = counter
        #             counter += 1
        #     output_rank_arrays[class_j, current_fine_starting_r: current_fine_starting_r + resolution, current_fine_starting_c: current_fine_starting_c + resolution] = rank_arrays[class_j]

    # Do the allocation
    for class_j in range(n_allocation_classes):

        current_goals[class_j] = change_vector_1d[class_j]
        current_goal_left[class_j] = current_goals[class_j]
        current_positions[class_j] = 0
    for while_counter in range(n_fine_cells):
        if while_counter % 1000000 == 0:
            print (while_counter, n_fine_cells, class_j)
        for class_j in range(n_allocation_classes):
            if current_goal_left[class_j] > 0:

                if to_rank_arrays[class_j, <int> (raveled[class_j, current_positions[class_j]]), raveled[class_j, current_positions[class_j]] ] < 999999999.0:
                    # Get current position OVERALL (i.e., including current_fine_starting_x) based on dividing and moduloing the current id.
                    write_fine_r = raveled[class_j, current_positions[class_j]]
                    write_fine_c = raveled[class_j, current_positions[class_j]]

                    # Write 0-1 to output_change_arrays (3dim) specific to this fine location and this expansion class.
                    output_change_arrays[class_j, write_fine_r, write_fine_c] = 1

                    # Write the class_j's label value to the projected lulc map
                    projected_lulc[write_fine_r, write_fine_c] = change_class_ids[class_j]

                    # Increment the current allocation position by 1, moving the the next best cell.
                    current_positions[class_j] += 1

                    # Reduce the current class's goal by the amount of hectares in that
                    current_goal_left[class_j] -= hectares_per_grid_cell[write_fine_r, write_fine_c]

    if reporting_level >= 11:
        for i in range(n_allocation_classes):
            # hb.show(output_rank_arrays[i], output_path=hb.ruri(os.path.join(output_dir, 'output_rank_for_class_' + str(i) + '.png')), title='output_rank for class ' + str(i))
            hb.save_array_as_geotiff(output_rank_arrays[i], hb.suri(os.path.join(output_dir, 'output_rank_for_class_' + str(i) + '.tif'), call_string), os.path.join(os.path.join(output_dir, 'lulc_esa_simplified_2014.tif')), data_type=7)
    if reporting_level >= 5:
        for i in range(n_allocation_classes):
            # hb.show(np.where(output_to_rank_arrays[i] > 9.e9, np.nan, output_to_rank_arrays[i]), vmin=0, vmax=100, output_path=hb.ruri(os.path.join(output_dir, 'overall_suitability_for_class_' + str(i) + '.png')), data_type = 7, title='overall_suitability for class ' + str(i))
            hb.save_array_as_geotiff(output_to_rank_arrays[i], hb.suri(os.path.join(output_dir, 'output_to_rank_for_class_' + str(i) + '.tif'), call_string), os.path.join(os.path.join(output_dir, 'lulc_esa_simplified_2014.tif')), data_type=7)

    if reporting_level >= 5:
        for i in range(n_allocation_classes):
            hb.save_array_as_geotiff(output_change_arrays[i], hb.suri(os.path.join(output_dir, 'allocations_for_class_' + str(i) + '.tif'), call_string), os.path.join(os.path.join(output_dir, 'lulc_esa_simplified_2014.tif')))

            # hb.show(output_change_arrays[i], output_path=hb.ruri(os.path.join(output_dir, 'allocations_for_class_' + str(i) + '.png')), vmin=0, vmax=1, title='allocations for class ' + str(i))
    if reporting_level >= 5:
        hb.save_array_as_geotiff(projected_lulc, hb.suri(os.path.join(output_dir, 'projected_lulc.tif'), call_string), os.path.join(os.path.join(output_dir, 'lulc_esa_simplified_2014.tif')))

        # # hb.show(input_lulc, output_path=hb.ruri(os.path.join(output_dir, 'input_lulc.png')), title='input_lulc', vmin=0, vmax=7, ndv=255)
        # hb.show(projected_lulc, output_path=hb.ruri(os.path.join(output_dir, 'projected_lulc.png')), title='projected_lulc', vmin=0, vmax=7, ndv=255, block_plotting=False)

    return projected_lulc

@cython.cdivision(False)
@cython.boundscheck(True)
@cython.wraparound(True)
def seals_allocation_exclusive(ndarray[np.float64_t, ndim=3] coarse_change_3d not None,
                     ndarray[np.int64_t, ndim=2] input_lulc not None,
                     ndarray[np.float64_t, ndim=3] spatial_layers_3d not None,
                     ndarray[np.float64_t, ndim=2] spatial_layer_coefficients_2d not None,
                     ndarray[np.int64_t, ndim=1] spatial_layer_function_types_1d not None,
                     ndarray[np.int64_t, ndim=2] valid_mask_array not None,
                     ndarray[np.int64_t, ndim=1] change_class_ids not None,
                     ndarray[np.float64_t, ndim=2] hectares_per_grid_cell not None,
                     str output_dir,
                     double reporting_level,
                     str call_string,
                     ):

    cdef size_t n_coarse_rows = coarse_change_3d[0].shape[0]
    cdef size_t n_coarse_cols = coarse_change_3d[0].shape[1]

    cdef size_t n_fine_rows = input_lulc.shape[0]
    cdef size_t n_fine_cols = input_lulc.shape[1]

    cdef np.int64_t coarse_r, coarse_c, fine_r, fine_c, chunk_r, chunk_c, class_i, class_j, class_l_check, i, j, current_fine_starting_r, current_fine_starting_c, interior_allocation_step, k, regressor_k

    cdef long long counter = 1
    cdef long long while_counter = 1

    cdef np.int64_t resolution = input_lulc.shape[1] / coarse_change_3d[0].shape[1]
    cdef np.int64_t other_resolution = input_lulc.shape[0] / coarse_change_3d[0].shape[0]
    if not resolution == other_resolution:
        print ('WARNING, resolutions not amicable.', resolution, other_resolution)

    cdef size_t n_chunk_rows = resolution
    cdef size_t n_chunk_cols = resolution

    cdef size_t n_allocation_classes = coarse_change_3d.shape[0]

    # cdef np.float64_t coarse_outcome = 0.0
    # cdef np.float64_t fine_outcome = 0.0
    # cdef np.float64_t num_to_allocate_this_class = 0.0
    # cdef np.float64_t total_absolute_change_needed = 0.0

    cdef np.int64_t n_fine_grid_cells_per_coarse_cell = <int> (resolution * resolution)

    cdef np.ndarray[np.int64_t, ndim=2] current_raveled = np.zeros((n_allocation_classes, n_fine_grid_cells_per_coarse_cell), dtype=np.int64)

    # cdef np.ndarray[np.int64_t, ndim=2] current_ranked_rows = np.zeros((n_allocation_classes, n_fine_grid_cells_per_coarse_cell), dtype=np.int64)
    # cdef np.ndarray[np.int64_t, ndim=2] current_ranked_cols = np.zeros((n_allocation_classes, n_fine_grid_cells_per_coarse_cell), dtype=np.int64)


    cdef int write_fine_r = 0
    cdef int write_fine_c = 0

    cdef np.ndarray[np.float64_t, ndim=3] current_to_rank_arrays = np.zeros((n_allocation_classes, resolution, resolution), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=3] current_rank_arrays = np.zeros((n_allocation_classes, resolution, resolution), dtype=np.float64)

    cdef np.ndarray[np.float64_t, ndim=1] current_goals = np.zeros(n_allocation_classes, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] current_goal_left = np.zeros(n_allocation_classes, dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim=1] current_positions = np.zeros(n_allocation_classes, dtype=np.int64)

    cdef np.ndarray[np.float64_t, ndim=3] output_to_rank_arrays = np.zeros((n_allocation_classes, n_fine_rows, n_fine_cols), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=3] output_rank_arrays = np.zeros((n_allocation_classes, n_fine_rows, n_fine_cols), dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim=3] output_change_arrays = np.zeros((n_allocation_classes, n_fine_rows, n_fine_cols), dtype=np.int64)

    cdef np.ndarray[np.int64_t, ndim=2] projected_lulc = np.copy(input_lulc).astype(np.int64)

    if call_string is not '':
        pass
        # print ('Cython call_string: ' + call_string)

    # Get to correct coarse cell
    for coarse_r in range(n_coarse_rows):
        for coarse_c in range(n_coarse_cols):

            # First check if there's any allocation to do, skipping if not for speed
            total_absolute_change_needed = 0
            for class_j in range(n_allocation_classes):
                total_absolute_change_needed += abs(coarse_change_3d[class_j, coarse_r, coarse_c])

            if total_absolute_change_needed > 0.0:
                current_to_rank_arrays = np.zeros((n_allocation_classes, resolution, resolution), dtype=np.float64)
                current_rank_arrays = np.zeros((n_allocation_classes, resolution, resolution), dtype=np.float64)
                current_fine_starting_r = coarse_r * resolution
                current_fine_starting_c = coarse_c * resolution

                ## Create the arrays to rank
                for class_j in range(n_allocation_classes):
                    current_goals[class_j] = coarse_change_3d[class_j, coarse_r, coarse_c]
                    current_goal_left[class_j] = current_goals[class_j]
                    current_positions[class_j] = 0

                # Add in all within-cell regressors
                for class_j in range(n_allocation_classes):
                    # NOTE: we iterate through regressors first for additive, then for multiplicative because the formula really is (a + b + c + d +...) * e * f

                    for regressor_k in range(len(spatial_layers_3d)):
                        if spatial_layer_function_types_1d[regressor_k] == 2 and spatial_layer_coefficients_2d[class_j, regressor_k] != 0: # Additive
                            current_to_rank_arrays[class_j] += (spatial_layer_coefficients_2d[class_j, regressor_k] *
                                                            spatial_layers_3d[regressor_k, current_fine_starting_r: current_fine_starting_r + resolution,
                                                            current_fine_starting_c: current_fine_starting_c + resolution])

                    # NOTE THE VERY SPECIFIC and potentially strange logic of booleans that actually aren't booleans.
                    # If you have a coefficient of 0, then values with 1 cannot have that cell. But if you try multiplicative
                    # on a continuous, 0 to 1 value we end up subtracting 1 - (0 - 1) * -1 * 0.25 = 0.75. You may want to clarifiy the difference between a
                    # multiplicative binary and multipliciative continuous.
                    for regressor_k in range(len(spatial_layers_3d)):
                        if spatial_layer_function_types_1d[regressor_k] == 1: # Multiplicative

                            current_to_rank_arrays[class_j] *= 1.0 - (
                                ((spatial_layer_coefficients_2d[class_j, regressor_k] - 1) * -1.0) *
                                    spatial_layers_3d[regressor_k, current_fine_starting_r: current_fine_starting_r + resolution, current_fine_starting_c: current_fine_starting_c + resolution]
                            )

                            # Get the class of regressor_k. NOTE THAT THIS ASSUMES THE FIRST spatial_layers are the classes who have allocation goals.

                            ### START HERE: Exclusive allocation is working, but note that it means that classes with contraction don't necessarily get contracted the exact ammount expected.
                            # I think the only solution to this is to consider i-j allocations as all unique, not just i in aggregate
                            if regressor_k < n_allocation_classes:
                                if current_goals[regressor_k] > 0:

                                    current_to_rank_arrays[class_j] *= -1 * (spatial_layers_3d[regressor_k, current_fine_starting_r: current_fine_starting_r + resolution, current_fine_starting_c: current_fine_starting_c + resolution]
                                                                         - 1)

                    # Also mask invalid now
                    current_to_rank_arrays[class_j] *= valid_mask_array[current_fine_starting_r: current_fine_starting_r + resolution, current_fine_starting_c: current_fine_starting_c + resolution]

                    # Set locations that already have class_j as its type to zero
                    current_to_rank_arrays[class_j] *= np.where(input_lulc[current_fine_starting_r: current_fine_starting_r + resolution, current_fine_starting_c: current_fine_starting_c + resolution] == change_class_ids[class_j], 0, 1)

                    # Flip array so that we start from highest value
                    current_to_rank_arrays[class_j] *= -1.0

                    # Set zeros to high value for ranking purposes
                    current_to_rank_arrays[class_j][current_to_rank_arrays[class_j] == 0] = 9.e9  # NOTE LOGIC, low values ranked first, but areas with ZERO are used as no data, so we don't want them in the rank between pos and neg values.

                    # Values very close to zero no longer have smooth ranking surfaces due to floating point errors.
                    current_to_rank_arrays[class_j][(current_to_rank_arrays[class_j] < 1.e-14) & (current_to_rank_arrays[class_j] > -1.e-14)] = 9.e9  # NOTE LOGIC, low values ranked first, but areas with ZERO are used as no data, so we don't want them in the rank between pos and neg values.
                    current_to_rank_arrays[class_j][current_to_rank_arrays[class_j] < -9999999999999999] = 9.e9 # NOTE, necessary to fix what I think was an underflow error. was getting a lot of 1e-35 numbers.
                    # For visualization, also save to full-extent array
                    output_to_rank_arrays[class_j, current_fine_starting_r: current_fine_starting_r + resolution, current_fine_starting_c: current_fine_starting_c + resolution] = current_to_rank_arrays[class_j]

                # Do the ranking
                for class_j in range(n_allocation_classes):

                    # TODOO Speed up ranking by eliminating nans.
                    # Sort each class_j array
                    current_raveled[class_j] = current_to_rank_arrays[class_j].argsort(axis=None)

                    # TODOO Abandonded start: It is trickier than expected to "rerank"
                    # the rank to eliminate edges because it calls adjacent cells. Need to have a new instantiat9ion of the tiling-system
                    # that includes single-read memory blocking of all 9 cells.
                    # HOWEVER, probably the fastest approach would be to embrace both the clibration (on the fly) calculatinon of everything
                    # but then also a preprocesed version of the allocation for speed. the preprocessin gstep then could do the smoothing.
                    # if smooth_rank:
                    #     current_raveled[class_j] = current_to_rank_arrays[class_j].argsort(axis=None)


                    if reporting_level >= 5: # Write rank arrays
                        counter = 0
                        for i in range(n_fine_grid_cells_per_coarse_cell):
                            if output_to_rank_arrays[class_j, current_fine_starting_r + current_raveled[class_j, i] / resolution, current_fine_starting_c + <int>(current_raveled[class_j, i] % resolution)] <= 999999:
                                current_rank_arrays[class_j, current_raveled[class_j, i] / resolution, <int> (current_raveled[class_j, i] % resolution)] = counter
                                counter += 1
                        output_rank_arrays[class_j, current_fine_starting_r: current_fine_starting_r + resolution, current_fine_starting_c: current_fine_starting_c + resolution] = current_rank_arrays[class_j]

                # Do the allocation

                for while_counter in range(n_fine_grid_cells_per_coarse_cell):
                    for class_j in range(n_allocation_classes):
                        if current_goal_left[class_j] > 0:

                            if current_to_rank_arrays[class_j, <int> (current_raveled[class_j, current_positions[class_j]] / resolution), current_raveled[class_j, current_positions[class_j]] % resolution] < 999999999.0:

                                # Get current position OVERALL (i.e., including current_fine_starting_x) based on dividing and moduloing the current id.
                                current_fine_r = current_fine_starting_r + <int> (current_raveled[class_j, current_positions[class_j]] / resolution)
                                current_fine_c = current_fine_starting_c + current_raveled[class_j, current_positions[class_j]] % resolution

                                # Write 0-1 to output_change_arrays (3dim) specific to this fine location and this expansion class.
                                output_change_arrays[class_j, current_fine_r, current_fine_c] = 1

                                # Write the class_j's label value to the projected lulc map
                                projected_lulc[current_fine_r, current_fine_c] = change_class_ids[class_j]

                                # Increment the current allocation position by 1, moving the the next best cell.
                                current_positions[class_j] += 1

                                # Reduce the current class's goal by the amount of hectares in that
                                current_goal_left[class_j] -= hectares_per_grid_cell[current_fine_r, current_fine_c]

    if reporting_level >= 11:
        for i in range(n_allocation_classes):

            # hb.show(output_rank_arrays[i], output_path=hb.ruri(os.path.join(output_dir, 'output_rank_for_class_' + str(i) + '.png')), title='output_rank for class ' + str(i))
            hb.save_array_as_geotiff(output_rank_arrays[i], hb.suri(os.path.join(output_dir, 'output_rank_for_class_' + str(i) + '.tif'), call_string), os.path.join(os.path.join(output_dir, 'lulc_baseline.tif')), data_type=7)
    if reporting_level >= 5:
        for i in range(n_allocation_classes):
            # hb.show(np.where(output_to_rank_arrays[i] > 9.e9, np.nan, output_to_rank_arrays[i]), vmin=0, vmax=100, output_path=hb.ruri(os.path.join(output_dir, 'overall_suitability_for_class_' + str(i) + '.png')), data_type = 7, title='overall_suitability for class ' + str(i))
            hb.save_array_as_geotiff(output_to_rank_arrays[i], hb.suri(os.path.join(output_dir, 'output_to_rank_for_class_' + str(i) + '.tif'), call_string), os.path.join(os.path.join(output_dir, 'lulc_baseline.tif')), data_type=7)




    if reporting_level >= 5:
        for i in range(n_allocation_classes):
            hb.save_array_as_geotiff(output_change_arrays[i], hb.suri(os.path.join(output_dir, 'allocations_for_class_' + str(i) + '.tif'), call_string), os.path.join(os.path.join(output_dir, 'lulc_baseline.tif')))

            # hb.show(output_change_arrays[i], output_path=hb.ruri(os.path.join(output_dir, 'allocations_for_class_' + str(i) + '.png')), vmin=0, vmax=1, title='allocations for class ' + str(i))
    if reporting_level >= 4:
        hb.save_array_as_geotiff(projected_lulc, hb.suri(os.path.join(output_dir, 'projected_lulc.tif'), call_string), os.path.join(os.path.join(output_dir, 'lulc_baseline.tif')))

        # # hb.show(input_lulc, output_path=hb.ruri(os.path.join(output_dir, 'input_lulc.png')), title='input_lulc', vmin=0, vmax=7, ndv=255)
        # hb.show(projected_lulc, output_path=hb.ruri(os.path.join(output_dir, 'projected_lulc.png')), title='projected_lulc', vmin=0, vmax=7, ndv=255, block_plotting=False)

    return projected_lulc

@cython.cdivision(False)
@cython.boundscheck(True)
@cython.wraparound(True)
def seals_allocation_from_change_matrix(ndarray[np.float64_t, ndim=4] coarse_change_matrix_4d not None,
                     ndarray[np.int64_t, ndim=2] input_lulc not None,
                     ndarray[np.float64_t, ndim=3] spatial_layers_3d not None,
                     ndarray[np.float64_t, ndim=2] spatial_layer_coefficients_2d not None,
                     ndarray[np.int64_t, ndim=1] spatial_layer_function_types_1d not None,
                     ndarray[np.int64_t, ndim=2] valid_mask_array not None,
                     ndarray[np.int64_t, ndim=1] change_class_ids not None,
                     ndarray[np.float64_t, ndim=2] hectares_per_grid_cell not None,
                     str output_dir,
                     double reporting_level,
                     str call_string,
                     ):

    cdef size_t n_coarse_rows = coarse_change_matrix_4d.shape[0]
    cdef size_t n_coarse_cols = coarse_change_matrix_4d.shape[1]

    cdef size_t n_fine_rows = input_lulc.shape[0]
    cdef size_t n_fine_cols = input_lulc.shape[1]

    cdef np.int64_t coarse_r, coarse_c, fine_r, fine_c, chunk_r, chunk_c, class_i, class_j, class_l_check, i, j, current_fine_starting_r, current_fine_starting_c, interior_allocation_step, k, regressor_k
    cdef np.int64_t from_class, to_class

    cdef long long counter = 1
    cdef long long print_counter = 0
    cdef long long while_counter = 1
    cdef long long report_time
    cdef long long n_to_sort

    cdef np.int64_t resolution = input_lulc.shape[1] / coarse_change_matrix_4d.shape[1]
    cdef np.int64_t other_resolution = input_lulc.shape[0] / coarse_change_matrix_4d.shape[0]
    if not resolution == other_resolution:
        print ('WARNING, resolutions not amicable.', resolution, other_resolution)

    cdef size_t n_chunk_rows = resolution
    cdef size_t n_chunk_cols = resolution

    cdef size_t n_allocation_classes = coarse_change_matrix_4d.shape[2]

    # cdef np.float64_t coarse_outcome = 0.0
    # cdef np.float64_t fine_outcome = 0.0
    # cdef np.float64_t num_to_allocate_this_class = 0.0
    cdef np.float64_t total_absolute_change_needed = 0.0
    cdef np.float64_t to_add = 0.0

    cdef np.int64_t n_fine_grid_cells_per_coarse_cell = <int> (resolution * resolution)

    cdef np.ndarray[np.int64_t, ndim=2] current_raveled = np.zeros((n_allocation_classes, n_fine_grid_cells_per_coarse_cell), dtype=np.int64)

    # cdef np.ndarray[np.int64_t, ndim=2] current_ranked_rows = np.zeros((n_allocation_classes, n_fine_grid_cells_per_coarse_cell), dtype=np.int64)
    # cdef np.ndarray[np.int64_t, ndim=2] current_ranked_cols = np.zeros((n_allocation_classes, n_fine_grid_cells_per_coarse_cell), dtype=np.int64)


    cdef int write_fine_r = 0
    cdef int write_fine_c = 0

    cdef int rankings_precached = 0
    cdef int print_regressor_modifications = 0

    cdef np.ndarray[np.float64_t, ndim=3] current_to_rank_arrays = np.zeros((n_allocation_classes, resolution, resolution), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=3] current_rank_arrays = np.zeros((n_allocation_classes, resolution, resolution), dtype=np.float64)

    cdef np.ndarray[np.float64_t, ndim=2] current_goals = np.zeros((n_allocation_classes, n_allocation_classes), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] current_goal_left = np.zeros((n_allocation_classes, n_allocation_classes), dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim=1] current_positions = np.zeros(n_allocation_classes, dtype=np.int64)

    cdef np.ndarray[np.float64_t, ndim=3] output_to_rank_arrays = np.zeros((n_allocation_classes, n_fine_rows, n_fine_cols), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=3] output_rank_arrays = np.zeros((n_allocation_classes, n_fine_rows, n_fine_cols), dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim=3] output_change_arrays = np.zeros((n_allocation_classes, n_fine_rows, n_fine_cols), dtype=np.int64)

    cdef np.ndarray[np.int64_t, ndim=2] projected_lulc = np.copy(input_lulc).astype(np.int64)

    if call_string is not '':
        pass
        # print ('Cython call_string: ' + call_string)

    # Get to correct coarse cell
    for coarse_r in range(n_coarse_rows):
        for coarse_c in range(n_coarse_cols):

            # Starting with from_class, consider each to_class allocations.
            for from_class in range(n_allocation_classes):
                report_time = time.time()
                # Check if there's any allocation to do, skipping if not for speed
                total_absolute_change_needed = 0.
                for to_class in range(n_allocation_classes):
                    if to_class != from_class:
                        total_absolute_change_needed += abs(coarse_change_matrix_4d[coarse_r, coarse_c, from_class, to_class])


                for to_class in range(n_allocation_classes):
                    current_goals[from_class, to_class] = coarse_change_matrix_4d[coarse_r, coarse_c, from_class, to_class]

                    # Set diagonal to zero
                    for i in range(n_allocation_classes):
                        current_goals[i, i] = 0

                    # Make copy to increment down. Actually, I'm not sure if this is doing anything because it might just be operating on the buffer.
                    current_goal_left[from_class, to_class] = coarse_change_matrix_4d[coarse_r, coarse_c, from_class, to_class]
                    for i in range(n_allocation_classes):
                        current_goal_left[i, i] = 0

                    # Keep track of where in each from_class the algorithm currently is as it marches down from best to worst suitability cells.
                    current_positions[from_class] = 0

                # TODO Check that this is right.
                if total_absolute_change_needed > 0.0 or True:



                    current_to_rank_arrays = np.zeros((n_allocation_classes, resolution, resolution), dtype=np.float64)
                    current_rank_arrays = np.zeros((n_allocation_classes, resolution, resolution), dtype=np.float64)
                    current_fine_starting_r = coarse_r * resolution
                    current_fine_starting_c = coarse_c * resolution

                    if rankings_precached == 1:
                        print('NYI, start with current_raveled[from_class]= input')

                    else:

                        for regressor_k in range(len(spatial_layers_3d)):

                            if spatial_layer_function_types_1d[regressor_k] == 2 and spatial_layer_coefficients_2d[from_class, regressor_k] != 0:  # Additive

                                current_to_rank_arrays[from_class] += (spatial_layer_coefficients_2d[from_class, regressor_k] *
                                                                    spatial_layers_3d[regressor_k, current_fine_starting_r: current_fine_starting_r + resolution,
                                                                    current_fine_starting_c: current_fine_starting_c + resolution])

                                if print_regressor_modifications:
                                    print('Analyzing regressor ' + str(regressor_k) + ' for from_class ' + str(from_class) + ' with ADDITIVE coefficient ' + str(spatial_layer_coefficients_2d[from_class, regressor_k]) + ' which had mean effect ' + str(np.mean(current_to_rank_arrays[from_class])))

                        # NOTE THE VERY SPECIFIC and potentially strange logic of booleans that actually aren't booleans.
                        # If you have a coefficient of 0, then values with 1 cannot have that cell. But if you try multiplicative
                        # on a continuous, 0 to 1 value we end up subtracting 1 - (0 - 1) * -1 * 0.25 = 0.75. You may want to clarifiy the difference between a
                        # multiplicative binary and multipliciative continuous.
                        for regressor_k in range(len(spatial_layers_3d)):
                            if spatial_layer_function_types_1d[regressor_k] == 1:  # Multiplicative

                                current_to_rank_arrays[from_class] *= 1.0 - (
                                        ((spatial_layer_coefficients_2d[from_class, regressor_k] - 1) * -1.0) *
                                        spatial_layers_3d[regressor_k, current_fine_starting_r: current_fine_starting_r + resolution, current_fine_starting_c: current_fine_starting_c + resolution]
                                )
                                if print_regressor_modifications:
                                    print('Analyzing regressor ' + str(regressor_k) + ' for from_class ' + str(from_class) + ' with MULTIPLICATIVE coefficient ' + str(spatial_layer_coefficients_2d[from_class, regressor_k]) + ' which had mean effect ' + str(np.mean(current_to_rank_arrays[from_class])))


                        # Also mask invalid now
                        current_to_rank_arrays[from_class] *= valid_mask_array[current_fine_starting_r: current_fine_starting_r + resolution, current_fine_starting_c: current_fine_starting_c + resolution]

                        # Set locations that already have from_class as its type to zero
                        current_to_rank_arrays[from_class] *= np.where(input_lulc[current_fine_starting_r: current_fine_starting_r + resolution, current_fine_starting_c: current_fine_starting_c + resolution] == change_class_ids[from_class], 0, 1)

                        # Flip array so that we start from highest value
                        current_to_rank_arrays[from_class] *= -1.0

                        # START HERE: Algorithm is close to working but I think I have an off-by-1 error on my layer matching. Secondly, note that eliminating the following two lines causes the rank
                        # to get strippy. WTF. dig into this.

                        # # Set zeros to high value for ranking purposes
                        current_to_rank_arrays[from_class][current_to_rank_arrays[from_class] == 0] = 9.e9  # NOTE LOGIC, low values ranked first, but areas with ZERO are used as no data, so we don't want them in the rank between pos and neg values.

                        # # Values very close to zero no longer have smooth ranking surfaces due to floating point errors.
                        # current_to_rank_arrays[from_class][(current_to_rank_arrays[from_class] < 1.e-14) & (current_to_rank_arrays[from_class] > -1.e-14)] = 9.e9  # NOTE LOGIC, low values ranked first, but areas with ZERO are used as no data, so we don't want them in the rank between pos and neg values.
                        # current_to_rank_arrays[from_class][current_to_rank_arrays[from_class] < -9999999999999999] = 9.e9  # NOTE, necessary to fix what I think was an underflow error. was getting a lot of 1e-35 numbers.
                        # For visualization, also save to full-extent array
                        output_to_rank_arrays[from_class, current_fine_starting_r: current_fine_starting_r + resolution, current_fine_starting_c: current_fine_starting_c + resolution] = current_to_rank_arrays[from_class]

                        # # Do the ranking
                        # # TODO Review this edit to see if it speeds things up
                        # n_to_sort = len(current_to_rank_arrays[from_class][current_to_rank_arrays[from_class] != 0])
                        #
                        # print('Starting to sort, n_to_sort: ' + str(n_to_sort))
                        # # TODOO Speed up ranking by eliminating nans.
                        # current_raveled[from_class, 0: n_to_sort] = current_to_rank_arrays[from_class][current_to_rank_arrays[from_class] != 0].argsort(axis=None)
                        #
                        current_raveled[from_class] = current_to_rank_arrays[from_class].argsort(axis=None)

                    # TODOO Abandonded start: It is trickier than expected to "rerank"
                    # the rank to eliminate edges because it calls adjacent cells. Need to have a new instantiat9ion of the tiling-system
                    # that includes single-read memory blocking of all 9 cells.
                    # HOWEVER, probably the fastest approach would be to embrace both the clibration (on the fly) calculatinon of everything
                    # but then also a preprocesed version of the allocation for speed. the preprocessin gstep then could do the smoothing.
                    # if smooth_rank:
                    #     current_raveled[from_class] = current_to_rank_arrays[from_class].argsort(axis=None)

                    report_time = time.time()
                    if reporting_level >= 5:  # Write rank arrays
                        counter = 0
                        for i in range(n_fine_grid_cells_per_coarse_cell):
                            if output_to_rank_arrays[from_class, current_fine_starting_r + current_raveled[from_class, i] / resolution, current_fine_starting_c + <int> (current_raveled[from_class, i] % resolution)] <= 999999:
                                current_rank_arrays[from_class, current_raveled[from_class, i] / resolution, <int> (current_raveled[from_class, i] % resolution)] = counter
                                counter += 1
                        output_rank_arrays[from_class, current_fine_starting_r: current_fine_starting_r + resolution, current_fine_starting_c: current_fine_starting_c + resolution] = current_rank_arrays[from_class]

                    # Do the allocation

                    # print('current_goal_left', current_goal_left)
                    for while_counter in range(n_fine_grid_cells_per_coarse_cell):
                        for from_class in range(n_allocation_classes):
                            for to_class in range(n_allocation_classes):

                                # START HERE ELIMINATE DIAGONAL FROM GOAL
                                if current_goal_left[from_class, to_class] > 0:

                                    if current_to_rank_arrays[from_class, <int> (current_raveled[from_class, current_positions[from_class]] / resolution), current_raveled[from_class, current_positions[from_class]] % resolution] < 999999999.0:
                                        # Get current position OVERALL (i.e., including current_fine_starting_x) based on dividing and moduloing the current id.
                                        current_fine_r = current_fine_starting_r + <int> (current_raveled[to_class, current_positions[to_class]] / resolution)
                                        current_fine_c = current_fine_starting_c + current_raveled[to_class, current_positions[to_class]] % resolution

                                        # Write 0-1 to output_change_arrays (3dim) specific to this fine location and this expansion class.
                                        output_change_arrays[to_class, current_fine_r, current_fine_c] = 1

                                        # Write the from_class's label value to the projected lulc map
                                        projected_lulc[current_fine_r, current_fine_c] = change_class_ids[to_class]

                                        # Increment the current allocation position by 1, moving the the next best cell.
                                        # if from_class == 1 and coarse_c == 0 and coarse_r == 0:
                                        #     print('current_goal_left', current_goal_left)
                                        #     print('current_positions', current_positions)
                                        current_positions[to_class] += 1

                                        # Reduce the current class's goal by the amount of hectares in that
                                        current_goal_left[from_class, to_class] -= 1
                                        # current_goal_left[from_class, to_class] -= hectares_per_grid_cell[current_fine_r, current_fine_c]


    if reporting_level >= 11:
        for i in range(n_allocation_classes):
            # hb.show(output_rank_arrays[i], output_path=hb.ruri(os.path.join(output_dir, 'output_rank_for_class_' + str(i) + '.png')), title='output_rank for class ' + str(i))
            hb.save_array_as_geotiff(output_rank_arrays[i], hb.suri(os.path.join(output_dir, 'output_rank_for_class_' + str(i) + '.tif'), call_string), os.path.join(os.path.join(output_dir, 'lulc_baseline.tif')), data_type=7)
    if reporting_level >= 5:
        for i in range(n_allocation_classes):
            # hb.show(np.where(output_to_rank_arrays[i] > 9.e9, np.nan, output_to_rank_arrays[i]), vmin=0, vmax=100, output_path=hb.ruri(os.path.join(output_dir, 'overall_suitability_for_class_' + str(i) + '.png')), data_type = 7, title='overall_suitability for class ' + str(i))
            hb.save_array_as_geotiff(output_to_rank_arrays[i], hb.suri(os.path.join(output_dir, 'output_to_rank_for_class_' + str(i) + '.tif'), call_string), os.path.join(os.path.join(output_dir, 'lulc_baseline.tif')), data_type=7)

    if reporting_level >= 5:
        for i in range(n_allocation_classes):
            hb.save_array_as_geotiff(output_change_arrays[i], hb.suri(os.path.join(output_dir, 'allocations_for_class_' + str(i) + '.tif'), call_string), os.path.join(os.path.join(output_dir, 'lulc_baseline.tif')))

    if reporting_level >= 4:
        hb.save_array_as_geotiff(projected_lulc, hb.suri(os.path.join(output_dir, 'projected_lulc.tif'), call_string), os.path.join(os.path.join(output_dir, 'lulc_baseline.tif')))

    return projected_lulc

@cython.cdivision(False)
@cython.boundscheck(True)
@cython.wraparound(True)
def seals_allocation_gridded_input(ndarray[np.float64_t, ndim=3] coarse_change_3d not None,
                     ndarray[np.int64_t, ndim=2] input_lulc not None,
                     ndarray[np.float64_t, ndim=3] spatial_layers_3d not None,
                     ndarray[np.float64_t, ndim=2] spatial_layer_coefficients_2d not None,
                     ndarray[np.int64_t, ndim=1] spatial_layer_function_types_1d not None,
                     ndarray[np.int64_t, ndim=2] valid_mask_array not None,
                     ndarray[np.int64_t, ndim=1] change_class_ids not None,
                     ndarray[np.float64_t, ndim=2] hectares_per_grid_cell not None,
                     str output_dir,
                     double reporting_level,
                     str call_string,
                     ):

    cdef size_t n_coarse_rows = coarse_change_3d[0].shape[0]
    cdef size_t n_coarse_cols = coarse_change_3d[0].shape[1]

    cdef size_t n_fine_rows = input_lulc.shape[0]
    cdef size_t n_fine_cols = input_lulc.shape[1]

    cdef size_t current_n_ranked = 0

    cdef np.int64_t coarse_r, coarse_c, fine_r, fine_c, chunk_r, chunk_c, class_i, class_j, i, j, current_fine_starting_r, current_fine_starting_c, interior_allocation_step, k, regressor_k

    cdef long long counter = 1
    cdef long long while_counter = 1

    cdef np.int64_t resolution = input_lulc.shape[1] / coarse_change_3d[0].shape[1]
    cdef np.int64_t other_resolution = input_lulc.shape[0] / coarse_change_3d[0].shape[0]
    if not resolution == other_resolution:
        print ('WARNING, resolutions not amicable.', resolution, other_resolution)

    cdef size_t n_chunk_rows = resolution
    cdef size_t n_chunk_cols = resolution

    cdef size_t n_allocation_classes = coarse_change_3d.shape[0]

    cdef np.float64_t coarse_outcome = 0.0
    cdef np.float64_t fine_outcome = 0.0
    cdef np.float64_t num_to_allocate_this_class = 0.0
    cdef np.float64_t total_absolute_change_needed = 0.0



    cdef np.int64_t n_fine_grid_cells_per_coarse_cell = <int> (resolution * resolution)

    cdef np.ndarray[np.int64_t, ndim=2] current_raveled = np.zeros((n_allocation_classes, n_fine_grid_cells_per_coarse_cell), dtype=np.int64)

    cdef np.ndarray[np.int64_t, ndim=2] current_ranked_rows = np.zeros((n_allocation_classes, n_fine_grid_cells_per_coarse_cell), dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=2] current_ranked_cols = np.zeros((n_allocation_classes, n_fine_grid_cells_per_coarse_cell), dtype=np.int64)


    cdef int write_fine_r = 0
    cdef int write_fine_c = 0

    cdef np.ndarray[np.float64_t, ndim=3] current_to_rank_arrays = np.zeros((n_allocation_classes, resolution, resolution), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=3] current_rank_arrays = np.zeros((n_allocation_classes, resolution, resolution), dtype=np.float64)

    cdef np.ndarray[np.float64_t, ndim=1] current_goals = np.zeros(n_allocation_classes, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] current_goal_left = np.zeros(n_allocation_classes, dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim=1] current_positions = np.zeros(n_allocation_classes, dtype=np.int64)

    cdef np.ndarray[np.float64_t, ndim=3] output_to_rank_arrays = np.zeros((n_allocation_classes, n_fine_rows, n_fine_cols), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=3] output_rank_arrays = np.zeros((n_allocation_classes, n_fine_rows, n_fine_cols), dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim=3] output_change_arrays = np.zeros((n_allocation_classes, n_fine_rows, n_fine_cols), dtype=np.int64)

    cdef np.ndarray[np.int64_t, ndim=2] projected_lulc = np.copy(input_lulc).astype(np.int64)

    if call_string is not '':
        pass
        # print ('Cython call_string: ' + call_string)

    # Get to correct coarse cell
    for coarse_r in range(n_coarse_rows):
        for coarse_c in range(n_coarse_cols):

            # First check if there's any allocation to do, skipping if not for speed
            total_absolute_change_needed = 0
            for class_j in range(n_allocation_classes):
                total_absolute_change_needed += abs(coarse_change_3d[class_j, coarse_r, coarse_c])

            if total_absolute_change_needed > 0.0:
                current_to_rank_arrays = np.zeros((n_allocation_classes, resolution, resolution), dtype=np.float64)
                current_rank_arrays = np.zeros((n_allocation_classes, resolution, resolution), dtype=np.float64)
                current_fine_starting_r = coarse_r * resolution
                current_fine_starting_c = coarse_c * resolution

                ## Create the arrays to rank

                # Add in all within-cell regressors
                for class_j in range(n_allocation_classes):
                    # NOTE: we iterate through regressors first for additive, then for multiplicative because the formula really is (a + b + c + d +...) * e * f

                    for regressor_k in range(len(spatial_layers_3d)):
                        if spatial_layer_function_types_1d[regressor_k] == 2 and spatial_layer_coefficients_2d[class_j, regressor_k] != 0: # Additive
                            current_to_rank_arrays[class_j] += (spatial_layer_coefficients_2d[class_j, regressor_k] *
                                                            spatial_layers_3d[regressor_k, current_fine_starting_r: current_fine_starting_r + resolution,
                                                            current_fine_starting_c: current_fine_starting_c + resolution])

                    # NOTE THE VERY SPECIFIC and potentially strange logic of booleans that actually aren't booleans.
                    # If you have a coefficient of 0, then values with 1 cannot have that cell. But if you try multiplicative
                    # on a continuous, 0 to 1 value we end up subtracting 1 - (0 - 1) * -1 * 0.25 = 0.75. You may want to clarifiy the difference between a
                    # multiplicative binary and multipliciative continuous.
                    for regressor_k in range(len(spatial_layers_3d)):
                        if spatial_layer_function_types_1d[regressor_k] == 1: # Multiplicative
                            current_to_rank_arrays[class_j] *= 1.0 - (
                                ((spatial_layer_coefficients_2d[class_j, regressor_k] - 1) * -1.0) *
                                    spatial_layers_3d[regressor_k, current_fine_starting_r: current_fine_starting_r + resolution, current_fine_starting_c: current_fine_starting_c + resolution]
                            )


                    # Also mask invalid now
                    current_to_rank_arrays[class_j] *= valid_mask_array[current_fine_starting_r: current_fine_starting_r + resolution, current_fine_starting_c: current_fine_starting_c + resolution]

                    # Set locations that already have class_j as its type to zero
                    current_to_rank_arrays[class_j] *= np.where(input_lulc[current_fine_starting_r: current_fine_starting_r + resolution, current_fine_starting_c: current_fine_starting_c + resolution] == change_class_ids[class_j], 0, 1)

                    # Flip array so that we start from highest value
                    current_to_rank_arrays[class_j] *= -1.0

                    # Set zeros to high value for ranking purposes
                    current_to_rank_arrays[class_j][current_to_rank_arrays[class_j] == 0] = 9.e9  # NOTE LOGIC, low values ranked first, but areas with ZERO are used as no data, so we don't want them in the rank between pos and neg values.

                    # Values very close to zero no longer have smooth ranking surfaces due to floating point errors.
                    current_to_rank_arrays[class_j][(current_to_rank_arrays[class_j] < 1.e-14) & (current_to_rank_arrays[class_j] > -1.e-14)] = 9.e9  # NOTE LOGIC, low values ranked first, but areas with ZERO are used as no data, so we don't want them in the rank between pos and neg values.
                    current_to_rank_arrays[class_j][current_to_rank_arrays[class_j] < -9999999999999999] = 9.e9 # NOTE, necessary to fix what I think was an underflow error. was getting a lot of 1e-35 numbers.
                    # For visualization, also save to full-extent array
                    output_to_rank_arrays[class_j, current_fine_starting_r: current_fine_starting_r + resolution, current_fine_starting_c: current_fine_starting_c + resolution] = current_to_rank_arrays[class_j]

                # Do the ranking
                for class_j in range(n_allocation_classes):

                    # Sort each class_j array
                    current_raveled[class_j] = current_to_rank_arrays[class_j].argsort(axis=None)

                    # TODOO Abandonded start: It is trickier than expected to "rerank"
                    # the rank to eliminate edges because it calls adjacent cells. Need to have a new instantiat9ion of the tiling-system
                    # that includes single-read memory blocking of all 9 cells.
                    # HOWEVER, probably the fastest approach would be to embrace both the clibration (on the fly) calculatinon of everything
                    # but then also a preprocesed version of the allocation for speed. the preprocessin gstep then could do the smoothing.
                    # if smooth_rank:
                    #     current_raveled[class_j] = current_to_rank_arrays[class_j].argsort(axis=None)


                    if reporting_level >= 5: # Write rank arrays
                        counter = 0
                        for i in range(n_fine_grid_cells_per_coarse_cell):
                            if output_to_rank_arrays[class_j, current_fine_starting_r + current_raveled[class_j, i] / resolution, current_fine_starting_c + <int>(current_raveled[class_j, i] % resolution)] <= 999999:
                                current_rank_arrays[class_j, current_raveled[class_j, i] / resolution, <int> (current_raveled[class_j, i] % resolution)] = counter
                                counter += 1
                        output_rank_arrays[class_j, current_fine_starting_r: current_fine_starting_r + resolution, current_fine_starting_c: current_fine_starting_c + resolution] = current_rank_arrays[class_j]

                # Do the allocation
                for class_j in range(n_allocation_classes):
                    current_goals[class_j] = coarse_change_3d[class_j, coarse_r, coarse_c]
                    current_goal_left[class_j] = current_goals[class_j]
                    current_positions[class_j] = 0
                for while_counter in range(n_fine_grid_cells_per_coarse_cell):
                    for class_j in range(n_allocation_classes):
                        if current_goal_left[class_j] > 0:

                            if current_to_rank_arrays[class_j, <int> (current_raveled[class_j, current_positions[class_j]] / resolution), current_raveled[class_j, current_positions[class_j]] % resolution] < 999999999.0:

                                # Get current position OVERALL (i.e., including current_fine_starting_x) based on dividing and moduloing the current id.
                                current_fine_r = current_fine_starting_r + <int> (current_raveled[class_j, current_positions[class_j]] / resolution)
                                current_fine_c = current_fine_starting_c + current_raveled[class_j, current_positions[class_j]] % resolution

                                # Write 0-1 to output_change_arrays (3dim) specific to this fine location and this expansion class.
                                output_change_arrays[class_j, current_fine_r, current_fine_c] = 1

                                # Write the class_j's label value to the projected lulc map
                                projected_lulc[current_fine_r, current_fine_c] = change_class_ids[class_j]

                                # Increment the current allocation position by 1, moving the the next best cell.
                                current_positions[class_j] += 1

                                # Reduce the current class's goal by the amount of hectares in that
                                current_goal_left[class_j] -= hectares_per_grid_cell[current_fine_r, current_fine_c]

    if reporting_level >= 11:
        for i in range(n_allocation_classes):

            # hb.show(output_rank_arrays[i], output_path=hb.ruri(os.path.join(output_dir, 'output_rank_for_class_' + str(i) + '.png')), title='output_rank for class ' + str(i))
            hb.save_array_as_geotiff(output_rank_arrays[i], hb.suri(os.path.join(output_dir, 'output_rank_for_class_' + str(i) + '.tif'), call_string), os.path.join(os.path.join(output_dir, 'lulc_baseline.tif')), data_type=7)
    if reporting_level >= 5:
        for i in range(n_allocation_classes):
            # hb.show(np.where(output_to_rank_arrays[i] > 9.e9, np.nan, output_to_rank_arrays[i]), vmin=0, vmax=100, output_path=hb.ruri(os.path.join(output_dir, 'overall_suitability_for_class_' + str(i) + '.png')), data_type = 7, title='overall_suitability for class ' + str(i))
            hb.save_array_as_geotiff(output_to_rank_arrays[i], hb.suri(os.path.join(output_dir, 'output_to_rank_for_class_' + str(i) + '.tif'), call_string), os.path.join(os.path.join(output_dir, 'lulc_baseline.tif')), data_type=7)




    if reporting_level >= 5:
        for i in range(n_allocation_classes):
            hb.save_array_as_geotiff(output_change_arrays[i], hb.suri(os.path.join(output_dir, 'allocations_for_class_' + str(i) + '.tif'), call_string), os.path.join(os.path.join(output_dir, 'lulc_baseline.tif')))

            # hb.show(output_change_arrays[i], output_path=hb.ruri(os.path.join(output_dir, 'allocations_for_class_' + str(i) + '.png')), vmin=0, vmax=1, title='allocations for class ' + str(i))
    if reporting_level >= 4:
        hb.save_array_as_geotiff(projected_lulc, hb.suri(os.path.join(output_dir, 'projected_lulc.tif'), call_string), os.path.join(os.path.join(output_dir, 'lulc_baseline.tif')))

        # # hb.show(input_lulc, output_path=hb.ruri(os.path.join(output_dir, 'input_lulc.png')), title='input_lulc', vmin=0, vmax=7, ndv=255)
        # hb.show(projected_lulc, output_path=hb.ruri(os.path.join(output_dir, 'projected_lulc.png')), title='projected_lulc', vmin=0, vmax=7, ndv=255, block_plotting=False)

    return projected_lulc

@cython.cdivision(False)
@cython.boundscheck(True)
@cython.wraparound(True)
def calibrate(ndarray[np.float64_t, ndim=3] coarse_change_3d not None,
                     ndarray[np.int64_t, ndim=2] input_lulc not None,
                     ndarray[np.float64_t, ndim=3] spatial_layers_3d not None,
                     ndarray[np.float64_t, ndim=2] spatial_layer_coefficients_2d not None,
                     ndarray[np.int64_t, ndim=1] spatial_layer_function_types_1d not None,
                     ndarray[np.int64_t, ndim=2] valid_mask_array not None,
                     ndarray[np.int64_t, ndim=1] change_class_ids not None,
                     ndarray[np.int64_t, ndim=2] observed_lulc_array not None,
                     ndarray[np.float64_t, ndim=2] hectares_per_grid_cell not None,
                     str output_dir,
                     double reporting_level,
                     np.float64_t sigma,
                     str call_string,
              ):


    projected_lulc_array = seals_allocation_gridded_input(coarse_change_3d,
                                            input_lulc,
                                            spatial_layers_3d,
                                            spatial_layer_coefficients_2d,
                                            spatial_layer_function_types_1d,
                                            valid_mask_array,
                                            change_class_ids,
                                            hectares_per_grid_cell,
                                            output_dir,
                                            reporting_level,
                                            call_string,)

    overall_similarity_score, overall_similarity_plot, class_similarity_scores, class_similarity_plots = \
        calc_fit_of_projected_against_observed_loss_function(input_lulc, projected_lulc_array, observed_lulc_array, list(change_class_ids), sigma)

    if reporting_level >= 10:
        for i in range(len(class_similarity_plots)):
            hb.save_array_as_geotiff(class_similarity_plots[i], os.path.join(output_dir, 'class_' + str(i) + '_similarity_plot.tif'), os.path.join(os.path.join(output_dir, 'lulc_baseline.tif')), data_type=6)
        hb.save_array_as_geotiff(overall_similarity_plot, os.path.join(output_dir, 'overall_similarity_plot.tif'), os.path.join(os.path.join(output_dir, 'lulc_baseline.tif')), data_type=6)
    return overall_similarity_score, projected_lulc_array, overall_similarity_plot, class_similarity_scores, class_similarity_plots

@cython.cdivision(False)
@cython.boundscheck(True)
@cython.wraparound(True)
def calibrate_exclusive(ndarray[np.float64_t, ndim=3] coarse_change_3d not None,
                     ndarray[np.int64_t, ndim=2] input_lulc not None,
                     ndarray[np.float64_t, ndim=3] spatial_layers_3d not None,
                     ndarray[np.float64_t, ndim=2] spatial_layer_coefficients_2d not None,
                     ndarray[np.int64_t, ndim=1] spatial_layer_function_types_1d not None,
                     ndarray[np.int64_t, ndim=2] valid_mask_array not None,
                     ndarray[np.int64_t, ndim=1] change_class_ids not None,
                     ndarray[np.int64_t, ndim=2] observed_lulc_array not None,
                     ndarray[np.float64_t, ndim=2] hectares_per_grid_cell not None,
                     str output_dir,
                     double reporting_level,
                     np.float64_t sigma,
                     str call_string,
              ):


    projected_lulc_array = seals_allocation_exclusive(coarse_change_3d,
                                            input_lulc,
                                            spatial_layers_3d,
                                            spatial_layer_coefficients_2d,
                                            spatial_layer_function_types_1d,
                                            valid_mask_array,
                                            change_class_ids,
                                            hectares_per_grid_cell,
                                            output_dir,
                                            reporting_level,
                                            call_string,)

    overall_similarity_score, overall_similarity_plot, class_similarity_scores, class_similarity_plots = \
        calc_fit_of_projected_against_observed_loss_function(input_lulc, projected_lulc_array, observed_lulc_array, list(change_class_ids), sigma)

    if reporting_level >= 10:
        for i in range(len(class_similarity_plots)):
            hb.save_array_as_geotiff(class_similarity_plots[i], os.path.join(output_dir, 'class_' + str(i) + '_similarity_plot.tif'), os.path.join(os.path.join(output_dir, 'lulc_baseline.tif')), data_type=6)
        hb.save_array_as_geotiff(overall_similarity_plot, os.path.join(output_dir, 'overall_similarity_plot.tif'), os.path.join(os.path.join(output_dir, 'lulc_baseline.tif')), data_type=6)
    return overall_similarity_score, projected_lulc_array, overall_similarity_plot, class_similarity_scores, class_similarity_plots



@cython.cdivision(False)
@cython.boundscheck(True)
@cython.wraparound(True)
def calibrate_from_change_matrix(ndarray[np.float64_t, ndim=4] coarse_change_matrix_4d not None,
                     ndarray[np.int64_t, ndim=2] input_lulc not None,
                     ndarray[np.float64_t, ndim=3] spatial_layers_3d not None,
                     ndarray[np.float64_t, ndim=2] spatial_layer_coefficients_2d not None,
                     ndarray[np.int64_t, ndim=1] spatial_layer_function_types_1d not None,
                     ndarray[np.int64_t, ndim=2] valid_mask_array not None,
                     ndarray[np.int64_t, ndim=1] change_class_ids not None,
                     ndarray[np.int64_t, ndim=2] observed_lulc_array not None,
                     ndarray[np.float64_t, ndim=2] hectares_per_grid_cell not None,
                     str output_dir,
                     double reporting_level,
                     np.float64_t sigma,
                     str call_string,
              ):


    projected_lulc_array = seals_allocation_from_change_matrix(coarse_change_matrix_4d,
                                            input_lulc,
                                            spatial_layers_3d,
                                            spatial_layer_coefficients_2d,
                                            spatial_layer_function_types_1d,
                                            valid_mask_array,
                                            change_class_ids,
                                            hectares_per_grid_cell,
                                            output_dir,
                                            reporting_level,
                                            call_string,)

    overall_similarity_score, overall_similarity_plot, class_similarity_scores, class_similarity_plots = \
        calc_fit_of_projected_against_observed_loss_function(input_lulc, projected_lulc_array, observed_lulc_array, list(change_class_ids), sigma)

    if reporting_level >= 10:
        for i in range(len(class_similarity_plots)):
            hb.save_array_as_geotiff(class_similarity_plots[i], os.path.join(output_dir, 'class_' + str(i) + '_similarity_plot.tif'), os.path.join(os.path.join(output_dir, 'lulc_baseline.tif')), data_type=6)
        hb.save_array_as_geotiff(overall_similarity_plot, os.path.join(output_dir, 'overall_similarity_plot.tif'), os.path.join(os.path.join(output_dir, 'lulc_baseline.tif')), data_type=6)
    return overall_similarity_score, projected_lulc_array, overall_similarity_plot, class_similarity_scores, class_similarity_plots



def calc_fit_of_projected_against_observed_loss_function(baseline_array, projected_array, observed_array, similarity_class_ids, sigma):
    """Compare allocation success of baseline to projected against some observed using a l2 loss function
    If similarity_class_ids is given, only calculates score based on the given values (otherwise considers all).
    """

    overall_similarity_plot = np.zeros(baseline_array.shape, dtype=np.float64)

    class_similarity_scores = []
    class_similarity_plots = []

    for id in similarity_class_ids:
        similarity_plot = np.zeros(baseline_array.shape, dtype=np.float64)

        baseline_binary = np.where(baseline_array.astype(np.float64) == id, 1.0, 0.0)
        projected_binary = np.where(projected_array.astype(np.float64) == id, 1.0, 0.0)
        observed_binary = np.where(observed_array.astype(np.float64) == id, 1.0, 0.0)

        pb_difference = projected_binary - baseline_binary
        ob_difference = observed_binary - baseline_binary

        pb_diff_counts = hb.enumerate_array_as_odict(pb_difference)
        ob_diff_counts = hb.enumerate_array_as_odict(ob_difference)

        # pb_total_counts = 0
        # for k, v in pb_diff_counts:
        #     if k != 0.0:
        #         pb_total_counts += v
        # pb_total_counts

        pb_expansions = np.where(baseline_binary == 0, projected_binary, 0)
        ob_expansions = np.where(baseline_binary == 0, observed_binary, 0)
        pb_contractions = np.where((baseline_binary == 1) & (projected_binary == 0), 1, 0)
        ob_contractions = np.where((baseline_binary == 1) & (observed_binary == 0), 1, 0)

        pb_expansions_blurred = scipy.ndimage.filters.gaussian_filter(pb_expansions, sigma=sigma)
        ob_expansions_blurred = scipy.ndimage.filters.gaussian_filter(ob_expansions, sigma=sigma)
        pb_contractions_blurred = scipy.ndimage.filters.gaussian_filter(pb_contractions, sigma=sigma)
        ob_contractions_blurred = scipy.ndimage.filters.gaussian_filter(ob_contractions, sigma=sigma)

        l1_gaussian = abs(pb_expansions_blurred - ob_expansions_blurred) + abs(pb_contractions_blurred - ob_contractions_blurred)
        class_similarity_plots.append(l1_gaussian)
        class_similarity_scores.append(np.sum(l1_gaussian))

        overall_similarity_plot += l1_gaussian

    overall_similarity_score = sum(class_similarity_scores)
    return overall_similarity_score, overall_similarity_plot, class_similarity_scores, class_similarity_plots
