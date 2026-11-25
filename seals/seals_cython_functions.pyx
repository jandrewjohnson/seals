# cython: cdivision=True
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#cython: boundscheck=False, wraparound=False
from libc.math cimport log
import hazelbean as hb
import os
import time
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

@cython.cdivision(True)
@cython.boundscheck(True)
@cython.wraparound(False)
def assign_transitions_for_donor(ndarray[np.float64_t, ndim=1] requested not None,
                                 ndarray[np.int64_t, ndim=1] donor_cells not None,
                                 ndarray[np.float64_t, ndim=2] scores_to_rank not None,
                                 ndarray[np.int64_t, ndim=2] ranked_cells not None,
                                 ):
    """Choose which cells of one donor class change, and to which receiver class.

    Each receiver goes down its own ranking of the donor's cells. Receivers that want the same cell are compared by where that
    cell stands in their rankings, because scores of different classes are not on one scale. Equal standing goes to the larger
    request, then to the receiver that loses more by taking its next cell instead, as a share of its own largest score, then
    to the class listed first. A receiver left short then takes a cell from a receiver that can be moved to another one, so the
    number of cells placed is the largest possible. Receivers equally short are served in order of request, then
    the loss from their best to their second cell, then the class listed first. Every cell changes at most once.

    Scaling all scores of a class by a positive number changes nothing, up to rounding and while they stay below the limit for a
    cell that cannot be allocated. Renaming the classes changes nothing unless receivers are equal in every comparison above,
    when the class listed first is served.

    Args:
        requested (array): Cells requested by each receiver class. A fraction counts as a whole cell.
        donor_cells (array): 1 where the cell, in raveled order, holds the donor class.
        scores_to_rank (array): Receiver by raveled cell. Lower is better. 999999999 or more marks a cell that cannot be allocated.
        ranked_cells (array): Receiver by rank. Raveled cells, best first.

    Returns:
        array: Receiver index of each raveled cell, or -1 where nothing changes.
    """
    cdef np.int64_t n_classes = scores_to_rank.shape[0]
    cdef np.int64_t n_cells = scores_to_rank.shape[1]
    cdef np.int64_t receiver, rival, taker, cell, rank, head, tail, best
    cdef int moved

    cdef np.ndarray[np.int64_t, ndim=1] needed = np.ceil(requested).astype(np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] next_rank = np.zeros(n_classes, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] n_candidates = np.zeros(n_classes, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=2] candidates = np.zeros((n_classes, n_cells), dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=2] standing = np.zeros((n_classes, n_cells), dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim=2] loss_if_refused = np.zeros((n_classes, n_cells), dtype=np.float64)
    cdef np.float64_t score_scale, next_score
    cdef np.ndarray[np.int64_t, ndim=1] holder = np.full(n_cells, -1, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] taken_from = np.zeros(n_classes, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] offered_by = np.zeros(n_classes, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] queue = np.zeros(n_classes, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] is_short = np.zeros(n_classes, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] visited = np.zeros(n_classes, dtype=np.int64)

    for receiver in range(n_classes):
        if needed[receiver] > 0:
            for rank in range(n_cells):
                cell = ranked_cells[receiver, rank]
                if donor_cells[cell] == 1 and scores_to_rank[receiver, cell] < 999999999.0:
                    candidates[receiver, n_candidates[receiver]] = cell
                    standing[receiver, cell] = n_candidates[receiver]
                    n_candidates[receiver] += 1
            # With no other cell to take, the next score is zero, the score of a cell that cannot be allocated.
            if n_candidates[receiver] > 0:
                score_scale = max(fabs(scores_to_rank[receiver, candidates[receiver, 0]]),
                                  fabs(scores_to_rank[receiver, candidates[receiver, n_candidates[receiver] - 1]]))
                for rank in range(n_candidates[receiver]):
                    cell = candidates[receiver, rank]
                    next_score = scores_to_rank[receiver, candidates[receiver, rank + 1]] if rank + 1 < n_candidates[receiver] else 0.0
                    loss_if_refused[receiver, cell] = (next_score - scores_to_rank[receiver, cell]) / score_scale

    # Receivers ask for cells in their own order until each is satisfied or has asked for every candidate.
    moved = 1
    while moved:
        moved = 0
        for receiver in range(n_classes):
            while needed[receiver] > 0 and next_rank[receiver] < n_candidates[receiver]:
                cell = candidates[receiver, next_rank[receiver]]
                next_rank[receiver] += 1
                rival = holder[cell]
                if rival >= 0:
                    if standing[receiver, cell] != standing[rival, cell]:
                        if standing[receiver, cell] > standing[rival, cell]:
                            continue
                    elif requested[receiver] != requested[rival]:
                        if requested[receiver] < requested[rival]:
                            continue
                    elif loss_if_refused[receiver, cell] != loss_if_refused[rival, cell]:
                        if loss_if_refused[receiver, cell] < loss_if_refused[rival, cell]:
                            continue
                    elif receiver > rival:
                        continue
                    needed[rival] += 1
                holder[cell] = receiver
                needed[receiver] -= 1
                moved = 1

    # A short receiver can still be served if a receiver holding one of its candidates can move to a free cell.
    while True:
        receiver = -1
        for rival in range(n_classes):
            if needed[rival] > 0 and is_short[rival] == 0 and n_candidates[rival] > 0:
                if receiver >= 0:
                    if needed[rival] != needed[receiver]:
                        if needed[rival] < needed[receiver]:
                            continue
                    elif requested[rival] != requested[receiver]:
                        if requested[rival] < requested[receiver]:
                            continue
                    elif loss_if_refused[rival, candidates[rival, 0]] <= loss_if_refused[receiver, candidates[receiver, 0]]:
                        continue
                receiver = rival
        if receiver < 0:
            break
        visited[:] = 0
        visited[receiver] = 1
        queue[0] = receiver
        head, tail, best = 0, 1, -1
        while head < tail and best < 0:
            taker = queue[head]
            head += 1
            for rank in range(n_candidates[taker]):
                cell = candidates[taker, rank]
                rival = holder[cell]
                if rival < 0:
                    best = cell
                    break
                if visited[rival] == 0:
                    visited[rival] = 1
                    taken_from[rival] = cell
                    offered_by[rival] = taker
                    queue[tail] = rival
                    tail += 1
        if best < 0:
            is_short[receiver] = 1
            continue
        while taker != receiver:
            holder[best] = taker
            best = taken_from[taker]
            taker = offered_by[taker]
        holder[best] = receiver
        needed[receiver] -= 1

    return holder

# NZ_brazil fix (2026-05-28): cdivision=True so int64/int64 returns int (C semantics)
# instead of float64 (Python semantics). Required for line 231's array indexing under
# NumPy 2.x which rejects float scalars as indices. The file-level directive at line 1
# already says cdivision=True; this decorator was an explicit per-function override that
# inverted the file default. Sister function seals_allocation_gridded_input still has the
# False override at line ~265 — may need same fix later if that path is exercised.
@cython.cdivision(True)
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
                     double cython_reporting_level,
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

    cdef np.int64_t resolution = np.int64(input_lulc.shape[1]) / np.int64(coarse_change_matrix_4d.shape[1])
    cdef np.int64_t other_resolution = np.int64(input_lulc.shape[0]) / np.int64(coarse_change_matrix_4d.shape[0])
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

    cdef np.ndarray[np.float64_t, ndim=1] requested = np.zeros(n_allocation_classes, dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim=1] donor_cells = np.zeros(n_fine_grid_cells_per_coarse_cell, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] receiver_of_cell = np.zeros(n_fine_grid_cells_per_coarse_cell, dtype=np.int64)

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

            current_to_rank_arrays = np.zeros((n_allocation_classes, resolution, resolution), dtype=np.float64)
            current_rank_arrays = np.zeros((n_allocation_classes, resolution, resolution), dtype=np.float64)

            # Rank the cells for every class before anything is allocated. In this loop from_class is the class that would gain the cell.
            for from_class in range(n_allocation_classes):
                report_time = time.time()
                # Check if there's any allocation to do, skipping if not for speed
                total_absolute_change_needed = 0.
                for to_class in range(n_allocation_classes):
                    if to_class != from_class:
                        total_absolute_change_needed += abs(coarse_change_matrix_4d[coarse_r, coarse_c, from_class, to_class])


                # TODO Check that this is right.
                if total_absolute_change_needed > 0.0 or True:



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
                                    print ('Analyzing regressor ' + str(regressor_k) + ' for from_class ' + str(from_class) + ' with ADDITIVE coefficient ' + str(spatial_layer_coefficients_2d[from_class, regressor_k]) + ' which had mean effect ' + str(np.mean(current_to_rank_arrays[from_class])))

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
                                    print ('Analyzing regressor ' + str(regressor_k) + ' for from_class ' + str(from_class) + ' with MULTIPLICATIVE coefficient ' + str(spatial_layer_coefficients_2d[from_class, regressor_k]) + ' which had mean effect ' + str(np.mean(current_to_rank_arrays[from_class])))


                        # Also mask invalid now
                        current_to_rank_arrays[from_class] *= valid_mask_array[current_fine_starting_r: current_fine_starting_r + resolution, current_fine_starting_c: current_fine_starting_c + resolution]

                        # Set locations that already have from_class as its type to zero
                        current_to_rank_arrays[from_class] *= np.where(input_lulc[current_fine_starting_r: current_fine_starting_r + resolution, current_fine_starting_c: current_fine_starting_c + resolution] == change_class_ids[from_class], 0, 1)

                        # Flip array so that we start from highest value
                        current_to_rank_arrays[from_class] *= -1.0

                        # TODOOO: Algorithm is close to working but I think I have an off-by-1 error on my layer matching. Secondly, note that eliminating the following two lines causes the rank
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
                        current_raveled[from_class] = current_to_rank_arrays[from_class].argsort(axis=None, kind='stable')

                    # TODOO Abandonded start: It is trickier than expected to "rerank"
                    # the rank to eliminate edges because it calls adjacent cells. Need to have a new instantiat9ion of the tiling-system
                    # that includes single-read memory blocking of all 9 cells.
                    # HOWEVER, probably the fastest approach would be to embrace both the clibration (on the fly) calculatinon of everything
                    # but then also a preprocesed version of the allocation for speed. the preprocessin gstep then could do the smoothing.
                    # if smooth_rank:
                    #     current_raveled[from_class] = current_to_rank_arrays[from_class].argsort(axis=None)

                    report_time = time.time()
                    if cython_reporting_level >= 5:  # Write rank arrays
                        counter = 0
                        for i in range(n_fine_grid_cells_per_coarse_cell):
                            if output_to_rank_arrays[from_class, current_fine_starting_r + current_raveled[from_class, i] / resolution, current_fine_starting_c + current_raveled[from_class, i] % resolution] <= 999999:
                                current_rank_arrays[from_class, current_raveled[from_class, i] / resolution, <int> (current_raveled[from_class, i] % resolution)] = counter
                                counter += 1
                        output_rank_arrays[from_class, current_fine_starting_r: current_fine_starting_r + resolution, current_fine_starting_c: current_fine_starting_c + resolution] = current_rank_arrays[from_class]

            # Do the allocation, one donor class at a time. A donor's cells are its own, so the order of donors does not matter.
            for from_class in range(n_allocation_classes):
                requested = np.array(coarse_change_matrix_4d[coarse_r, coarse_c, from_class])
                requested[from_class] = 0
                if not np.any(requested > 0):
                    continue
                current_lulc = input_lulc[current_fine_starting_r: current_fine_starting_r + resolution,
                                          current_fine_starting_c: current_fine_starting_c + resolution]
                donor_cells = (current_lulc == change_class_ids[from_class]).ravel().astype(np.int64)
                receiver_of_cell = assign_transitions_for_donor(requested, donor_cells,
                                                                current_to_rank_arrays.reshape(n_allocation_classes, -1), current_raveled)
                for i in range(n_fine_grid_cells_per_coarse_cell):
                    to_class = receiver_of_cell[i]
                    if to_class >= 0:
                        current_fine_r = current_fine_starting_r + i / resolution
                        current_fine_c = current_fine_starting_c + i % resolution
                        output_change_arrays[to_class, current_fine_r, current_fine_c] = 1
                        projected_lulc[current_fine_r, current_fine_c] = change_class_ids[to_class]

    if cython_reporting_level >= 11:
        for i in range(n_allocation_classes):
            # hb.show(output_rank_arrays[i], output_path=hb.ruri(os.path.join(output_dir, 'output_rank_for_class_' + str(i) + '.png')), title='output_rank for class ' + str(i))
            hb.save_array_as_geotiff(output_rank_arrays[i], hb.suri(os.path.join(output_dir, 'output_rank_for_class_' + str(i) + '.tif'), call_string), os.path.join(os.path.join(output_dir, 'lulc_baseline.tif')), data_type=7)
    if cython_reporting_level >= 5:
        for i in range(n_allocation_classes):
            # hb.show(np.where(output_to_rank_arrays[i] > 9.e9, np.nan, output_to_rank_arrays[i]), vmin=0, vmax=100, output_path=hb.ruri(os.path.join(output_dir, 'overall_suitability_for_class_' + str(i) + '.png')), data_type = 7, title='overall_suitability for class ' + str(i))
            hb.save_array_as_geotiff(output_to_rank_arrays[i], hb.suri(os.path.join(output_dir, 'output_to_rank_for_class_' + str(i) + '.tif'), call_string), os.path.join(os.path.join(output_dir, 'lulc_baseline.tif')), data_type=7)

    if cython_reporting_level >= 5:
        for i in range(n_allocation_classes):
            hb.save_array_as_geotiff(output_change_arrays[i], hb.suri(os.path.join(output_dir, 'allocations_for_class_' + str(i) + '.tif'), call_string), os.path.join(os.path.join(output_dir, 'lulc_baseline.tif')))

    if cython_reporting_level >= 4:
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
                     ndarray[np.int64_t, ndim=1] changing_class_indices not None,
                     list changing_class_labels not None, # Not speed dependent
                     ndarray[np.float64_t, ndim=2] hectares_per_grid_cell not None,
                     str output_dir,
                     double cython_reporting_level,
                     np.int64_t allow_contracting,
                     str output_match_path,
                     str call_string,
                     ):



    cdef size_t n_coarse_rows = coarse_change_3d[0].shape[0]
    cdef size_t n_coarse_cols = coarse_change_3d[0].shape[1]

    cdef size_t n_fine_rows = input_lulc.shape[0]
    cdef size_t n_fine_cols = input_lulc.shape[1]

    cdef size_t current_n_ranked = 0

    cdef np.int64_t coarse_r, coarse_c, class_position, fine_r, fine_c, chunk_r, chunk_c, class_i, class_j, i, j, current_fine_starting_r, current_fine_starting_c, interior_allocation_step, k, regressor_k
    cdef np.int64_t class_displaced, class_displaced_position, max_class_index
    cdef long long counter = 1
    cdef long long while_counter = 1

    cdef np.int64_t resolution = input_lulc.shape[1] / coarse_change_3d[0].shape[1]
    cdef np.int64_t other_resolution = input_lulc.shape[0] / coarse_change_3d[0].shape[0]
    if not resolution == other_resolution:
        print ('WARNING, resolutions not amicable.', resolution, other_resolution)

    cdef size_t n_allocation_classes = coarse_change_3d.shape[0]

    cdef np.float64_t total_absolute_change_needed = 0.0

    cdef np.int64_t n_fine_grid_cells_per_coarse_cell = <int> (resolution * resolution)

    cdef np.ndarray[np.int64_t, ndim=2] current_raveled = np.zeros((n_allocation_classes, n_fine_grid_cells_per_coarse_cell), dtype=np.int64)

    cdef np.ndarray[np.float64_t, ndim=3] current_to_rank_arrays = np.zeros((n_allocation_classes, resolution, resolution), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=3] current_rank_arrays = np.zeros((n_allocation_classes, resolution, resolution), dtype=np.float64)

    cdef np.ndarray[np.float64_t, ndim=1] current_goals = np.zeros(n_allocation_classes, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] current_goal_left = np.zeros(n_allocation_classes, dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim=1] current_positions = np.zeros(n_allocation_classes, dtype=np.int64)

    cdef np.ndarray[np.float64_t, ndim=3] output_to_rank_arrays = np.zeros((n_allocation_classes, n_fine_rows, n_fine_cols), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=3] output_rank_arrays = np.zeros((n_allocation_classes, n_fine_rows, n_fine_cols), dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim=3] output_change_arrays = np.zeros((n_allocation_classes, n_fine_rows, n_fine_cols), dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=2] change_happened = np.zeros((n_fine_rows, n_fine_cols), dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=2] projected_lulc = np.copy(input_lulc).astype(np.int64)

    
    cdef np.ndarray[np.int64_t, ndim=1] changing_class_positions = np.arange(n_allocation_classes, dtype=np.int64)
    
    max_class_index = np.max(changing_class_indices) + 1
    cdef np.ndarray[np.int64_t, ndim=1] class_indices_to_class_positions = np.zeros(max_class_index + 1, dtype=np.int64)
    
    # A little surprsing here, but this is an optimization to avoid a dictionary lookup. However, 
    # it assumes that the max_class_index will be smallish
    counter = 0
    for i in changing_class_indices:
        class_indices_to_class_positions[i] = counter
        counter += 1

    
    if call_string is not '': 
        pass
        # print ('Cython call_string: ' + call_string)  

    # Get to correct coarse cell
    for coarse_r in range(n_coarse_rows):
        for coarse_c in range(n_coarse_cols):  
            
            # First check if there's any allocation to do, skipping if not for speed
            for class_position in changing_class_positions:
                total_absolute_change_needed += abs(coarse_change_3d[class_position, coarse_r, coarse_c])

            if total_absolute_change_needed > 0.0:
                current_to_rank_arrays = np.zeros((n_allocation_classes, resolution, resolution), dtype=np.float64)
                current_rank_arrays = np.zeros((n_allocation_classes, resolution, resolution), dtype=np.float64)
                current_fine_starting_r = coarse_r * resolution
                current_fine_starting_c = coarse_c * resolution

                ## Create the arrays to rank

                # Add in all within-cell regressors
                for class_position in changing_class_positions: 

                    # NOTE: we iterate through regressors first for additive, then for multiplicative because the formula really is (a + b + c + d +...) * e * f
                    for regressor_k in range(len(spatial_layers_3d)):
                        if spatial_layer_function_types_1d[regressor_k] == 2 and spatial_layer_coefficients_2d[class_position, regressor_k] != 0: # Additive
                            current_to_rank_arrays[class_position] += (spatial_layer_coefficients_2d[class_position, regressor_k] *
                                                            spatial_layers_3d[regressor_k, current_fine_starting_r: current_fine_starting_r + resolution,
                                                            current_fine_starting_c: current_fine_starting_c + resolution])

                    # NOTE THE VERY SPECIFIC and potentially strange logic of booleans that actually aren't booleans.
                    # If you have a coefficient of 0, then values with 1 cannot have that cell. But if you try multiplicative
                    # on a continuous, 0 to 1 value we end up subtracting 1 - (0 - 1) * -1 * 0.25 = 0.75. You may want to clarifiy the difference between a
                    # multiplicative binary and multipliciative continuous.
                    for regressor_k in range(len(spatial_layers_3d)):
                        if spatial_layer_function_types_1d[regressor_k] == 1:
                            
                            current_to_rank_arrays[class_position] *= 1.0 - (
                                ((spatial_layer_coefficients_2d[class_position, regressor_k] - 1) * -1.0) *
                                    spatial_layers_3d[regressor_k, current_fine_starting_r: current_fine_starting_r + resolution, current_fine_starting_c: current_fine_starting_c + resolution]
                            )


                    # Also mask invalid now
                    current_to_rank_arrays[class_position] *= valid_mask_array[current_fine_starting_r: current_fine_starting_r + resolution, current_fine_starting_c: current_fine_starting_c + resolution]

                    # Set locations that already have class_position as its type to zero
                    current_to_rank_arrays[class_position] *= np.where(input_lulc[current_fine_starting_r: current_fine_starting_r + resolution, current_fine_starting_c: current_fine_starting_c + resolution] == changing_class_indices[class_position], 0, 1)

                    # Flip array so that we start from highest value
                    # TODOO this could be optimized by changing the inputs
                    current_to_rank_arrays[class_position] *= -1.0

                    # Set zeros to high value for ranking purposes
                    current_to_rank_arrays[class_position][current_to_rank_arrays[class_position] == 0] = 9.e9  # NOTE LOGIC, low values ranked first, but areas with ZERO are used as no data, so we don't want them in the rank between pos and neg values.

                    # Values very close to zero no longer have smooth ranking surfaces due to floating point errors.
                    current_to_rank_arrays[class_position][(current_to_rank_arrays[class_position] < 1.e-14) & (current_to_rank_arrays[class_position] > -1.e-14)] = 9.e9  # NOTE LOGIC, low values ranked first, but areas with ZERO are used as no data, so we don't want them in the rank between pos and neg values.
                    current_to_rank_arrays[class_position][current_to_rank_arrays[class_position] < -9999999999999999] = 9.e9 # NOTE, necessary to fix what I think was an underflow error. was getting a lot of 1e-35 numbers.
                    # For visualization, also save to full-extent array
                    output_to_rank_arrays[class_position, current_fine_starting_r: current_fine_starting_r + resolution, current_fine_starting_c: current_fine_starting_c + resolution] = current_to_rank_arrays[class_position]

                # Do the ranking
                for class_position in changing_class_positions:

                    # Sort each class_position array
                    current_raveled[class_position] = current_to_rank_arrays[class_position].argsort(axis=None)

                    # TODOO Abandonded start: It is trickier than expected to "rerank"
                    # the rank to eliminate edges because it calls adjacent cells. Need to have a new instantiat9ion of the tiling-system
                    # that includes single-read memory blocking of all 9 cells.
                    # HOWEVER, probably the fastest approach would be to embrace both the clibration (on the fly) calculatinon of everything
                    # but then also a preprocesed version of the allocation for speed. the preprocessin gstep then could do the smoothing.
                    # if smooth_rank:
                    #     current_raveled[class_position] = current_to_rank_arrays[class_position].argsort(axis=None)



                    if cython_reporting_level >= 5: # Write rank arrays
                        counter = 0
                        for i in range(n_fine_grid_cells_per_coarse_cell):

                            if output_to_rank_arrays[class_position, current_fine_starting_r + current_raveled[class_position, i] / resolution, current_fine_starting_c + current_raveled[class_position, i] % resolution] <= 999999:
                                current_rank_arrays[class_position, current_raveled[class_position, i] / resolution, <int> (current_raveled[class_position, i] % resolution)] = counter
                                counter += 1
                        output_rank_arrays[class_position, current_fine_starting_r: current_fine_starting_r + resolution, current_fine_starting_c: current_fine_starting_c + resolution] = current_rank_arrays[class_position]

                # Do the allocation
                for class_position in changing_class_positions:
                    current_goals[class_position] = coarse_change_3d[class_position, coarse_r, coarse_c]
                    current_goal_left[class_position] = current_goals[class_position]
                    current_positions[class_position] = 0

                for while_counter in range(n_fine_grid_cells_per_coarse_cell):
                    for class_position in changing_class_positions:
                        if current_goal_left[class_position] > 0:

                            block_fine_r = <int> (current_raveled[class_position, current_positions[class_position]] / resolution)
                            block_fine_c = <int> (current_raveled[class_position, current_positions[class_position]] % resolution)                        

                            current_fine_r = current_fine_starting_r + block_fine_r
                            current_fine_c = current_fine_starting_c + block_fine_c

                            class_displaced = input_lulc[current_fine_r, current_fine_c]
                            if class_displaced in changing_class_indices:
                                class_displaced_position = class_indices_to_class_positions[class_displaced]
                            else:
                                class_displaced_position = -1

                            #if while_counter % 10000 == 0:
                            #    print('checking if change_happened', current_fine_r, current_fine_c, np.sum(change_happened), change_happened[current_fine_r, current_fine_c], 'current goal', current_goal_left[class_position] )
                            if current_to_rank_arrays[class_position, block_fine_r, block_fine_c] < 999999999.0 and change_happened[current_fine_r, current_fine_c] == 0 and class_displaced_position != class_position:

                                # Record where change has happened. This is used
                                change_happened[current_fine_r, current_fine_c] = 1

                                # Write 0-1 to output_change_arrays (3dim) specific to this fine location and this expansion class.
                                output_change_arrays[class_position, current_fine_r, current_fine_c] = 1

                                # Write the class_position's label value to the projected lulc map
                                projected_lulc[current_fine_r, current_fine_c] = changing_class_indices[class_position]

                                # Reduce the current class's goal by the amount of hectares in that
                                current_goal_left[class_position] -= hectares_per_grid_cell[current_fine_r, current_fine_c]

                                # If a class that also was supposed to expand got booted from the cell, increment it's goal left UP
                                if allow_contracting:
                                    if current_goal_left[class_displaced_position] > 0:
                                        current_goal_left[class_displaced_position] += hectares_per_grid_cell[current_fine_r, current_fine_c]
                                        #print('class ' + str(class_position) + ': ' + changing_class_labels[class_position] + ' replaced EXPANDING class ' + str(class_displaced_position) + ': ' + changing_class_labels[class_displaced_position]  + ' who had goals of ' + str(current_goal_left[class_position]) + ' and ' + str(current_goal_left[class_displaced_position ]) + ' left at location ' + str(current_fine_r) + ' ' + str(current_fine_c))
                                    #else:
                                        #print('class ' + str(class_position) + ': ' + changing_class_labels[class_position] + ' replaced CONTRACTING class ' + str(class_displaced_position) + ': ' + changing_class_labels[class_displaced_position]  + ' who had goals of ' + str(current_goal_left[class_position]) + ' and ' + str(current_goal_left[class_displaced_position ]) + ' left at location ' + str(current_fine_r) + ' ' + str(current_fine_c))
                            
                            # WHETHER OR NOT IT MADE THE CHANGE, 
                            # Increment the current allocation position by 1, moving the the next best cell.
                            current_positions[class_position] += 1

    if cython_reporting_level >= 11:
        for i in changing_class_positions:

            # hb.show(output_rank_arrays[i], output_path=hb.ruri(os.path.join(output_dir, 'output_rank_for_class_' + str(i) + '.png')), title='output_rank for class ' + str(i))
            hb.save_array_as_geotiff(output_rank_arrays[i], hb.suri(os.path.join(output_dir, 'output_rank_for_class_' + changing_class_labels[i] + '.tif'), call_string), output_match_path, data_type=7)
    if cython_reporting_level >= 5:
        for i in changing_class_positions:
            # hb.show(np.where(output_to_rank_arrays[i] > 9.e9, np.nan, output_to_rank_arrays[i]), vmin=0, vmax=100, output_path=hb.ruri(os.path.join(output_dir, 'overall_suitability_for_class_' + str(i) + '.png')), data_type = 7, title='overall_suitability for class ' + str(i))
            hb.save_array_as_geotiff(output_to_rank_arrays[i], hb.suri(os.path.join(output_dir, 'output_to_rank_for_class_' + changing_class_labels[i] + '.tif'), call_string), output_match_path, data_type=7)


    return projected_lulc, output_change_arrays, change_happened

@cython.cdivision(False)
@cython.boundscheck(True)
@cython.wraparound(True)
def calibrate(ndarray[np.float64_t, ndim=3] coarse_change_3d not None,
                     ndarray[np.int64_t, ndim=2] input_lulc not None,
                     ndarray[np.float64_t, ndim=3] spatial_layers_3d not None,
                     ndarray[np.float64_t, ndim=2] spatial_layer_coefficients_2d not None,
                     ndarray[np.int64_t, ndim=1] spatial_layer_function_types_1d not None,
                     ndarray[np.int64_t, ndim=2] valid_mask_array not None,
                     ndarray[np.int64_t, ndim=1] changing_class_indices not None,
                     list changing_class_labels not None,
                     ndarray[np.int64_t, ndim=2] observed_lulc_array not None,
                     ndarray[np.float64_t, ndim=2] hectares_per_grid_cell not None,
                     str output_dir,
                     double cython_reporting_level,
                     np.int64_t allow_contracting,
                     np.float64_t sigma,
                     str output_match_path,
                     str call_string,
              ):


    projected_lulc_array, output_change_arrays, change_happened = seals_allocation_gridded_input(coarse_change_3d,
                                            input_lulc,
                                            spatial_layers_3d,
                                            spatial_layer_coefficients_2d,
                                            spatial_layer_function_types_1d,
                                            valid_mask_array,
                                            changing_class_indices,
                                            changing_class_labels,
                                            hectares_per_grid_cell,
                                            output_dir,
                                            cython_reporting_level,
                                            allow_contracting,
                                            output_match_path,
                                            call_string,)

    overall_similarity_score, overall_similarity_plot, class_similarity_scores, class_similarity_plots = \
        calc_fit_of_projected_against_observed_loss_function(input_lulc, projected_lulc_array, observed_lulc_array, list(changing_class_indices), sigma)

    if cython_reporting_level >= 10:
        for i in range(len(class_similarity_plots)):
            hb.save_array_as_geotiff(class_similarity_plots[i], os.path.join(output_dir, 'similarity_plot_class_' + changing_class_labels[i] + '.tif'), output_match_path, data_type=6)
        hb.save_array_as_geotiff(overall_similarity_plot, os.path.join(output_dir, 'similarity_overall.tif'), output_match_path, data_type=6)
    return overall_similarity_score, projected_lulc_array, overall_similarity_plot, class_similarity_scores, class_similarity_plots, output_change_arrays, change_happened


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
                     double cython_reporting_level,
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
                                            cython_reporting_level,
                                            call_string,)

    overall_similarity_score, overall_similarity_plot, class_similarity_scores, class_similarity_plots = \
        calc_fit_of_projected_against_observed_loss_function(input_lulc, projected_lulc_array, observed_lulc_array, list(change_class_ids), sigma)

    if cython_reporting_level >= 10:
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

#
# # cython: cdivision=True
# # define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
# #cython: boundscheck=False, wraparound=False
# from libc.math cimport log
# import hazelbean as hb
# import time
# from collections import OrderedDict
# from cython.parallel cimport prange
# import cython
# cimport cython
# import numpy as np  # NOTE, both imports are required. cimport adds extra information to the pyd while the import actually defines numppy
# cimport numpy as np
# from numpy cimport ndarray
# from libc.math cimport sin
# from libc.math cimport fabs
# import math, time
# from cython.view cimport array as cvarray
#

@cython.cdivision(False)
@cython.boundscheck(True)
@cython.wraparound(True)
def write_carbon_table_to_array(
        ndarray[np.float32_t, ndim=2] lulc_array not None,  # NOTE Funny float usage
        ndarray[np.float32_t, ndim=2] carbon_zones_array not None,  # NOTE Funny float usage
        ndarray[np.float32_t, ndim=2] lookup_table not None,
        dict row_names,
        dict col_names):
    cdef long long cr, cc
    cdef double c_lulc_class, c_carbon_zone  # These are doubles to match float32
    cdef long long n_rows = lulc_array.shape[0]
    cdef long long n_cols = lulc_array.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2] output_carbon = np.zeros((n_rows, n_cols), dtype=np.float32)

    start = time.time()
    # print('starting cython loop', start)
    for cr in range(n_rows):
        # if cr % 10 == 0:
            # print(cr)
        for cc in range(n_cols):
            c_lulc_class = lulc_array[cr, cc]
            c_carbon_zone = carbon_zones_array[cr, cc]
            if c_carbon_zone > 0 and c_lulc_class > 0:
                # print('c_lulc_class', c_lulc_class)
                # print('c_carbon_zone', c_carbon_zone)
                # print('c_lulc_class[cr, cc]', c_lulc_class)
                # lookup_r_id =col_names[c_lulc_class]
                # print('c_carbon_zone c_lulc_class', c_carbon_zone, c_lulc_class)
                # print('row_names[c_carbon_zone], col_names[c_lulc_class]', row_names[c_carbon_zone], col_names[c_lulc_class])
                try:
                    output_carbon[cr, cc] = lookup_table[row_names[c_carbon_zone], col_names[c_lulc_class]]
                except:
                    pass
    # print('ending cython loop', str(time.time() - start))
    return output_carbon
