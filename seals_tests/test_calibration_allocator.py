"""The allocator that calibration uses must place the transitions it is asked for, from the classes it is asked to take them from."""
import itertools

import numpy as np
import pytest

from seals import seals_cython_functions

HALVES = np.repeat([1, 2], 8).reshape(4, 4)
ROWS = np.array([[1] * 4, [2] * 4, [3] * 4, [3] * 4])
RAMP = np.arange(16.).reshape(1, 4, 4) / 16 + .1


def allocate(requests, lulc, layers, coefficients=None, valid=None, blocks=1):
    """Allocate requests, either {(donor id, receiver id): cells} or a full change matrix. Every layer is additive."""
    n = int(lulc.max()) if coefficients is None else len(coefficients)
    matrix = requests
    if isinstance(requests, dict):
        matrix = np.zeros((blocks, blocks, n, n))
        for (donor, receiver), cells in requests.items():
            matrix[0, 0, donor - 1, receiver - 1] = cells
    return seals_cython_functions.seals_allocation_from_change_matrix(
        matrix.copy(), lulc.astype(np.int64), layers, np.ones((n, len(layers))) if coefficients is None else coefficients, np.full(len(layers), 2, np.int64),
        np.ones(lulc.shape, np.int64) if valid is None else valid, np.arange(1, n + 1), np.ones(lulc.shape), '', 0.0, '')


def count_flows(lulc, projected, n):
    flows = np.array([[((lulc == i) & (projected == j)).sum() for j in range(1, n + 1)] for i in range(1, n + 1)])
    return flows * (1 - np.eye(n, dtype=int))


def make_random_case(rng, n, resolution=6, blocks=2, demand=.6):
    size = resolution * blocks
    lulc, layers = rng.integers(1, n + 1, (size, size)), rng.random((4, size, size)) + .01
    coefficients, valid = rng.random((n, 4)) + .01, (rng.random((size, size)) > .1).astype(np.int64)
    matrix = np.zeros((blocks, blocks, n, n))
    for r, c, donor in itertools.product(range(blocks), range(blocks), range(n)):
        block = np.s_[r * resolution:(r + 1) * resolution, c * resolution:(c + 1) * resolution]
        wanted = rng.multinomial(int(((lulc[block] == donor + 1) & (valid[block] == 1)).sum() * demand), np.ones(n) / n)
        wanted[donor] = 0
        matrix[r, c, donor] = wanted
    return matrix, lulc, layers, coefficients, valid


@pytest.mark.parametrize('lulc, requests', [
    (HALVES, {(1, 2): 3}), (HALVES, {(2, 1): 3}), (HALVES, {(1, 2): 8, (2, 1): 8}),
    (ROWS, {(2, 1): 3}), (ROWS, {(1, 3): 3}), (ROWS, {(2, 3): 3}), (ROWS, {(2, 1): 2, (3, 1): 2, (3, 2): 2})])
def test_requests_are_placed_from_the_requested_class(lulc, requests):
    flows = count_flows(lulc, allocate(requests, lulc, RAMP), lulc.max())
    assert {(i + 1, j + 1): v for (i, j), v in np.ndenumerate(flows) if v} == requests


@pytest.mark.parametrize('seed', range(10))
def test_every_coarse_cell_gets_its_own_flows(seed):
    """Rankings must not carry over from one coarse cell to the next, and masked cells must not change."""
    rng = np.random.default_rng(seed)
    n = 2 + seed % 4
    matrix, lulc, layers, coefficients, valid = make_random_case(rng, n)
    projected = allocate(matrix, lulc, layers, coefficients, valid)
    for r, c in itertools.product(range(2), range(2)):
        block = np.s_[r * 6:(r + 1) * 6, c * 6:(c + 1) * 6]
        assert np.array_equal(count_flows(lulc[block], projected[block], n), matrix[r, c])
    assert (projected == lulc)[valid == 0].all()


@pytest.mark.parametrize('seed', range(10))
def test_renaming_the_classes_does_not_change_the_map(seed):
    """Holds because no two receivers tie here. Ties are covered below."""
    rng = np.random.default_rng(seed)
    n = 2 + seed % 4
    matrix, lulc, layers, coefficients, valid = make_random_case(rng, n)
    new_index = rng.permutation(n)
    renamed_matrix, renamed_coefficients = np.zeros_like(matrix), np.zeros_like(coefficients)
    renamed_matrix[:, :, new_index[:, None], new_index[None, :]] = matrix
    renamed_coefficients[new_index] = coefficients
    projected = allocate(matrix, lulc, layers, coefficients, valid)
    renamed = allocate(renamed_matrix, new_index[lulc - 1] + 1, layers, renamed_coefficients, valid)
    assert np.array_equal(new_index[projected - 1] + 1, renamed)


def test_calls_are_repeatable_and_leave_inputs_alone():
    case = make_random_case(np.random.default_rng(3), 4)
    before = [a.copy() for a in case]
    first = allocate(*case)
    assert np.array_equal(first, allocate(*case))
    assert all(np.array_equal(a, b) for a, b in zip(case, before))


def two_receivers(scores_2, scores_3):
    """One row of class 1 cells and the scores classes 2 and 3 give them. Rows of 2 and 3 fill the square."""
    k = len(scores_2)
    lulc = np.full((k, k), 2)
    lulc[0] = 1
    layers = np.zeros((2, k, k))
    layers[0, 0], layers[1, 0] = scores_2, scores_3
    return lulc, layers, np.array([[0., 0.], [1., 0.], [0., 1.]])


def test_a_contested_cell_goes_to_the_receiver_that_ranks_it_higher_whatever_the_scale():
    lulc, layers, coefficients = two_receivers([9, 8, 1], [7, 9, 8])
    for scale in (1., 1e5):
        coefficients[2, 1] = scale
        projected = allocate({(1, 2): 2, (1, 3): 1}, lulc, layers, coefficients)
        assert projected[0].tolist() == [2, 3, 2]


def test_equal_standing_goes_to_the_larger_request_then_the_larger_loss_then_the_class_listed_first():
    larger = allocate({(1, 2): 1, (1, 3): 2}, *two_receivers([9, 5, 1], [9, 5, 1]))
    assert larger[0].tolist() == [3, 3, 2]
    # Class 3 loses seven ninths of its best score by moving on, class 2 four ninths of its own.
    loses_more = allocate({(1, 2): 1, (1, 3): 1}, *two_receivers([9, 5, 1], [9, 2, 1]))
    assert loses_more[0].tolist() == [3, 2, 1]
    exact = allocate({(1, 2): 1, (1, 3): 1}, *two_receivers([9, 5, 1], [9, 5, 1]))
    assert exact[0].tolist() == [2, 3, 1]


def test_ties_do_not_depend_on_the_scale_of_a_class():
    """A higher raw score must not decide a tie: class 3 has the larger loss at every scale."""
    lulc, layers, coefficients = two_receivers([9, 5, 1], [9, 2, 1])
    for scale in (2. ** -20, 1., 2. ** 20):
        coefficients[1, 0] = scale
        assert allocate({(1, 2): 1, (1, 3): 1}, lulc, layers, coefficients)[0].tolist() == [3, 2, 1]


def test_equal_scores_are_taken_in_raster_order():
    flat = np.ones((1, 4, 4))
    assert allocate({(1, 2): 3}, HALVES, flat)[0].tolist() == [2, 2, 2, 1]


def test_a_request_is_not_stranded_when_another_receiver_can_move():
    """Class 2 ranks the first cell best and could use the second. Class 3 can only use the first."""
    lulc, layers, coefficients = two_receivers([9, 8, 0], [1, 0, 0])
    assert allocate({(1, 2): 1, (1, 3): 1}, lulc, layers, coefficients)[0].tolist() == [3, 2, 1]


@pytest.mark.parametrize('seed', range(40))
def test_uneven_eligibility_places_as_many_cells_as_any_assignment_could(seed):
    from scipy.optimize import linear_sum_assignment
    rng = np.random.default_rng(seed)
    n, k = 4, 5
    lulc = np.full((k, k), 1)
    layers = rng.random((n, k, k)) * (rng.random((n, k, k)) > .7)
    coefficients = np.eye(n)
    wanted = np.array([0, *rng.integers(0, 9, n - 1)])
    matrix = np.zeros((1, 1, n, n))
    matrix[0, 0, 0] = wanted
    projected = allocate(matrix, lulc, layers, coefficients)
    eligible = np.repeat(layers.reshape(n, -1) > 0, wanted, axis=0)
    rows, columns = linear_sum_assignment(-eligible.astype(float))
    assert (projected != lulc).sum() == eligible[rows, columns].sum()
    for receiver in range(1, n):
        taken = projected == receiver + 1
        assert taken.sum() <= wanted[receiver] and (layers[receiver][taken] > 0).all()


def make_sparse_case(rng, n=4, k=4, share=.3):
    """One donor class covering the tile, receivers with few eligible cells each, strictly distinct scores."""
    layers = rng.random((n, k, k)) * (rng.random((n, k, k)) < share)
    layers[0] = 0
    wanted = np.array([0, *rng.integers(1, 5, n - 1)])
    eligible = np.repeat(layers.reshape(n, -1) > 0, wanted, axis=0)
    return layers, wanted, eligible


def place_without_moving_receivers(layers, wanted):
    """The first phase alone, written independently: refused receivers ask on, nobody is moved."""
    flat, n_cells = layers.reshape(len(layers), -1), layers[0].size
    order = [[c for c in np.argsort(-f, kind='stable') if f[c] > 0] for f in flat]
    held = {}
    queue = [j for j in range(len(wanted)) for _ in range(wanted[j])]
    asked = [0] * len(wanted)
    while queue:
        j = queue.pop()
        while asked[j] < len(order[j]):
            cell = order[j][asked[j]]
            asked[j] += 1
            rival = held.get(cell)
            if rival is None or (order[rival].index(cell), -wanted[rival]) > (order[j].index(cell), -wanted[j]):
                held[cell] = j
                if rival is not None:
                    queue.append(rival)
                break
    return len(held)


def allocate_sparse(layers, wanted):
    n, k = len(wanted), layers.shape[1]
    matrix = np.zeros((1, 1, n, n))
    matrix[0, 0, 0] = wanted
    return allocate(matrix, np.ones((k, k), dtype=int), layers, np.eye(n))


@pytest.fixture(scope='module')
def sparse_cases():
    """Random cases, with the most cells any assignment could place."""
    from scipy.optimize import linear_sum_assignment
    rng, cases = np.random.default_rng(0), []
    for _ in range(400):
        layers, wanted, eligible = make_sparse_case(rng)
        rows, columns = linear_sum_assignment(-eligible.astype(float))
        cases.append((layers, wanted, int(eligible[rows, columns].sum())))
    return cases


def rename(layers, wanted, order):
    """Reorder the receivers. The donor stays first."""
    return layers[order], wanted[order]


def test_the_sparse_cases_need_receivers_to_be_moved(sparse_cases):
    short = [best - place_without_moving_receivers(layers, wanted) for layers, wanted, best in sparse_cases]
    assert sum(s > 0 for s in short) >= 10 and sum(s > 0 and best == wanted.sum() for s, (_, wanted, best) in zip(short, sparse_cases)) >= 5


@pytest.mark.parametrize('feasible', [True, False])
def test_sparse_eligibility_is_unaffected_by_class_order(sparse_cases, feasible):
    """Covers requests that can all be met, and those that cannot, where which receiver goes short is itself a choice."""
    n_checked = 0
    for number, (layers, wanted, best) in enumerate(sparse_cases):
        if (best == wanted.sum()) != feasible:
            continue
        n_checked += 1
        projected = allocate_sparse(layers, wanted)
        assert (projected != 1).sum() == best
        # A single candidate says nothing on a relative scale, so two such receivers with equal requests are an exact tie.
        single = [wanted[j] for j in range(1, 4) if (layers[j] > 0).sum() == 1]
        if len(set(single)) < len(single):
            continue
        for order in itertools.permutations([1, 2, 3]):
            order = np.array([0, *order])
            renamed = allocate_sparse(*rename(layers, wanted, order))
            assert np.array_equal(order[renamed - 1], projected - 1), number
    assert n_checked >= 20


def test_exact_symmetry_is_settled_by_class_order_the_same_way_every_time():
    """Two receivers alike in every comparison, one eligible cell. The class listed first gets it, so renaming does change the map here."""
    layers = np.zeros((3, 2, 2))
    layers[1:, 0, 0] = .5
    wanted = np.array([0, 1, 1])
    assert allocate_sparse(layers, wanted)[0, 0] == 2
    assert allocate_sparse(*rename(layers, wanted, np.array([0, 2, 1])))[0, 0] == 2


def test_negative_coefficients_rank_the_lowest_layer_values_first():
    projected = allocate({(1, 2): 3}, HALVES, RAMP, coefficients=np.array([[1.], [-10.]]))
    assert projected[0].tolist() == [2, 2, 2, 1]


def test_multiplicative_zero_constraints_keep_a_receiver_out():
    layers = np.stack([RAMP[0], np.zeros((4, 4))])
    layers[1, 0] = 1
    matrix = np.zeros((1, 1, 2, 2))
    matrix[0, 0, 0, 1] = 8
    projected = seals_cython_functions.seals_allocation_from_change_matrix(
        matrix, HALVES.astype(np.int64), layers, np.array([[1., 1.], [1., 0.]]), np.array([2, 1]), np.ones((4, 4), np.int64),
        np.array([1, 2]), np.ones((4, 4)), '', 0.0, '')
    assert (projected[0] == 1).all() and (projected[1] == 2).all()


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_masked_cells_stay_unchanged_whatever_their_scores(bad):
    valid = np.ones((4, 4), np.int64)
    valid[0] = 0
    layer = RAMP.copy()
    layer[0, 0] = bad
    projected = allocate({(1, 2): 8}, HALVES, layer, valid=valid)
    assert (projected[0] == 1).all() and (projected[1] == 2).all()


def test_impossible_requests_stop_at_what_the_donor_has():
    projected = allocate({(1, 2): 12}, HALVES, RAMP)
    assert count_flows(HALVES, projected, 2).tolist() == [[0, 8], [0, 0]]


def test_mask_limits_the_donor():
    valid = np.ones((4, 4), np.int64)
    valid[0] = 0
    assert count_flows(HALVES, allocate({(1, 2): 8}, HALVES, RAMP, valid=valid), 2).tolist() == [[0, 4], [0, 0]]


def test_a_receiver_whose_scores_are_all_zero_gets_nothing_and_takes_nothing_from_others():
    coefficients = np.array([[1.], [1.], [0.]])
    projected = allocate({(1, 2): 4, (1, 3): 4}, np.where(HALVES == 1, 1, 3), RAMP, coefficients)
    assert (projected[:2] == 2).sum() == 4 and (projected == 3).sum() == 8


def test_fractional_requests_count_as_whole_cells():
    assert (allocate({(1, 2): 2.2}, HALVES, RAMP) != HALVES).sum() == 3


def test_loss_is_zero_when_the_observed_map_is_reproduced():
    """The wrapper calibration_zones calls: allocation plus the loss."""
    lulc = HALVES.astype(np.int64)
    observed = allocate({(1, 2): 3, (2, 1): 2}, lulc, RAMP)
    matrix = count_flows(lulc, observed, 2)[None, None].astype(float)
    loss, projected, *_ = seals_cython_functions.calibrate_from_change_matrix(
        matrix, lulc, RAMP, np.ones((2, 1)), np.array([2]), np.ones((4, 4), np.int64),
        np.array([1, 2]), observed, np.ones((4, 4)), '', 0.0, np.float64(7.0), '')
    assert np.array_equal(projected, observed) and loss == 0
