import pytest

from seals.seals_generate_base_data import build_change_class_adjacency_effects


SEALS7_ALL = ['urban', 'cropland', 'grassland', 'forest', 'othernat', 'water', 'other']
SEALS7_CHANGING = ['urban', 'cropland', 'grassland', 'forest', 'othernat']

SEALS8_ALL = ['urban', 'cropland', 'pasture', 'natural_grassland', 'forest', 'othernat', 'water', 'other']
SEALS8_CHANGING = ['urban', 'cropland', 'pasture', 'natural_grassland', 'forest', 'othernat']

# The matrix that was hardcoded in local_data_regressors_starting_values before it was
# generated. Kept verbatim so any change to the generated prior shows up as a failure
# here rather than silently altering calibration for existing seals7 projects.
SEALS7_ORIGINAL_LITERAL = [
    [10, 5, 1, 1, 1],
    [1, 10, 1, 1, 1],
    [1, 1, 10, 1, 1],
    [1, 1, 1, 10, 1],
    [1, 1, 1, 1, 10],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
]


def test_seals7_reproduces_the_original_hardcoded_matrix():
    """The generated prior must be a no-op for every existing seals7 project."""
    assert build_change_class_adjacency_effects(SEALS7_ALL, SEALS7_CHANGING) == SEALS7_ORIGINAL_LITERAL


@pytest.mark.parametrize(
    'all_labels, changing_labels',
    [(SEALS7_ALL, SEALS7_CHANGING), (SEALS8_ALL, SEALS8_CHANGING)],
)
def test_shape_follows_the_class_lists(all_labels, changing_labels):
    effects = build_change_class_adjacency_effects(all_labels, changing_labels)

    assert len(effects) == len(all_labels)
    assert all(len(row) == len(changing_labels) for row in effects)


def test_each_changing_class_gets_its_own_strong_weight():
    effects = build_change_class_adjacency_effects(SEALS8_ALL, SEALS8_CHANGING)

    for column, changing_label in enumerate(SEALS8_CHANGING):
        row = SEALS8_ALL.index(changing_label)
        assert effects[row][column] == 10


def test_the_split_grassland_classes_do_not_share_a_weight():
    """pasture and natural_grassland are distinct classes, not one grassland column."""
    effects = build_change_class_adjacency_effects(SEALS8_ALL, SEALS8_CHANGING)

    pasture_row = effects[SEALS8_ALL.index('pasture')]
    natural_row = effects[SEALS8_ALL.index('natural_grassland')]

    assert pasture_row[SEALS8_CHANGING.index('natural_grassland')] == 1
    assert natural_row[SEALS8_CHANGING.index('pasture')] == 1


def test_non_changing_classes_stay_neutral():
    effects = build_change_class_adjacency_effects(SEALS8_ALL, SEALS8_CHANGING)

    for label in ['water', 'other']:
        assert effects[SEALS8_ALL.index(label)] == [1] * len(SEALS8_CHANGING)


def test_urban_follows_cropland_edges():
    effects = build_change_class_adjacency_effects(SEALS8_ALL, SEALS8_CHANGING)

    assert effects[SEALS8_ALL.index('urban')][SEALS8_CHANGING.index('cropland')] == 5


def test_scheme_without_urban_or_cropland_is_purely_diagonal():
    all_labels = ['forest', 'grassland', 'water']
    changing_labels = ['forest', 'grassland']

    assert build_change_class_adjacency_effects(all_labels, changing_labels) == [
        [10, 1],
        [1, 10],
        [1, 1],
    ]
