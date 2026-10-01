"""The output half of the scheme guard: nothing may be allocated out of a frozen class.

test_coefficient_scheme_check.py covers the input half (right coefficients for the scheme).
These fail independently, which is the reason for both.
"""
import numpy as np
import pytest
from osgeo import gdal

from seals import seals_utils

gdal.UseExceptions()

# seals7. Note water=6, other=7 -- under seals8 those ids mean othernat and water, which is
# why the code resolves classes by LABEL and why test_ids_are_not_hardcoded exists below.
SEALS7_LABELS = ['urban', 'cropland', 'grassland', 'forest', 'othernat', 'water', 'other']
SEALS7_INDICES = [1, 2, 3, 4, 5, 6, 7]
SEALS7_CHANGING = ['urban', 'cropland', 'grassland', 'forest', 'othernat']

SEALS8_LABELS = ['urban', 'cropland', 'pasture', 'natural_grassland', 'forest',
                 'othernat', 'water', 'other']
SEALS8_INDICES = [1, 2, 3, 4, 5, 6, 7, 8]
SEALS8_CHANGING = ['urban', 'cropland', 'pasture', 'natural_grassland', 'forest', 'othernat']


def write(path, array):
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(str(path), array.shape[1], array.shape[0], 1, gdal.GDT_Byte)
    ds.SetGeoTransform([0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
    ds.GetRasterBand(1).WriteArray(array)
    ds = None
    return str(path)


@pytest.fixture
def scene(tmp_path):
    """A 10x10 base map holding every seals7 class."""
    base = np.full((10, 10), 3, dtype=np.uint8)
    base[0:2, :] = 6      # water
    base[2:4, :] = 7      # other
    base[4:6, :] = 1      # urban
    base[6:8, :] = 4      # forest
    return base, tmp_path


def test_unchanged_map_passes(scene):
    base, tmp = scene
    b = write(tmp / 'base.tif', base)
    p = write(tmp / 'projected.tif', base.copy())
    seals_utils.assert_non_changing_classes_unchanged(
        p, b, SEALS7_LABELS, SEALS7_INDICES, SEALS7_CHANGING, ['urban'])


def test_changing_classes_may_move(scene):
    """grassland -> forest is ordinary allocation and must not trip the check."""
    base, tmp = scene
    after = base.copy()
    after[8:10, :] = 4
    b = write(tmp / 'base.tif', base)
    p = write(tmp / 'projected.tif', after)
    seals_utils.assert_non_changing_classes_unchanged(
        p, b, SEALS7_LABELS, SEALS7_INDICES, SEALS7_CHANGING, ['urban'])


def test_water_leaving_raises(scene):
    base, tmp = scene
    after = base.copy()
    after[0, 0:5] = 4                      # 5 px of water -> forest
    b = write(tmp / 'base.tif', base)
    p = write(tmp / 'projected.tif', after)
    with pytest.raises(ValueError) as e:
        seals_utils.assert_non_changing_classes_unchanged(
            p, b, SEALS7_LABELS, SEALS7_INDICES, SEALS7_CHANGING, ['urban'])
    assert 'water' in str(e.value)
    assert '5 px' in str(e.value)          # states the relationship, not just "check failed"
    assert 'forest' in str(e.value)        # names where it went


def test_declared_and_derived_give_different_messages(scene):
    """Urban has a budget and is frozen by choice, so its failure must read as a decision."""
    base, tmp = scene
    after = base.copy()
    after[4, 0:3] = 2                      # urban -> cropland
    b = write(tmp / 'base.tif', base)
    p = write(tmp / 'projected.tif', after)
    with pytest.raises(ValueError) as e:
        seals_utils.assert_non_changing_classes_unchanged(
            p, b, SEALS7_LABELS, SEALS7_INDICES, SEALS7_CHANGING, ['urban'])
    message = str(e.value)
    assert 'additional_protected_class_labels' in message
    assert 'scenario choice' in message
    assert 'no coarse budget' not in message   # that is the derived wording, not this one


def test_urban_unprotected_is_allowed(scene):
    """Dropping urban from the declared list must make the same map legal."""
    base, tmp = scene
    after = base.copy()
    after[4, 0:3] = 2
    b = write(tmp / 'base.tif', base)
    p = write(tmp / 'projected.tif', after)
    seals_utils.assert_non_changing_classes_unchanged(
        p, b, SEALS7_LABELS, SEALS7_INDICES, SEALS7_CHANGING, [])


def test_ids_are_not_hardcoded():
    """The regression that motivated this: seals8 shifts water 6->7 and other 7->8.

    A check written against seals7 ids would read id 6 as water when it is othernat, pass
    confidently, and miss the violation -- the silent misassignment the scheme guard exists
    to prevent.
    """
    base = np.full((6, 6), 3, dtype=np.uint8)
    base[0, :] = 7            # water under SEALS8
    base[1, :] = 6            # othernat under SEALS8 -- a CHANGING class, may move
    after = base.copy()
    after[1, :] = 5           # othernat -> forest, legitimate
    after[0, 0:2] = 5         # water -> forest, a violation

    import tempfile, os
    d = tempfile.mkdtemp()
    b = write(os.path.join(d, 'base.tif'), base)
    p = write(os.path.join(d, 'projected.tif'), after)

    with pytest.raises(ValueError) as e:
        seals_utils.assert_non_changing_classes_unchanged(
            p, b, SEALS8_LABELS, SEALS8_INDICES, SEALS8_CHANGING, [])
    assert 'water' in str(e.value)
    assert 'othernat' not in str(e.value).split('map:')[0]


def test_extent_mismatch_is_reported_not_silently_passed(scene, capsys):
    """An unrun check must never look like a passed one."""
    base, tmp = scene
    b = write(tmp / 'base.tif', base)
    p = write(tmp / 'projected.tif', base[:5, :5].copy())
    seals_utils.assert_non_changing_classes_unchanged(
        p, b, SEALS7_LABELS, SEALS7_INDICES, SEALS7_CHANGING, ['urban'])
    assert 'NOT CHECKED' in capsys.readouterr().out


class _Fake:
    """Stands in for ProjectFlow's p, which is just an attribute bag here."""


@pytest.mark.parametrize('value,expected', [
    (None, []),
    ([], []),
    (['urban'], ['urban']),
    ('urban', ['urban']),                    # scenario column, one class
    ('urban water', ['urban', 'water']),     # space separated
    ('urban, water', ['urban', 'water']),    # comma separated
    ('', []),                                # blank cell
    (float('nan'), []),                      # blank cell through pandas
    ('nan', []),
])
def test_protected_labels_resolve_from_scenario_column(value, expected):
    """Protecting a class is a scenario statement, so it must be settable per scenario row.

    assign_df_row_to_object_attributes puts every scenario column on p, so the value can
    arrive as a string rather than the project's list.
    """
    p = _Fake()
    if value is not None:
        p.additional_protected_class_labels = value
    assert seals_utils.resolve_additional_protected_class_labels(p) == expected


def test_two_scenarios_can_differ(scene):
    """The point of the move: protected and unprotected in ONE scenarios CSV.

    With the setting on the project this needed two separate projects.
    """
    base, tmp = scene
    after = base.copy()
    after[4, 0:3] = 2                        # urban -> cropland
    b = write(tmp / 'base.tif', base)
    p = write(tmp / 'projected.tif', after)

    protected = _Fake(); protected.additional_protected_class_labels = 'urban'
    unprotected = _Fake(); unprotected.additional_protected_class_labels = ''

    with pytest.raises(ValueError):
        seals_utils.assert_non_changing_classes_unchanged(
            p, b, SEALS7_LABELS, SEALS7_INDICES, SEALS7_CHANGING,
            seals_utils.resolve_additional_protected_class_labels(protected))

    seals_utils.assert_non_changing_classes_unchanged(
        p, b, SEALS7_LABELS, SEALS7_INDICES, SEALS7_CHANGING,
        seals_utils.resolve_additional_protected_class_labels(unprotected))


def test_raw_source_lulc_is_refused_not_falsely_flagged(scene, capsys):
    """The regression: handing this the RAW LULC instead of the simplified map.

    The two use the same small integers for different classes -- MapBiomas 6 is
    floodable_forest where seals7 6 is water -- so a naive comparison reports millions of
    impossible conversions with total confidence. A live Brazil run produced exactly that,
    4,029,747 'water -> forest' pixels, before the call site was corrected. The function must
    recognise a foreign scheme and decline rather than accuse.
    """
    base, tmp = scene
    raw = base.copy()
    raw[0, :] = 39            # soybean in MapBiomas; no such class in seals7
    raw[1, :] = 21            # mosaic_of_uses
    b = write(tmp / 'raw_base.tif', raw)
    p = write(tmp / 'projected.tif', base.copy())

    seals_utils.assert_non_changing_classes_unchanged(
        p, b, SEALS7_LABELS, SEALS7_INDICES, SEALS7_CHANGING, ['urban'])

    out = capsys.readouterr().out
    assert 'NOT CHECKED' in out
    assert '39' in out or '21' in out          # names what it did not recognise
