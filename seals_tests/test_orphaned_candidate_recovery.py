"""A tile interrupted between writing its maps and recording them must be refused by name.

This is the state a killed worker leaves: maps on disk, no candidate record, so nothing says the
maps are complete and they may be truncated. Allocation must refuse the tile rather than treat the
maps as finished work, must NOT delete them, and must say which tile and which files -- the old
message said only "use a fresh run directory", which on a global run means discarding ~16,000
accepted tiles to recover one. Moving the files aside must restore the tile to a state the normal
path rebuilds.
"""
import json
from pathlib import Path

import pytest

from seals import allocation_tasks


class FakeProject:
    run_this = True
    lulc_src_label = 'esa'
    lulc_simplification_label = 'seals7'
    exogenous_label = 'ssp2'
    climate_label = 'rcp70'
    model_label = 'magpie'
    counterfactual_label = 'baseline_ignore_dependencies'
    year = 2030

    def __init__(self, cur_dir):
        self.cur_dir = str(cur_dir)


@pytest.fixture
def interrupted_tile(tmp_path, monkeypatch):
    """A tile with a valid budget and maps on disk, but no candidate record."""
    tile = tmp_path / '22_21'
    tile.mkdir()
    monkeypatch.setattr(allocation_tasks, 'checked_inputs',
                        lambda p: {'signature': 'matching-signature'})
    (tile / allocation_tasks.TILE_RECORD).write_text(json.dumps({
        'budget': {'input_signature': 'matching-signature', 'targets': [], 'cells': []},
        'coefficients': {'source': 'filled-from-neighbour'},
    }))
    project = FakeProject(tile)
    maps = [tile / allocation_tasks.map_name(project), tile / 'cumulative_change_happened.tif']
    for path in maps:
        path.write_bytes(b'partially written raster')
    return project, tile, maps


def test_an_interrupted_tile_is_refused_by_name_and_its_maps_are_kept(interrupted_tile):
    project, tile, maps = interrupted_tile
    with pytest.raises(RuntimeError) as raised:
        allocation_tasks.allocation(project)
    message = str(raised.value)

    assert 'Orphaned candidate map' in message
    assert '22_21' in message, 'the tile must be named: %s' % message
    for path in maps:
        assert path.name in message, 'each orphaned file must be named: %s' % message
        assert path.is_file(), 'the guard must not delete the evidence'
        assert path.read_bytes() == b'partially written raster'
    assert 'do not delete' in message.lower()
    assert 'fresh run directory' in message, 'it must warn against the destructive recovery'
    assert 'discard' in message


def test_moving_the_maps_aside_lets_the_tile_rebuild(interrupted_tile, tmp_path):
    project, tile, maps = interrupted_tile
    preserved = tmp_path / 'preserved' / '22_21'
    preserved.mkdir(parents=True)
    for path in maps:
        path.rename(preserved / path.name)

    # The guard no longer fires: the tile is back to "has a budget, needs allocating". It fails
    # later for want of the arrays this fixture does not build, which is past the point at issue.
    with pytest.raises(Exception) as raised:
        allocation_tasks.allocation(project)
    assert 'Orphaned' not in str(raised.value), (
        'the orphan guard must be cleared once the maps are moved aside: %s' % raised.value)

    # Nothing was lost in the move.
    for path in maps:
        assert (preserved / path.name).read_bytes() == b'partially written raster'
    assert (tile / allocation_tasks.TILE_RECORD).is_file(), 'the budget record must survive'
    record = json.loads((tile / allocation_tasks.TILE_RECORD).read_text())
    assert 'budget' in record and 'candidate' not in record


def test_a_recorded_candidate_is_not_treated_as_orphaned(tmp_path, monkeypatch):
    """The guard must fire on MISSING provenance, not on every tile that has maps."""
    tile = tmp_path / '22_22'
    tile.mkdir()
    project = FakeProject(tile)
    maps = [tile / allocation_tasks.map_name(project), tile / 'cumulative_change_happened.tif']
    for path in maps:
        path.write_bytes(b'complete raster')
    budget = {'input_signature': 'matching-signature', 'targets': [], 'cells': []}
    monkeypatch.setattr(allocation_tasks, 'checked_inputs',
                        lambda p: {'signature': 'matching-signature'})
    (tile / allocation_tasks.TILE_RECORD).write_text(json.dumps({
        'budget': budget,
        'candidate': {
            'input_signature': 'matching-signature',
            'budget_sha256': allocation_tasks.section_sha256(budget),
            'outputs': {p.name: allocation_tasks.hb.hash_file_path(str(p)) for p in maps},
        },
    }))
    allocation_tasks.allocation(project)      # returns, reusing the recorded candidate
    for path in maps:
        assert path.read_bytes() == b'complete raster'
