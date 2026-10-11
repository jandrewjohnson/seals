"""coarse_interpolated_states must survive a kill at any point in publication.

The task replaces a raster and then its manifest. Those are two separate os.replace calls, so a
process can die between them and leave the two disagreeing. That window is what the publishing
marker exists for: finding one means the previous attempt died mid-publication, which is a known
state to recover from rather than the hash mismatch it would otherwise resemble.

Prior versions are kept as an actual file, not as a hash. A hash cannot reconstruct a raster, and
a version produced by an implementation this checkout no longer contains is not recomputable from
the inputs either.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import numpy as np
import pandas as pd
import pytest
import hazelbean as hb

from seals import seals_utils
from seals.coarse_interpolation_tasks import coarse_interpolated_states
from test_allocation_task_integration import raster


REL = Path('ssp2/rcp70/magpie/baseline_ignore_dependencies')
TEMPLATE = 'forest_prop_ssp2_rcp70_magpie_baseline_ignore_dependencies_{}.tif'


def _project(tmp_path, monkeypatch):
    source = tmp_path / 'coarse_simplified_proportion'
    out = tmp_path / 'coarse_interpolated_states'
    sources = [raster(source / REL / str(y) / TEMPLATE.format(y), np.array([[v]], dtype='float64'), 1)
               for y, v in [(2020, .2), (2025, .7)]]
    p = SimpleNamespace(
        run_this=True, cur_dir=str(out), coarse_simplified_proportion_dir=str(source),
        coarse_interpolated_states_dir=str(out), input_dir=str(tmp_path), base_data_dir=str(tmp_path),
        scenarios_df=pd.DataFrame([dict(
            scenario_type='baseline_ignore_dependencies', exogenous_label='ssp2', climate_label='rcp70',
            model_label='magpie', counterfactual_label='baseline_ignore_dependencies',
            key_base_year=2023, coarse_correspondence_path='unused')]))
    p.get_path = lambda *a, **k: 'unused'
    monkeypatch.setattr(hb.utils, 'get_reclassification_dict_from_df', lambda *a: {'dst_labels': ['forest']})
    published = out / 'ssp2/magpie/2023/forest_prop_ssp2_magpie_2023.tif'
    return p, sources, published


def _change_implementation(salt='-edited'):
    """Make the implementation hashes differ, as an ordinary code edit does.

    Returns a (apply, restore) pair rather than using monkeypatch, because these tests have to
    keep the change in force across a monkeypatch.undo() that removes only the injected failure.
    It salts the hash of .py paths and leaves raster hashes real, so the output digests the task
    records stay true. A test must never edit the source files it is testing.
    """
    import seals.coarse_interpolation_tasks as module
    real = module.hb.hash_file_path

    def salted(path):
        return real(path) + salt if str(path).endswith('.py') else real(path)

    return (lambda: setattr(module.hb, 'hash_file_path', salted),
            lambda: setattr(module.hb, 'hash_file_path', real))


def _rebuild_differently(monkeypatch, value):
    """Make the rebuild produce DIFFERENT bytes without touching the inputs.

    This is what a real code change does: same sources, same input hashes, a different result.
    It is also the only way to observe the slot being overwritten -- an implementation-only
    rebuild that returns the same numbers is byte-identical, so a test asserting on the slot's
    content cannot tell a preserved original from a new raster written over it.
    """
    real = seals_utils.interpolate_coarse_state_at_year

    def different(source, target_year, brackets, filename_template, *a, **k):
        values, match = real(source, target_year, brackets, filename_template, *a, **k)
        return np.full_like(values, value), match

    monkeypatch.setattr('seals.coarse_interpolation_tasks.seals_utils.interpolate_coarse_state_at_year',
                        different)


def test_a_rebuild_keeps_the_prior_raster_as_a_file(tmp_path, monkeypatch):
    p, sources, published = _project(tmp_path, monkeypatch)
    coarse_interpolated_states(p)
    first = published.read_bytes()
    apply_edit, restore = _change_implementation()
    apply_edit()
    try:
        coarse_interpolated_states(p)
    finally:
        restore()
    kept = published.with_name(published.stem + '.superseded' + published.suffix)
    assert kept.is_file(), 'the replaced version must be kept as a file, not only as a hash'
    assert kept.read_bytes() == first
    record = json.loads(published.with_suffix('.json').read_text())
    assert record['superseded'][-1]['path'] == str(kept)
    assert not published.with_name(published.name + '.publishing.json').exists()


def test_the_kept_slot_is_bounded_to_one_version(tmp_path, monkeypatch):
    p, sources, published = _project(tmp_path, monkeypatch)
    coarse_interpolated_states(p)
    for n in range(3):
        apply_edit, restore = _change_implementation('-edit%d' % n)
        apply_edit()
        try:
            coarse_interpolated_states(p)
        finally:
            restore()
    siblings = sorted(q.name for q in published.parent.glob('*superseded*'))
    assert len(siblings) == 1, f'one slot per output, got {siblings}'


def test_a_kill_between_raster_and_manifest_is_recovered_not_fatal(tmp_path, monkeypatch):
    p, sources, published = _project(tmp_path, monkeypatch)
    coarse_interpolated_states(p)
    manifest = published.with_suffix('.json')
    before = json.loads(manifest.read_text())

    # Die after the raster is published but before the manifest is.
    real_replace = os.replace
    calls = {'n': 0}

    def dying_replace(src, dst):
        calls['n'] += 1
        if str(dst).endswith('.json'):
            raise KeyboardInterrupt('killed between raster and manifest replacement')
        return real_replace(src, dst)

    apply_edit, restore = _change_implementation()
    apply_edit()
    monkeypatch.setattr('seals.coarse_interpolation_tasks.os.replace', dying_replace)
    with pytest.raises(KeyboardInterrupt):
        coarse_interpolated_states(p)

    # This scenario publishes to two destinations -- its own path and the shared reference path --
    # and the kill fires on whichever is reached first, so look tree-wide rather than at one name.
    root = Path(p.coarse_interpolated_states_dir)
    markers = list(root.rglob('*.publishing.json'))
    assert markers, 'the window must be marked'
    killed = markers[0]
    killed_raster = killed.with_name(killed.name[:-len('.publishing.json')])
    killed_manifest = killed_raster.with_suffix('.json')
    intent = json.loads(killed.read_text())
    assert intent['output'] == str(killed_raster)
    # For an implementation-only rebuild the inputs are unchanged, so the recomputed raster is
    # byte-identical and output_sha256 cannot differ. The staleness is in the recorded signature.
    stale_record = json.loads(killed_manifest.read_text())
    assert stale_record['signature']['implementation'] != intent['signature']['implementation'], \
        'the manifest still records the implementation that produced the previous version'

    # The next run must recover rather than raise on the mismatch it now sees.
    monkeypatch.undo()
    monkeypatch.setattr(hb.utils, 'get_reclassification_dict_from_df', lambda *a: {'dst_labels': ['forest']})
    apply_edit()
    try:
        coarse_interpolated_states(p)
    finally:
        restore()
    assert not list(root.rglob('*.publishing.json')), 'markers are cleared once publication completes'
    after = json.loads(killed_manifest.read_text())
    assert after['output_sha256'] == hb.hash_file_path(str(killed_raster))
    assert after['signature']['implementation'] == intent['signature']['implementation']


def test_a_kill_after_the_prior_version_is_moved_aside_leaves_no_hole(tmp_path, monkeypatch):
    """The dangerous window: the prior raster has been moved to its slot and the new one is not
    published yet, so the output path does not exist at all. The next run must restore it rather
    than leave downstream work looking at a missing file."""
    p, sources, published = _project(tmp_path, monkeypatch)
    coarse_interpolated_states(p)
    first = published.read_bytes()

    real_replace = os.replace

    def dying_replace(src, dst):
        if '.rebuilding' in str(src):
            raise KeyboardInterrupt('killed after the prior version was moved aside')
        return real_replace(src, dst)

    apply_edit, restore = _change_implementation()
    apply_edit()
    monkeypatch.setattr('seals.coarse_interpolation_tasks.os.replace', dying_replace)
    with pytest.raises(KeyboardInterrupt):
        coarse_interpolated_states(p)

    root = Path(p.coarse_interpolated_states_dir)
    markers = list(root.rglob('*.publishing.json'))
    assert markers, 'the window must be marked'
    intent = json.loads(markers[0].read_text())
    hole = Path(intent['output'])
    assert not hole.exists(), 'this is the state being tested: the output path is empty'
    assert Path(intent['superseded_path']).is_file(), 'the prior version is in its slot'

    monkeypatch.undo()
    monkeypatch.setattr(hb.utils, 'get_reclassification_dict_from_df', lambda *a: {'dst_labels': ['forest']})
    apply_edit()
    try:
        coarse_interpolated_states(p)
    finally:
        restore()
    assert hole.is_file(), 'the next run must leave a published raster, not a hole'
    assert not list(root.rglob('*.publishing.json'))
    assert published.read_bytes() == first, 'and the content is the same, the inputs not having changed'


def test_a_kill_during_the_rebuild_leaves_the_published_version_intact(tmp_path, monkeypatch):
    p, sources, published = _project(tmp_path, monkeypatch)
    coarse_interpolated_states(p)
    first = published.read_bytes()
    manifest_before = published.with_suffix('.json').read_text()

    def dying_interpolate(*args, **kwargs):
        raise KeyboardInterrupt('killed while rebuilding')

    apply_edit, restore = _change_implementation()
    apply_edit()
    monkeypatch.setattr('seals.coarse_interpolation_tasks.seals_utils.interpolate_coarse_state_at_year',
                        dying_interpolate)
    try:
        with pytest.raises(KeyboardInterrupt):
            coarse_interpolated_states(p)
    finally:
        restore()
    assert published.read_bytes() == first, 'a failed rebuild must not touch the published raster'
    assert published.with_suffix('.json').read_text() == manifest_before
    assert not published.with_name(published.name + '.publishing.json').exists(), 'no marker before any move'


def test_a_changed_input_still_raises_rather_than_replacing(tmp_path, monkeypatch):
    p, sources, published = _project(tmp_path, monkeypatch)
    coarse_interpolated_states(p)
    raster(Path(sources[1]), np.array([[.8]], dtype='float64'), 1)
    with pytest.raises(RuntimeError, match='provenance differs'):
        coarse_interpolated_states(p)
    kept = published.with_name(published.stem + '.superseded' + published.suffix)
    assert not kept.exists(), 'nothing is replaced when an input changed, so nothing is superseded'


def _published_bytes(published):
    return published.read_bytes()


def test_recovery_does_not_overwrite_the_saved_original(tmp_path, monkeypatch):
    """The kill-between-raster-and-manifest case must not destroy the preserved version.

    At that point `out` already holds the NEW raster. The rebuild's move-aside would then shove
    the new raster into the superseded slot and overwrite the original it was keeping, so the
    original bytes would be gone after a recovery that looked successful.
    """
    p, sources, published = _project(tmp_path, monkeypatch)
    coarse_interpolated_states(p)
    original = published.read_bytes()

    real_replace = os.replace

    def die_before_manifest(src, dst):
        if str(dst).endswith('.json'):
            raise KeyboardInterrupt('killed between raster and manifest replacement')
        return real_replace(src, dst)

    apply_edit, restore = _change_implementation()
    apply_edit()
    _rebuild_differently(monkeypatch, 0.9)
    monkeypatch.setattr('seals.coarse_interpolation_tasks.os.replace', die_before_manifest)
    with pytest.raises(KeyboardInterrupt):
        coarse_interpolated_states(p)

    root = Path(p.coarse_interpolated_states_dir)
    marker = list(root.rglob('*.publishing.json'))[0]
    intent = json.loads(marker.read_text())
    slot = Path(intent['superseded_path'])
    assert slot.is_file(), 'the original was moved aside before the kill'
    preserved = slot.read_bytes()

    monkeypatch.undo()
    monkeypatch.setattr(hb.utils, 'get_reclassification_dict_from_df', lambda *a: {'dst_labels': ['forest']})
    _rebuild_differently(monkeypatch, 0.9)
    apply_edit()
    try:
        coarse_interpolated_states(p)
    finally:
        restore()

    assert published.read_bytes() != original, 'the rebuild really did produce different bytes'
    assert slot.is_file(), 'the slot must still hold a version after recovery'
    assert slot.read_bytes() == preserved, 'recovery must not overwrite the saved original'
    assert slot.read_bytes() == original
    assert not list(root.rglob('*.publishing.json'))


def test_a_second_interruption_still_leaves_the_original_recoverable(tmp_path, monkeypatch):
    """Interrupt the recovery itself. The preserved version must survive that too."""
    p, sources, published = _project(tmp_path, monkeypatch)
    coarse_interpolated_states(p)
    original = published.read_bytes()
    real_replace = os.replace

    def die_after_moving_aside(src, dst):
        if '.rebuilding' in str(src):
            raise KeyboardInterrupt('killed after the prior version was moved aside')
        return real_replace(src, dst)

    apply_edit, restore = _change_implementation()
    apply_edit()
    _rebuild_differently(monkeypatch, 0.9)
    monkeypatch.setattr('seals.coarse_interpolation_tasks.os.replace', die_after_moving_aside)
    with pytest.raises(KeyboardInterrupt):
        coarse_interpolated_states(p)

    root = Path(p.coarse_interpolated_states_dir)
    marker = list(root.rglob('*.publishing.json'))[0]
    slot = Path(json.loads(marker.read_text())['superseded_path'])
    assert slot.read_bytes() == original

    # Now interrupt the RECOVERY, at the same point.
    with pytest.raises(KeyboardInterrupt):
        coarse_interpolated_states(p)
    assert slot.is_file(), 'an interrupted recovery must not consume the preserved version'
    assert slot.read_bytes() == original

    monkeypatch.undo()
    monkeypatch.setattr(hb.utils, 'get_reclassification_dict_from_df', lambda *a: {'dst_labels': ['forest']})
    _rebuild_differently(monkeypatch, 0.9)
    apply_edit()
    try:
        coarse_interpolated_states(p)
    finally:
        restore()
    assert slot.read_bytes() == original, 'and it survives the completed recovery'
    assert Path(json.loads(marker.read_text())['output']).is_file() if marker.exists() else True
