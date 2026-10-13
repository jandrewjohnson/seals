"""Exercise real allocation I/O, recursive masks, stitching and refusal on resume."""
from pathlib import Path
import json
import logging
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from seals import seals_main as sm
from seals import allocation_tasks as tasks


def raster(path, values, pixel=.1):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with rasterio.open(path,'w',driver='GTiff',height=values.shape[0],width=values.shape[1],count=1,
                       dtype=values.dtype,crs='EPSG:4326',transform=from_origin(0,1,pixel,pixel),nodata=255) as dst:
        dst.write(values,1)
    return str(path)


def project(tmp_path):
    base=raster(tmp_path/'lulc/esa/seals7/lulc_esa_seals7_2023.tif',np.full((10,10),5,dtype='uint8'))
    area=raster(tmp_path/'area.tif',np.ones((10,10),dtype='float64'))
    coarse=raster(tmp_path/'coarse.tif',np.array([[100.]],dtype='float64'),1.)
    layer=raster(tmp_path/'layer.tif',np.ones((10,10),dtype='float64'))
    classes=['urban','cropland','grassland','forest','othernat']
    frame=pd.DataFrame([dict(spatial_regressor_name='suitability',type='additive',data_location=layer,**{c:1. for c in classes})])
    p=SimpleNamespace(run_this=True,key_base_year=2023,lulc_simplification_label='seals7',model_label='magpie',
      aoi_lulc_simplified_paths={2023:base},lulc_simplified_paths={2023:base},base_year_lulc_path=base,
      calibration_parameters_source='fixture',combined_calibration_parameters_df=frame,calibration_parameters_override_dict={},
      use_calibration_created_coefficients=False,changing_class_indices=[1,2,3,4,5],changing_class_labels=classes,
      lulc_src_label='esa',exogenous_label='ssp2',climate_label='rcp26',counterfactual_label='net_zero',scenario_label='net_zero',
      scenario_type='policy',years=[2030,2040],seals_allocation_mode='cell_residual_strict',skip_created_downscaling_zones=True,
      fine_blocks_list=[0,0,10,10],global_fine_blocks_list=[0,0,10,10],coarse_blocks_list=[0,0,1,1],global_coarse_blocks_list=[0,0,1,1],
      gaussian_sigmas_to_test=[],aoi_ha_per_cell_coarse_path=coarse,aoi_ha_per_cell_fine_path=area,output_writing_level=0,
      intermediate_dir=str(tmp_path),fine_processed_inputs_dir=str(tmp_path),input_dir=str(tmp_path),base_data_dir=str(tmp_path/'base_data'),
      write_projected_coarse_change_chunks=False,L=logging.getLogger('integration'),force_to_global_bb=False,
      clip_to_aoi=False,all_class_indices=[1,2,3,4,5,6,7],all_class_labels=classes+['water','barren'],baseline_reference_label='baseline')
    p.get_path=lambda *parts,**kwargs: str(tmp_path.joinpath(*parts)) if not Path(parts[0]).is_absolute() else parts[0]
    p.allocations_dir=str(tmp_path/'allocations')
    return p


def configure_year(p,tmp_path,year,previous):
    p.year=year;p.previous_year=previous
    root=Path(p.allocations_dir)/'ssp2/rcp26/magpie/net_zero'/str(year)/'allocation_zones'
    p.cur_dir=str(root/'0_0');p.cur_dir_parent_dir=str(root/'0_0');Path(p.cur_dir).mkdir(parents=True,exist_ok=True)
    (root/'global_processing_blocks_list.csv').write_text('0,0,1,1\n')
    p.projected_coarse_change_dir=str(tmp_path/'requests'/str(year))
    for c in p.changing_class_labels:
        raster(Path(p.projected_coarse_change_dir)/f'{c}_{year}_{previous}_ha_diff_ssp2_rcp26_magpie_net_zero.tif',
               np.array([[20. if c=='cropland' else 0.]],dtype='float64'),1.)


def run_tile_tasks(p):
    root=Path(p.cur_dir_parent_dir)
    for task in (tasks.zone_change_budgets,tasks.allocation,tasks.allocation_reconciliation,tasks.allocation_audit):
        p.cur_dir=str(root);Path(p.cur_dir).mkdir(parents=True,exist_ok=True)
        task(p)
    p.cur_dir=str(root)


def test_real_allocation_resume_and_stitch(tmp_path):
    p=project(tmp_path)
    for year,previous in [(2030,2023),(2040,2030)]:
        configure_year(p,tmp_path,year,previous);run_tile_tasks(p)
        output=Path(p.cur_dir)/f'lulc_esa_seals7_ssp2_rcp26_magpie_net_zero_{year}.tif'
        with rasterio.open(output) as src: assert (src.read(1)==2).sum()==(20 if year==2030 else 40)
        stamp=output.stat().st_mtime_ns;run_tile_tasks(p);assert output.stat().st_mtime_ns==stamp
    # Actual stitching uses the registered task function, with local fixtures.
    p.scenarios_df=pd.DataFrame([dict(scenario_label='baseline',scenario_type='baseline',model_label='magpie'),
       dict(scenario_label='net_zero',scenario_type='policy',model_label='magpie')])
    p.cur_dir=str(tmp_path/'stitch');Path(p.cur_dir).mkdir()
    # Base is deliberately larger than the test tile extent to use the local stitch branch.
    p.base_year_lulc_path=raster(tmp_path/'raw_extent.tif',np.full((20,20),5,dtype='uint8'))
    sm.stitched_lulc_simplified_scenarios(p)
    sm.stitched_lulc_simplified_scenarios(p)
    assert len(list(Path(p.cur_dir).glob('*.allocation.json')))==2
    configure_year(p,tmp_path,2040,2030)
    output=Path(p.cur_dir)/'lulc_esa_seals7_ssp2_rcp26_magpie_net_zero_2040.tif'
    with rasterio.open(output,'r+') as dst:
        a=dst.read(1);a[0,0]=4;dst.write(a,1)
    with pytest.raises(RuntimeError,match='provenance differs'):run_tile_tasks(p)
    p.cur_dir=str(tmp_path/'stitch')
    with pytest.raises(RuntimeError,match='differs from acceptance'):sm.stitched_lulc_simplified_scenarios(p)


def test_projectflow_stops_before_downstream_on_invalid_resume(tmp_path):
    import hazelbean as hb
    fixture=project(tmp_path)
    configure_year(fixture,tmp_path,2030,2023)
    task_dir=fixture.cur_dir
    def build():
        flow=hb.ProjectFlow(project_dir=str(tmp_path/'flow'))
        for key,value in vars(fixture).items():setattr(flow,key,value)
        flow.run_in_parallel=False
        # Mirror the real tree: the four tasks are children of the allocation-zones ITERATOR and
        # share the tile's own directory, which is what creates_dir=False selects under an
        # iterator. Registering them on a flat tree would exercise a layout production never uses.
        def allocation_zones(p):
            p.iterator_replacements={'cur_dir_parent_dir':[str(task_dir)]}
        zones=flow.add_iterator(allocation_zones,run_in_parallel=False)
        for task in (tasks.zone_change_budgets,tasks.allocation,tasks.allocation_reconciliation,tasks.allocation_audit):
            flow.add_task(task,parent=zones,skip_existing=0,creates_dir=False)
        def downstream(p):
            if p.run_this:(tmp_path/'DOWNSTREAM_RAN').write_text('ran')
        flow.add_task(downstream,skip_existing=0)
        return flow
    build().execute()
    assert (tmp_path/'DOWNSTREAM_RAN').is_file()
    (tmp_path/'DOWNSTREAM_RAN').unlink()
    # Change an actual request raster, retaining a completed map. Resume must fail.
    request=Path(fixture.projected_coarse_change_dir)/'cropland_2030_2023_ha_diff_ssp2_rcp26_magpie_net_zero.tif'
    with rasterio.open(request,'r+') as dst:dst.write(np.array([[21.]]),1)
    with pytest.raises(RuntimeError,match='provenance differs'):build().execute()
    assert not (tmp_path/'DOWNSTREAM_RAN').exists()


def test_stitch_refuses_missing_planned_tile(tmp_path):
    from seals.allocation_provenance import validated_tiles
    p=project(tmp_path);configure_year(p,tmp_path,2030,2023);run_tile_tasks(p)
    root=Path(p.cur_dir).parents[0]
    (root/'global_processing_blocks_list.csv').write_text('0,0,1,1\n1,0,1,1\n')
    with pytest.raises(RuntimeError,match='Missing strict tile acceptance'):
        validated_tiles(root.parent,'lulc_esa_seals7_ssp2_rcp26_magpie_net_zero_2030.tif')


def test_resume_rewrites_no_accepted_tile_artifact(tmp_path):
    """A resume must leave every accepted tile artifact alone, not just the lulc map.

    allocation_reconciliation wrote reconciliation.csv unconditionally, so every resume re-read
    the tile's inputs and rasters to reproduce a file the audit had already accepted: 1,158
    rewrites in the bounded fixture, 435,942 at global scale. Measured by mtime AND inode, because
    identical content cannot show the work was skipped -- a rewrite producing the same bytes
    changes the mtime, and an atomic replace changes the inode.
    """
    p = project(tmp_path)
    configure_year(p, tmp_path, 2030, 2023)
    run_tile_tasks(p)
    root = Path(p.cur_dir_parent_dir)
    artifacts = sorted(q for q in root.iterdir() if q.is_file())
    assert [q.name for q in artifacts].count('tile_record.json') == 1, [q.name for q in artifacts]
    before = {q.name: (q.stat().st_mtime_ns, q.stat().st_ino) for q in artifacts}
    run_tile_tasks(p)
    after = {q.name: (q.stat().st_mtime_ns, q.stat().st_ino) for q in artifacts}
    rewritten = sorted(name for name in before if before[name] != after[name])
    assert rewritten == [], f'resume rewrote {rewritten}'


def _tamper(path, mutate):
    """Rewrite the accepted SECTION of a tile record, returning the original text to restore."""
    original = path.read_text()
    record = json.loads(original)
    mutate(record['accepted'])
    path.write_text(json.dumps(record, indent=2))
    return original


def test_reconciliation_is_rebuilt_when_the_inputs_it_depends_on_moved_on(tmp_path):
    """Reuse must require the table to be CURRENT, not merely once-accepted.

    A hash match on reconciliation.csv alone would also accept a table whose upstream has since
    changed, so each of the audit's records is checked: the signature of the inputs as they are
    now, the budget the table was balanced from, and the maps it measured. Each is tampered with
    in turn and the table must be rebuilt every time.
    """
    p = project(tmp_path)
    configure_year(p, tmp_path, 2030, 2023)
    run_tile_tasks(p)
    root = Path(p.cur_dir_parent_dir)
    table = root / 'tile_record.json'
    accepted = table

    for field, mutate in [
        ('input_signature', lambda r: r.__setitem__('input_signature', 'not-the-current-inputs')),
        ('budget_sha256', lambda r: r.__setitem__('budget_sha256', '0' * 64)),
        ('outputs', lambda r: r.__setitem__('outputs', {k: '0' * 64 for k in r['outputs']})),
    ]:
        original = _tamper(accepted, mutate)
        stamp = (table.stat().st_mtime_ns, table.stat().st_ino)
        tasks.allocation_reconciliation(p)
        assert (table.stat().st_mtime_ns, table.stat().st_ino) != stamp, \
            f'reuse survived a changed {field}; a stale table would be kept'
        accepted.write_text(original)
        # Re-accept so the next iteration starts from a consistent tile again.
        tasks.allocation_audit(p)


def test_reconciliation_is_reused_when_nothing_moved(tmp_path):
    """The other half: with every record still matching, the table is left alone."""
    p = project(tmp_path)
    configure_year(p, tmp_path, 2030, 2023)
    run_tile_tasks(p)
    table = Path(p.cur_dir_parent_dir) / 'tile_record.json'
    stamp = (table.stat().st_mtime_ns, table.stat().st_ino)
    tasks.allocation_reconciliation(p)
    assert (table.stat().st_mtime_ns, table.stat().st_ino) == stamp


def test_record_stitched_leaves_an_identical_record_alone(tmp_path):
    """Rewriting an identical provenance record is work repeated on resume.

    record_stitched runs for every stitched map on every run, including maps the resume reused, so
    it replaced each .allocation.json atomically -- a new inode -- for no change. That both repeats
    work and makes an unchanged-resume check impossible to satisfy: six stitched records were
    rewritten on a resume that had reused all six maps.
    """
    from seals.allocation_provenance import record_stitched
    mosaic = raster(tmp_path / 'stitched.tif', np.full((4, 4), 2, dtype='uint8'))
    tiles = {'tile_a': 'a' * 64}
    baseline = {'base.tif': 'b' * 64}
    record_stitched(mosaic, tiles, baseline)
    audit = Path(mosaic).with_suffix('.allocation.json')
    stamp = (audit.stat().st_mtime_ns, audit.stat().st_ino)
    record_stitched(mosaic, tiles, baseline)
    assert (audit.stat().st_mtime_ns, audit.stat().st_ino) == stamp, 'an identical record was rewritten'


def test_record_stitched_rewrites_when_the_accepted_tiles_change(tmp_path):
    """The other half: a genuinely different record must be written."""
    from seals.allocation_provenance import record_stitched
    mosaic = raster(tmp_path / 'stitched.tif', np.full((4, 4), 2, dtype='uint8'))
    record_stitched(mosaic, {'tile_a': 'a' * 64}, {'base.tif': 'b' * 64})
    audit = Path(mosaic).with_suffix('.allocation.json')
    stamp = (audit.stat().st_mtime_ns, audit.stat().st_ino)
    record_stitched(mosaic, {'tile_a': 'c' * 64}, {'base.tif': 'b' * 64})
    assert (audit.stat().st_mtime_ns, audit.stat().st_ino) != stamp
    assert json.loads(audit.read_text())['tiles'] == {'tile_a': 'c' * 64}


# ---- consolidated tile record: every section's absence and tampering must be refused ----

def _record(root):
    return json.loads((Path(root) / 'tile_record.json').read_text())


def _write_record(root, record):
    (Path(root) / 'tile_record.json').write_text(json.dumps(record, indent=2))


def test_the_consolidated_tile_costs_four_inodes(tmp_path):
    """The reason for the change: five metadata files became one section-bearing record."""
    p = project(tmp_path)
    configure_year(p, tmp_path, 2030, 2023)
    run_tile_tasks(p)
    root = Path(p.cur_dir_parent_dir)
    files = sorted(q.name for q in root.iterdir() if q.is_file())
    assert files == ['cumulative_change_happened.tif',
                     'lulc_esa_seals7_ssp2_rcp26_magpie_net_zero_2030.tif',
                     'tile_record.json'], files
    assert len(list(root.rglob('*'))) == 3


def test_every_audit_field_survives_the_merge(tmp_path):
    """No audit information may be lost: each section and each accepted field must still be there."""
    p = project(tmp_path)
    configure_year(p, tmp_path, 2030, 2023)
    run_tile_tasks(p)
    record = _record(p.cur_dir_parent_dir)
    assert set(record) == {'budget', 'coefficients', 'candidate', 'reconciliation', 'accepted'}
    accepted = record['accepted']
    for field in ('status', 'mode', 'input_signature', 'cells', 'outputs',
                  'budget_sha256', 'reconciliation_sha256', 'predecessor'):
        assert field in accepted, field
    assert accepted['status'] == 'accepted' and accepted['mode'] == 'cell_residual_strict'
    assert record['coefficients'], 'the calibration the targets came from must be kept'
    assert record['reconciliation'], 'the reconciliation rows must be kept'


@pytest.mark.parametrize('section', ['budget', 'candidate', 'reconciliation'])
def test_a_missing_section_is_refused_not_defaulted(tmp_path, section):
    """Absence must raise. A defaulted empty section would let the audit accept an unbuilt tile."""
    p = project(tmp_path)
    configure_year(p, tmp_path, 2030, 2023)
    run_tile_tasks(p)
    root = p.cur_dir_parent_dir
    record = _record(root)
    del record[section]
    del record['accepted']          # force the audit to re-derive rather than short-circuit
    _write_record(root, record)
    with pytest.raises(RuntimeError, match='no .* section|provenance differs'):
        tasks.allocation_audit(p)


def test_a_tampered_budget_section_breaks_the_audit_hash(tmp_path):
    """budget_sha256 is now a hash of the SECTION, so editing the section must be caught."""
    p = project(tmp_path)
    configure_year(p, tmp_path, 2030, 2023)
    run_tile_tasks(p)
    root = p.cur_dir_parent_dir
    record = _record(root)
    record['budget']['targets'][0][0][0] = 999.0
    del record['accepted']
    _write_record(root, record)
    with pytest.raises(RuntimeError, match='provenance differs'):
        tasks.allocation_audit(p)


def test_a_tampered_reconciliation_section_is_caught_by_the_accepted_hash(tmp_path):
    """The accepted record pins the reconciliation section; editing it must not pass unnoticed."""
    p = project(tmp_path)
    configure_year(p, tmp_path, 2030, 2023)
    run_tile_tasks(p)
    root = p.cur_dir_parent_dir
    record = _record(root)
    stored = record['accepted']['reconciliation_sha256']
    record['reconciliation'][0]['realised_ha'] = -12345.0
    _write_record(root, record)
    assert tasks.section_sha256(_record(root)['reconciliation']) != stored
    # The reconciliation task must refuse to reuse a table whose hash no longer matches the audit.
    tasks.allocation_reconciliation(p)
    assert tasks.section_sha256(_record(root)['reconciliation']) == stored, \
        'the tampered table was neither rebuilt nor refused'


def test_section_hash_is_order_independent():
    """Canonical JSON: the same content must hash the same whatever the key order."""
    a = {'x': 1, 'y': [1, 2], 'z': {'b': 2, 'a': 1}}
    b = {'z': {'a': 1, 'b': 2}, 'y': [1, 2], 'x': 1}
    assert tasks.section_sha256(a) == tasks.section_sha256(b)
    assert tasks.section_sha256(a) != tasks.section_sha256({'x': 2, 'y': [1, 2], 'z': {}})
