"""ProjectFlow outputs for cell-local allocation, reconciliation and acceptance.

Numerical routines remain in signed_allocation and zone_change_budgets. No
private evidence directory or job identifier is an execution dependency.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import hazelbean as hb
from seals import seals_utils
from seals import signed_allocation as signed


def map_name(p, year=None):
    return f'lulc_{p.lulc_src_label}_{p.lulc_simplification_label}_{p.exogenous_label}_{p.climate_label}_{p.model_label}_{p.counterfactual_label}_{p.year if year is None else year}.tif'


TILE_RECORD = 'tile_record.json'


def section_sha256(section):
    """Hash a record section the way a separate file used to be hashed.

    The audit recorded budget_sha256 and reconciliation_sha256 over FILES. With the five metadata
    files merged into one record there is no separate file to hash, so the same guarantee is kept
    over the section's canonical JSON: same content, same digest, independent of key order.
    """
    return hashlib.sha256(json.dumps(section, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_tile_record(root):
    """The tile's one metadata file, or an empty record when it does not exist yet."""
    path = Path(root)/TILE_RECORD
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def require_section(root, name):
    """A section a later task depends on. Absence is a fault, never a default.

    Returning an empty section here would let allocation run without a budget, or the audit accept
    a tile whose candidate was never written.
    """
    record = read_tile_record(root)
    if name not in record:
        raise RuntimeError('Tile record %s has no %r section: %s' % (TILE_RECORD, name, Path(root)/TILE_RECORD))
    return record[name]


def write_tile_section(root, name, section):
    """Merge one section into the tile record, atomically, leaving the others untouched.

    Tasks within a tile run in sequence so there is no contention, and tiles write different files
    so parallel tiles are unaffected -- the same concurrency the five separate files had.
    """
    path = Path(root)/TILE_RECORD
    record = read_tile_record(root)
    record[name] = section
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True))
    tmp.replace(path)
    return record


def _block_offsets(block_list):
    """A block list as integers, whichever path produced it.

    On a FRESH run seals_main builds these with
    get_subglobal_block_list_from_resolution_and_bb, which yields ints. On a RESUME it reads the
    precached CSV with hb.file_to_python_object(..., '2d_list') and passes no
    output_value_data_type, so every element comes back a STRING: measured on this pilot's cached
    file, ['0', '0', '360', '360', '0', '0']. A string is a sequence, so the geotransform offset
    below raised "can't multiply sequence by non-int of type 'float'" on the resume path only,
    after 772 tiles had allocated cleanly on the fresh path. Job 3472099.

    Normalised once, here, because every consumer of these lists is affected, not only the
    geotransform: the chunk reads take them too.

    Args:
        block_list (sequence): a block list, or one row of one, as ints, floats or strings.

    Returns:
        list: the same values as ints.

    Raises:
        ValueError: if an element is not a whole number. These are pixel indices and sizes; a
            fractional one means the list is not what this task thinks it is.
    """
    out = []
    for value in block_list:
        number = float(value)
        if number != int(number):
            raise ValueError('block list element %r is not a whole number; these are pixel '
                             'indices and sizes' % (value,))
        out.append(int(number))
    return out


def read_inputs(p):
    """Read one tile's standard inputs. No raster/CSV writes or downloads."""
    if list(p.changing_class_indices) != [1,2,3,4,5]:
        raise ValueError('Cell-residual tasks require class codes 1..5')
    if p.calibration_parameters_override_dict or p.use_calibration_created_coefficients:
        raise ValueError('Cell-residual task path requires the explicit combined calibration table; overrides need validation')
    # Integers, whichever path built these: a resume supplies them as strings from the cached CSV.
    fine_blocks=_block_offsets(p.fine_blocks_list)
    coarse_blocks=_block_offsets(p.coarse_blocks_list)
    global_fine_blocks=_block_offsets(p.global_fine_blocks_list)
    root=Path(p.cur_dir_parent_dir)
    df=p.combined_calibration_parameters_df.copy()
    if 'calibration_block_index' in df:
        df=df[df.calibration_block_index==root.name+'_1_1']
    if df.empty:
        raise ValueError(f'Missing coefficients for tile {root.name}')
    labels=list(p.changing_class_labels)
    columns=['class_'+str(c) for c in labels]
    if not all(c in df for c in columns):columns=labels
    if not all(c in df for c in columns):raise ValueError('Missing named class coefficient columns')
    coefficients=df[columns].apply(pd.to_numeric).to_numpy(dtype=float).T
    # Shared with the standard allocation path rather than restated here. The restatement this
    # replaces accepted ANY 'gaussian_*' string, so a table carrying a sigma that was never fitted
    # was silently coded as additive instead of failing.
    types = seals_utils.regressor_type_codes(df.type, p.gaussian_sigmas_to_test)
    layer_paths=[]
    for value in df.data_location:
        value=Path(str(value))
        candidates=[value] if value.is_absolute() else [Path(d)/value for d in (p.intermediate_dir,p.fine_processed_inputs_dir,p.input_dir,p.base_data_dir)]
        found=next((q for q in candidates if q.is_file()),None)
        if found is None:raise FileNotFoundError(str(value))
        layer_paths.append(found)
    layers=np.array([hb.load_geotiff_chunk_by_cr_size(str(q),global_fine_blocks if hb.get_bounding_box(str(q))==hb.global_bounding_box else fine_blocks) for q in layer_paths],dtype=float)
    base=Path(p.lulc_simplified_paths[p.key_base_year])
    before=hb.load_geotiff_chunk_by_cr_size(str(base),fine_blocks).astype(np.int64)
    mask=(before!=hb.get_ndv_from_path(str(base))).astype(np.int64)
    cumulative=np.zeros_like(mask)
    predecessor={'base':str(base)}
    # Missing earlier tiles are allowed only when their planned footprint excludes
    # this tile. A planned-but-missing predecessor must never silently use the base.
    for year in sorted((int(y) for y in p.years if p.key_base_year<int(y)<p.year),reverse=True):
        prior=Path(str(root).replace('/'+str(p.year)+'/', '/'+str(year)+'/'))
        footprint=prior.parent/'global_processing_blocks_list.csv'
        if not footprint.is_file():raise RuntimeError(f'Missing predecessor footprint: {footprint}')
        blocks=np.loadtxt(footprint,delimiter=',',dtype=int,ndmin=2)
        if root.name not in {f'{r[0]}_{r[1]}' for r in blocks}:continue
        audit_path=prior/TILE_RECORD
        audit=json.loads(audit_path.read_text()).get('accepted',{})
        if audit['status']!='accepted':raise RuntimeError('Unaccepted predecessor')
        for name,digest in audit['outputs'].items():
            if hb.hash_file_path(str(prior/name))!=digest:raise RuntimeError('Predecessor output differs from acceptance')
        before=hb.as_array(str(prior/map_name(p,year))).astype(np.int64)
        cumulative=hb.as_array(str(prior/'cumulative_change_happened.tif')).astype(np.int64)
        predecessor={'audit':str(audit_path),'sha256':hb.hash_file_path(str(audit_path))}
        break
    mask=np.where(cumulative==1,0,mask)
    hectares=hb.load_geotiff_chunk_by_cr_size(p.aoi_ha_per_cell_fine_path,fine_blocks).astype(float)
    suffix=f'_{p.year}_{p.previous_year}_ha_diff_{p.exogenous_label}_{p.climate_label}_{p.model_label}_{p.counterfactual_label}.tif'
    request_paths=[Path(p.projected_coarse_change_dir)/(c+suffix) for c in labels]
    requests=np.array([hb.load_geotiff_chunk_by_cr_size(str(q),coarse_blocks) for q in request_paths],dtype=float)
    arrays=dict(requests=requests,before=before,hectares=hectares,mask=mask,layers=layers,coefficients=coefficients,types=types)
    digest=hashlib.sha256((signed.input_signature(*arrays.values())+hb.hash_file_path(__file__)+json.dumps(predecessor,sort_keys=True)).encode()).hexdigest()
    gt=list(hb.get_geotransform_path(str(base)))
    col,row=fine_blocks[:2];gt[0]+=col*gt[1]+row*gt[2];gt[3]+=col*gt[4]+row*gt[5]
    return dict(arrays=arrays,coefficients=df,signature=digest,cumulative=cumulative,base=str(base),geotransform=gt,
                predecessor=predecessor,source_paths=[str(q) for q in request_paths+layer_paths])


def checked_inputs(p):
    """Use process-local arrays only for the exact current tile; otherwise reload."""
    key=str(Path(p.cur_dir_parent_dir))
    cached=getattr(p,'_cell_allocation_inputs',None)
    return cached[1] if cached and cached[0]==key else read_inputs(p)


def zone_change_budgets(p):
    """Persist original requests, eligible stocks and balanced cell-local targets."""
    if not p.run_this:return
    data=read_inputs(p)
    p._cell_allocation_inputs=(str(Path(p.cur_dir_parent_dir)),data)
    root=Path(p.cur_dir);root.mkdir(parents=True,exist_ok=True)
    targets,rows=signed.budget_grid(**data['arrays'])
    existing=read_tile_record(root)
    if 'budget' in existing:
        prior=existing['budget']
        if prior['input_signature']!=data['signature'] or prior['targets']!=targets.tolist() or prior['cells']!=rows:
            raise RuntimeError('Budget provenance differs; use a fresh run directory')
        return
    budget=dict(input_signature=data['signature'],targets=targets.tolist(),cells=rows,
                predecessor=data['predecessor'],source_paths=data['source_paths'])
    # The coefficients table travels in the record rather than a sibling CSV. It is the calibration
    # the targets were computed from, so losing it would make the budget unreproducible.
    write_tile_section(root,'coefficients',data['coefficients'].to_dict(orient='records'))
    write_tile_section(root,'budget',budget)


def allocation(p):
    """Write candidate maps using the persisted budget task output.

    A pixel failure is carried to allocation_audit, which owns exact replay
    inputs and raises before stitching or any downstream service work.
    """
    if not p.run_this:return
    data=checked_inputs(p);root=Path(p.cur_dir);root.mkdir(parents=True,exist_ok=True)
    budget=require_section(root,'budget')
    if budget['input_signature']!=data['signature']:raise RuntimeError('Budget provenance differs')
    outputs=[root/map_name(p),root/'cumulative_change_happened.tif']
    existing=read_tile_record(root)
    if 'candidate' in existing:
        record=existing['candidate']
        if record['input_signature']!=data['signature'] or record['budget_sha256']!=section_sha256(budget) or any(not q.is_file() or hb.hash_file_path(str(q))!=record['outputs'].get(q.name) for q in outputs):
            raise RuntimeError('Candidate provenance differs; refusing reuse')
        p._cell_allocation_failure=None
        return
    if any(q.exists() for q in outputs):raise RuntimeError('Orphaned candidate map; use a fresh run directory')
    arrays=data['arrays'];args={k:v for k,v in arrays.items() if k!='requests'}
    try:
        after,_,changed,pixels=signed.allocate_budget_grid(np.asarray(budget['targets']),**args)
    except signed.AllocationFailure as error:
        p._cell_allocation_failure=(str(root),error)
        return
    p._cell_allocation_failure=None
    cumulative=np.maximum(data['cumulative'],changed)
    # The reference raster supplies the projection and data type, NOT the extent. save_array_as_geotiff
    # takes n_cols/n_rows from the match raster whenever the caller leaves them None -- its
    # array.shape fallback sits after that block and so never runs -- so passing the AOI-wide base
    # here wrote a 4680x6120 raster carrying this TILE's geotransform, and the audit read back
    # (4680,6120) against a (360,360) input. Job 3467067. State the tile's own dimensions.
    for path,values,ndv,dtype in [(outputs[0],after,255,1),(outputs[1],cumulative,-9999,5)]:
        hb.save_array_as_geotiff(values,str(path),data['base'],geotransform_override=data['geotransform'],
                                 n_cols=values.shape[1],n_rows=values.shape[0],
                                 ndv=ndv,data_type=dtype,compress=True)
        # Verify on READBACK, before anything consumes these. A candidate whose extent or
        # geotransform silently disagrees with the arrays it was allocated from cannot be
        # reconciled or audited meaningfully, and the mismatch surfaced three tasks downstream.
        written=hb.get_raster_info_hb(str(path))
        if tuple(written['raster_size'])!=(values.shape[1],values.shape[0]):
            raise RuntimeError('%s written as %s but the allocated array is %s (cols,rows). The '
                               'reference raster supplies projection and type, never extent.'
                               %(path,tuple(written['raster_size']),(values.shape[1],values.shape[0])))
        if not np.allclose(np.asarray(written['geotransform'],dtype=float),
                           np.asarray(data['geotransform'],dtype=float),rtol=0,atol=1e-12):
            raise RuntimeError('%s geotransform %s differs from the tile geotransform %s, so its '
                               'pixels do not sit where the allocation placed them.'
                               %(path,tuple(written['geotransform']),tuple(data['geotransform'])))
    record=dict(status='candidate',input_signature=data['signature'],budget_sha256=section_sha256(budget),
        outputs={q.name:hb.hash_file_path(str(q)) for q in outputs})
    write_tile_section(root,'candidate',record)


def allocation_reconciliation(p):
    """Own the original / constrained / realised / eligible-stock CSV."""
    if not p.run_this:return
    failure=getattr(p,'_cell_allocation_failure',None)
    if failure and failure[0]==str(Path(p.cur_dir_parent_dir)):return
    data=checked_inputs(p)
    root=Path(p.cur_dir);root.mkdir(parents=True,exist_ok=True)
    existing=read_tile_record(root)
    # Rewriting this on every resume reproduced a file the audit had already accepted: 1,158
    # rewrites in the bounded fixture, 435,942 at global scale. Reuse requires the table to be
    # CURRENT, not merely the one some past audit accepted, so every input the table depends on is
    # checked against that audit: the signature of the inputs as they are NOW, the budget it was
    # balanced from, the maps it measured, and the table itself. A hash match on the table alone
    # would also accept a table whose upstream has since moved on. Checked against the audit's
    # existing records rather than a new sidecar, because a per-tile sidecar costs an inode per
    # tile and the surrounding work exists to spend fewer.
    if 'reconciliation' in existing and 'accepted' in existing and 'budget' in existing:
        prior=existing['accepted']
        outputs=prior.get('outputs',{}) or {}
        if (prior.get('status')=='accepted'
                and prior.get('input_signature')==data['signature']
                and prior.get('budget_sha256')==section_sha256(existing['budget'])
                and outputs
                and all(hb.path_exists(str(root/name))
                        and hb.hash_file_path(str(root/name))==digest
                        for name,digest in outputs.items())
                and prior.get('reconciliation_sha256')==section_sha256(existing['reconciliation'])):
            return
    budget=require_section(root,'budget')
    after=hb.as_array(str(root/map_name(p)))
    rows=[]
    windows={(r,c):w for r,c,w in signed.cell_windows(data['arrays']['requests'],data['arrays']['before'])}
    for cell in budget['cells']:
        cr,cc=cell['coarse_row'],cell['coarse_col'];b=cell['budget']
        window=windows[cr,cc]
        before=data['arrays']['before'][window];h=data['arrays']['hectares'][window]
        for i,label in enumerate(p.changing_class_labels):
            actual=float(h[after[window]==i+1].sum()-h[before==i+1].sum())
            rows.append(dict(coarse_row=cr,coarse_col=cc,class_label=label,original_ha=b['original_ha'][i],
                constrained_ha=b['target_ha'][i],realised_ha=actual,eligible_stock_ha=b['eligible_stock_ha'][i],
                residual_ha=actual-b['target_ha'][i],expansion_factor=b['expansion_factor']))
    write_tile_section(root,'reconciliation',rows)


def allocation_audit(p):
    """Independently check map bytes and physical budgets; own success/failure evidence."""
    if not p.run_this:return
    root=Path(p.cur_dir);root.mkdir(parents=True,exist_ok=True);data=checked_inputs(p)
    failure=getattr(p,'_cell_allocation_failure',None)
    if failure and failure[0]==str(root):
        error=failure[1]
        np.savez_compressed(root/'failed_cell.npz',**error.inputs)
        (root/'failure.json').write_text(json.dumps(dict(status='failed',cell=error.cell,error=str(error),input_signature=data['signature']),indent=2))
        raise RuntimeError(f'Allocation audit failed; exact inputs: {root}/failed_cell.npz') from error
    budget=require_section(root,'budget')
    candidate=require_section(root,'candidate')
    if candidate['input_signature']!=data['signature'] or budget['input_signature']!=data['signature'] or candidate['budget_sha256']!=section_sha256(budget):
        raise RuntimeError('Audit input provenance differs')
    for name,digest in candidate['outputs'].items():
        if hb.hash_file_path(str(root/name))!=digest:raise RuntimeError('Candidate output differs')
    after=hb.as_array(str(root/map_name(p)))
    arrays=data['arrays'];targets=np.asarray(budget['targets']);rows=[]
    for cr,cc,w in signed.cell_windows(targets,arrays['before']):
        audit=signed.validate_cell(arrays['before'][w],after[w],arrays['hectares'][w],arrays['mask'][w],targets[:,cr,cc])
        rows.append(dict(coarse_row=cr,coarse_col=cc,pixels=audit))
        if not audit['accepted']:
            (root/'failure.json').write_text(json.dumps(dict(status='failed',cells=rows),indent=2))
            raise RuntimeError('Physical allocation audit failed')
    cumulative=hb.as_array(str(root/'cumulative_change_happened.tif'))
    if not np.array_equal(cumulative,np.maximum(data['cumulative'],after!=arrays['before'])):
        raise RuntimeError('Cumulative mask differs from actual changes')
    reconciliation=require_section(root,'reconciliation')
    record=dict(status='accepted',mode='cell_residual_strict',input_signature=data['signature'],cells=rows,
        outputs=candidate['outputs'],budget_sha256=section_sha256(budget),
        reconciliation_sha256=section_sha256(reconciliation),predecessor=data['predecessor'])
    prior=read_tile_record(root).get('accepted')
    if prior is not None:
        if prior!=record:raise RuntimeError('Accepted audit differs; refusing overwrite')
        return
    write_tile_section(root,'accepted',record)


def add_allocation_tasks(p, parent):
    """Register real sibling outputs in execution order, with fail-closed resume."""
    for task in (zone_change_budgets,allocation,allocation_reconciliation,allocation_audit):
        setattr(p,task.__name__+'_task',p.add_task(task,parent=parent,skip_existing=0,creates_dir=False))
