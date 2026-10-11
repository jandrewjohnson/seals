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


def read_inputs(p):
    """Read one tile's standard inputs. No raster/CSV writes or downloads."""
    if list(p.changing_class_indices) != [1,2,3,4,5]:
        raise ValueError('Cell-residual tasks require class codes 1..5')
    if p.calibration_parameters_override_dict or p.use_calibration_created_coefficients:
        raise ValueError('Cell-residual task path requires the explicit combined calibration table; overrides need validation')
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
    layers=np.array([hb.load_geotiff_chunk_by_cr_size(str(q),p.global_fine_blocks_list if hb.get_bounding_box(str(q))==hb.global_bounding_box else p.fine_blocks_list) for q in layer_paths],dtype=float)
    base=Path(p.lulc_simplified_paths[p.key_base_year])
    before=hb.load_geotiff_chunk_by_cr_size(str(base),p.fine_blocks_list).astype(np.int64)
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
        audit_path=prior/'allocation_audit/accepted.json'
        audit=json.loads(audit_path.read_text())
        if audit['status']!='accepted':raise RuntimeError('Unaccepted predecessor')
        for name,digest in audit['outputs'].items():
            if hb.hash_file_path(str(prior/'allocation'/name))!=digest:raise RuntimeError('Predecessor output differs from acceptance')
        before=hb.as_array(str(prior/'allocation'/map_name(p,year))).astype(np.int64)
        cumulative=hb.as_array(str(prior/'allocation/cumulative_change_happened.tif')).astype(np.int64)
        predecessor={'audit':str(audit_path),'sha256':hb.hash_file_path(str(audit_path))}
        break
    mask=np.where(cumulative==1,0,mask)
    hectares=hb.load_geotiff_chunk_by_cr_size(p.aoi_ha_per_cell_fine_path,p.fine_blocks_list).astype(float)
    suffix=f'_{p.year}_{p.previous_year}_ha_diff_{p.exogenous_label}_{p.climate_label}_{p.model_label}_{p.counterfactual_label}.tif'
    request_paths=[Path(p.projected_coarse_change_dir)/(c+suffix) for c in labels]
    requests=np.array([hb.load_geotiff_chunk_by_cr_size(str(q),p.coarse_blocks_list) for q in request_paths],dtype=float)
    arrays=dict(requests=requests,before=before,hectares=hectares,mask=mask,layers=layers,coefficients=coefficients,types=types)
    digest=hashlib.sha256((signed.input_signature(*arrays.values())+hb.hash_file_path(__file__)+json.dumps(predecessor,sort_keys=True)).encode()).hexdigest()
    gt=list(hb.get_geotransform_path(str(base)))
    col,row=p.fine_blocks_list[:2];gt[0]+=col*gt[1]+row*gt[2];gt[3]+=col*gt[4]+row*gt[5]
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
    out=root/'budgets.json'
    targets,rows=signed.budget_grid(**data['arrays'])
    if out.exists():
        prior=json.loads(out.read_text())
        if prior['input_signature']!=data['signature'] or prior['targets']!=targets.tolist() or prior['cells']!=rows:
            raise RuntimeError('Budget provenance differs; use a fresh run directory')
        return
    data['coefficients'].to_csv(root/'coefficients.csv',index=False)
    record=dict(input_signature=data['signature'],targets=targets.tolist(),cells=rows,
                predecessor=data['predecessor'],source_paths=data['source_paths'])
    tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(out)


def allocation(p):
    """Write candidate maps using the persisted budget task output.

    A pixel failure is carried to allocation_audit, which owns exact replay
    inputs and raises before stitching or any downstream service work.
    """
    if not p.run_this:return
    data=checked_inputs(p);root=Path(p.cur_dir);root.mkdir(parents=True,exist_ok=True)
    budget_path=root.parent/'zone_change_budgets/budgets.json'
    budget=json.loads(budget_path.read_text())
    if budget['input_signature']!=data['signature']:raise RuntimeError('Budget provenance differs')
    candidate=root/'candidate.json';outputs=[root/map_name(p),root/'cumulative_change_happened.tif']
    if candidate.exists():
        record=json.loads(candidate.read_text())
        if record['input_signature']!=data['signature'] or record['budget_sha256']!=hb.hash_file_path(str(budget_path)) or any(not q.is_file() or hb.hash_file_path(str(q))!=record['outputs'].get(q.name) for q in outputs):
            raise RuntimeError('Candidate provenance differs; refusing reuse')
        p._cell_allocation_failure=None
        return
    if any(q.exists() for q in outputs):raise RuntimeError('Orphaned candidate map; use a fresh run directory')
    arrays=data['arrays'];args={k:v for k,v in arrays.items() if k!='requests'}
    try:
        after,_,changed,pixels=signed.allocate_budget_grid(np.asarray(budget['targets']),**args)
    except signed.AllocationFailure as error:
        p._cell_allocation_failure=(str(root.parent),error)
        return
    p._cell_allocation_failure=None
    cumulative=np.maximum(data['cumulative'],changed)
    for path,values,ndv,dtype in [(outputs[0],after,255,1),(outputs[1],cumulative,-9999,5)]:
        hb.save_array_as_geotiff(values,str(path),data['base'],geotransform_override=data['geotransform'],ndv=ndv,data_type=dtype,compress=True)
    record=dict(status='candidate',input_signature=data['signature'],budget_sha256=hb.hash_file_path(str(budget_path)),
                cells=pixels,outputs={q.name:hb.hash_file_path(str(q)) for q in outputs})
    tmp=candidate.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(candidate)


def allocation_reconciliation(p):
    """Own the original / constrained / realised / eligible-stock CSV."""
    if not p.run_this:return
    failure=getattr(p,'_cell_allocation_failure',None)
    if failure and failure[0]==str(Path(p.cur_dir_parent_dir)):return
    data=checked_inputs(p);root=Path(p.cur_dir);root.mkdir(parents=True,exist_ok=True)
    budget=json.loads((root.parent/'zone_change_budgets/budgets.json').read_text())
    after=hb.as_array(str(root.parent/'allocation'/map_name(p)))
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
    pd.DataFrame(rows).to_csv(root/'reconciliation.csv',index=False)


def allocation_audit(p):
    """Independently check map bytes and physical budgets; own success/failure evidence."""
    if not p.run_this:return
    root=Path(p.cur_dir);root.mkdir(parents=True,exist_ok=True);data=checked_inputs(p)
    failure=getattr(p,'_cell_allocation_failure',None)
    if failure and failure[0]==str(root.parent):
        error=failure[1]
        np.savez_compressed(root/'failed_cell.npz',**error.inputs)
        (root/'failure.json').write_text(json.dumps(dict(status='failed',cell=error.cell,error=str(error),input_signature=data['signature']),indent=2))
        raise RuntimeError(f'Allocation audit failed; exact inputs: {root}/failed_cell.npz') from error
    budget_path=root.parent/'zone_change_budgets/budgets.json';budget=json.loads(budget_path.read_text())
    candidate_path=root.parent/'allocation/candidate.json';candidate=json.loads(candidate_path.read_text())
    if candidate['input_signature']!=data['signature'] or budget['input_signature']!=data['signature'] or candidate['budget_sha256']!=hb.hash_file_path(str(budget_path)):
        raise RuntimeError('Audit input provenance differs')
    for name,digest in candidate['outputs'].items():
        if hb.hash_file_path(str(root.parent/'allocation'/name))!=digest:raise RuntimeError('Candidate output differs')
    after=hb.as_array(str(root.parent/'allocation'/map_name(p)))
    arrays=data['arrays'];targets=np.asarray(budget['targets']);rows=[]
    for cr,cc,w in signed.cell_windows(targets,arrays['before']):
        audit=signed.validate_cell(arrays['before'][w],after[w],arrays['hectares'][w],arrays['mask'][w],targets[:,cr,cc])
        rows.append(dict(coarse_row=cr,coarse_col=cc,pixels=audit))
        if not audit['accepted']:
            (root/'failure.json').write_text(json.dumps(dict(status='failed',cells=rows),indent=2))
            raise RuntimeError('Physical allocation audit failed')
    cumulative=hb.as_array(str(root.parent/'allocation/cumulative_change_happened.tif'))
    if not np.array_equal(cumulative,np.maximum(data['cumulative'],after!=arrays['before'])):
        raise RuntimeError('Cumulative mask differs from actual changes')
    reconciliation=root.parent/'allocation_reconciliation/reconciliation.csv'
    if not reconciliation.is_file():raise RuntimeError('Missing reconciliation task output')
    out=root/'accepted.json'
    record=dict(status='accepted',mode='cell_residual_strict',input_signature=data['signature'],cells=rows,
        outputs=candidate['outputs'],budget_sha256=hb.hash_file_path(str(budget_path)),
        reconciliation_sha256=hb.hash_file_path(str(reconciliation)),predecessor=data['predecessor'])
    if out.exists():
        if json.loads(out.read_text())!=record:raise RuntimeError('Accepted audit differs; refusing overwrite')
        return
    tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(out)


def add_allocation_tasks(p, parent):
    """Register real sibling outputs in execution order, with fail-closed resume."""
    for task in (zone_change_budgets,allocation,allocation_reconciliation,allocation_audit):
        setattr(p,task.__name__+'_task',p.add_task(task,parent=parent,skip_existing=0))
