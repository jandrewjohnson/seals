"""Task-owned failure evidence and independent coarse interpolation outputs."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import pytest
import hazelbean as hb
from seals import allocation_tasks as tasks, signed_allocation as signed, seals_utils
from seals.coarse_interpolation_tasks import coarse_interpolated_states
from test_allocation_task_integration import project,configure_year,raster,run_tile_tasks


def test_failure_evidence_is_owned_by_audit_and_replayable(tmp_path,monkeypatch):
    p=project(tmp_path);configure_year(p,tmp_path,2030,2023)
    def failed(*args):raise RuntimeError('Controlled allocator failure')
    monkeypatch.setattr(signed,'allocate_cell',failed)
    with pytest.raises(RuntimeError,match='exact inputs'):
        run_tile_tasks(p)
    root=Path(p.cur_dir_parent_dir)
    assert not (root/'allocation'/tasks.map_name(p)).exists()
    assert not (root/'allocation_audit/accepted.json').exists()
    assert json.loads((root/'allocation_audit/failure.json').read_text())['status']=='failed'
    with np.load(root/'allocation_audit/failed_cell.npz') as saved:
        assert saved['before'].shape==(10,10)
        assert np.array_equal(saved['target'],[0,20,0,0,-20])
    assert (root/'zone_change_budgets/budgets.json').exists()


def test_changed_budget_cannot_be_consumed(tmp_path):
    p=project(tmp_path);configure_year(p,tmp_path,2030,2023);run_tile_tasks(p)
    root=Path(p.cur_dir_parent_dir);path=root/'zone_change_budgets/budgets.json'
    data=json.loads(path.read_text());data['targets'][1][0][0]=21;path.write_text(json.dumps(data))
    p.cur_dir=str(root/'allocation')
    with pytest.raises(RuntimeError,match='provenance differs'):tasks.allocation(p)


def test_interpolation_task_owns_paths_and_detects_stale_source(tmp_path,monkeypatch):
    source=tmp_path/'coarse_simplified_proportion';out=tmp_path/'coarse_interpolated_states'
    rel=Path('ssp2/rcp70/magpie/baseline_ignore_dependencies')
    template='forest_prop_ssp2_rcp70_magpie_baseline_ignore_dependencies_{}.tif'
    files=[]
    for year,value in [(2020,.2),(2025,.7)]:files.append(raster(source/rel/str(year)/template.format(year),np.array([[value]],dtype='float64'),1))
    p=SimpleNamespace(run_this=True,cur_dir=str(out),coarse_simplified_proportion_dir=str(source),coarse_interpolated_states_dir=str(out),input_dir=str(tmp_path),base_data_dir=str(tmp_path),
       scenarios_df=pd.DataFrame([dict(scenario_type='baseline_ignore_dependencies',exogenous_label='ssp2',climate_label='rcp70',model_label='magpie',counterfactual_label='baseline_ignore_dependencies',key_base_year=2023,coarse_correspondence_path='unused')]))
    p.get_path=lambda *args,**kwargs:'unused'
    monkeypatch.setattr(hb.utils,'get_reclassification_dict_from_df',lambda *a:{'dst_labels':['forest']})
    coarse_interpolated_states(p)
    reference=out/'ssp2/magpie/2023/forest_prop_ssp2_magpie_2023.tif'
    assert np.allclose(hb.as_array(str(reference)),.5)
    assert not (source/'ssp2/magpie/2023').exists()
    assert seals_utils.coarse_state_dir(p,'ssp2','magpie','2023')==str(reference.parent)
    stamp=reference.stat().st_mtime_ns;coarse_interpolated_states(p);assert reference.stat().st_mtime_ns==stamp
    raster(Path(files[1]),np.array([[.8]],dtype='float64'),1)
    with pytest.raises(RuntimeError,match='provenance differs'):coarse_interpolated_states(p)
