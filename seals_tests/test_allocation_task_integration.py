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
