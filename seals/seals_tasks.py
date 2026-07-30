import os
import hazelbean as hb
from hazelbean import spatial_projection
from hazelbean import pyramids
from seals import seals_utils

def project_aoi(p):
    
    p.ha_per_cell_coarse_path = p.get_path(hb.ha_per_cell_ref_paths[p.coarse_resolution_arcseconds])
    p.ha_per_cell_fine_path = p.get_path(hb.ha_per_cell_ref_paths[p.fine_resolution_arcseconds])
  
    # Process p.aoi to set the regional_vector, bb, bb_exact, and aoi_ha_per_cell_paths
    if p.aoi is not None:
    # if isinstance(p.aoi, str):
        # aoi's value can be a label ('global', an ISO3 code) or a vector path; its
        # name doesn't end in _path so hydration leaves it literal. Resolve
        # path-looking values here (leave_ref_path_if_fail so labels with dots
        # degrade to the label branch instead of raising).
        if isinstance(p.aoi, str) and hb.looks_like_path(p.aoi):
            p.aoi = p.get_path(p.aoi, leave_ref_path_if_fail=True)
        if hb.path_exists(p.aoi):
            p.aoi_path = p.aoi
            p.aoi_label = os.path.splitext(os.path.basename(p.aoi))[0]
            p.bb_exact = spatial_projection.get_bounding_box(p.aoi_path)
            p.bb = pyramids.get_pyramid_compatible_bb_from_vector_and_resolution(p.aoi_path, p.processing_resolution_arcseconds)


        if p.aoi == 'global':
            p.aoi_path = p.regions_vector_path # MISTAKE here where i overrode global. might make standard test seals fail until i update the base data and scenarios. csv
            # p.aoi_path = p.global_regions_vector_path
            p.aoi_label = 'global'
            p.bb_exact = hb.global_bounding_box
            p.bb = p.bb_exact

            p.aoi_ha_per_cell_coarse_path = p.ha_per_cell_coarse_path
            p.aoi_ha_per_cell_fine_path = p.ha_per_cell_fine_path
        
        elif p.regions_column_label is not None and hb.path_exists(p.regions_vector_path):
        # elif isinstance(p.aoi, str):
            gdf = hb.read_vector(p.regions_vector_path) 
            
            if not p.regions_column_label in gdf.columns:
                raise ValueError(f"The column '{p.regions_column_label}' is not found in the regions vector file: {p.regions_vector_path}. Please check the column name or the vector file.")


            if p.aoi == 'from_regional_projections_input_path':
                # Read the csv to get the unique AOI values
                df = hb.df_read(p.regional_projections_input_path)
                unique_aois = list(df[p.regions_column_label].unique())
                filter_column = p.regions_column_label
                filter_value = unique_aois
                p.aoi_path = os.path.join(p.cur_dir, 'aoi_' + str(p.aoi) + '.gpkg')
                p.aoi_label = p.aoi     
            else:
                
                p.aoi_path = os.path.join(p.cur_dir, 'aoi_' + str(p.aoi) + '.gpkg')
                p.aoi_label = p.aoi            
            
                filter_column = p.regions_column_label # if it's exactly 3 characters, assume it's an ISO3 code.
                filter_value = p.aoi
               
            
            for current_aoi_path in hb.list_filtered_paths_nonrecursively(p.cur_dir, include_strings='aoi'):
                if current_aoi_path != p.aoi_path:
                    hb.log('There is more than one AOI in the current directory. This means you are trying to run a project in a new area of interst in a project that was already run in a different area of interest. This is not allowed! You probably want to create a new project directory and set the p = hb.ProjectFlow(...) line to point to the new directory.')

            if not hb.path_exists(p.aoi_path):
                hb.extract_features_in_shapefile_by_attribute(p.regions_vector_path, p.aoi_path, filter_column, filter_value)

            p.bb_exact = spatial_projection.get_bounding_box(p.aoi_path)
            p.bb = pyramids.get_pyramid_compatible_bb_from_vector_and_resolution(p.aoi_path, p.processing_resolution_arcseconds)
                           
            # Create a PROJECT-SPECIFIC version of these clipped ones.
            p.aoi_ha_per_cell_fine_path = os.path.join(p.cur_dir, 'pyramids', 'aoi_ha_per_cell_fine.tif')
            if not hb.path_exists(p.aoi_ha_per_cell_fine_path):
                hb.create_directories(p.aoi_ha_per_cell_fine_path)
                
                #  make ha_per_cell_paths not be a dict but a project level ha_per_cell_fine_path etc
                hb.clip_raster_by_bb(p.ha_per_cell_fine_path, p.bb, p.aoi_ha_per_cell_fine_path)
            
            p.aoi_ha_per_cell_coarse_path = os.path.join(p.cur_dir, 'pyramids', 'aoi_ha_per_cell_coarse.tif')
            if not hb.path_exists(p.aoi_ha_per_cell_coarse_path):
                hb.create_directories(p.aoi_ha_per_cell_coarse_path)
                hb.clip_raster_by_bb(p.ha_per_cell_coarse_path, p.bb, p.aoi_ha_per_cell_coarse_path)
        
            
            
        else:
            p.bb_exact = hb.spatial_projection.get_bounding_box(p.aoi_path)
            p.bb = hb.pyramids.get_pyramid_compatible_bb_from_vector_and_resolution(p.aoi_path, p.processing_resolution_arcseconds)

       
        if p.aoi == 'global':
            p.aoi_ha_per_cell_fine_path = p.get_path(pyramids.pyramid_ha_per_cell_ref_paths[p.fine_resolution_arcseconds])
            p.aoi_ha_per_cell_coarse_path = p.get_path(pyramids.pyramid_ha_per_cell_ref_paths[p.coarse_resolution_arcseconds])

        else:     # Create a PROJECT-SPECIFIC version of these clipped ones.
            p.aoi_ha_per_cell_fine_path = os.path.join(p.cur_dir, 'pyramids', 'aoi_ha_per_cell_fine.tif')
            p.aoi_ha_per_cell_coarse_path = os.path.join(p.cur_dir, 'pyramids', 'aoi_ha_per_cell_coarse.tif')
        
        if not hb.path_exists(p.aoi_ha_per_cell_fine_path):
            hb.create_directories(p.aoi_ha_per_cell_fine_path)
            cur_path = p.get_path(hb.ha_per_cell_ref_paths[p.fine_resolution_arcseconds])
            hb.clip_raster_by_bb(cur_path, p.bb, p.aoi_ha_per_cell_fine_path)
        
        
        if not hb.path_exists(p.aoi_ha_per_cell_coarse_path):
            hb.create_directories(p.aoi_ha_per_cell_coarse_path)
            cur_path = p.get_path(hb.ha_per_cell_ref_paths[p.coarse_resolution_arcseconds])
            hb.clip_raster_by_bb(cur_path, p.bb, p.aoi_ha_per_cell_coarse_path)
                    
    else:
        raise NameError('Unable to interpret p.aoi.')


def seals(p):
    # Folder creation task, but with some skipper logic    
    if p.run_this:
        expected_paths = []
        for index, row in p.scenarios_df.iterrows():       
            seals_utils.assign_df_row_to_object_attributes(p, row)    
            
            for year in p.seals_years:
                if p.scenario_type != 'baseline':
                    stitched_output_name = 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(year)
                    expected_path = os.path.join(p.cur_dir, 'stitched_lulc_simplified_scenarios', stitched_output_name + '.tif')
                    expected_paths.append(expected_path)
        skip_all = True
        for path in expected_paths:
            if not hb.path_exists(path):
                skip_all = False
                break
            # 'C:/Users/jajohns/Files/gtap_invest/projects/ngfs/ngfs_pnas/intermediate/seals/stitched_lulc_simplified_scenarios/lulc_esa_seals7_ssp2_rcp45_ngfs-remind-magpie_baseline_ignore_dependencies_2050.tif'
            # 'C:/Users/jajohns/Files/gtap_invest/projects/ngfs/ngfs_pnas/intermediate/seals/stitched_lulc_simplified_scenarios/lulc_esa_seals7_ssp2_rcp45_ngfs-remind-magpie_baseline_ignore_dependencies_2050.tif'
        if skip_all:
            p.skip_children = True
        else:
            p.skip_children = False
        # if all([hb.path_exists(path) for path in expected_paths]):
        #     p.skip_children = True
            # Then skip the rest of the project because the base data isn't ready, which means the rest of the project can't be run without errors. This is a bit of a hacky way to do this, but it allows us to avoid having to write a bunch of skip logic in each individual task.

    pass    