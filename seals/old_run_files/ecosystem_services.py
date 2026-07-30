import os
import hazelbean as hb
import numpy as np
import pandas as pd
import logging
import pygeoprocessing
import csv
import seals_utils
from osgeo import gdal, ogr
from natcap.invest import pollination, carbon
from pyproj import CRS
from typing import Dict, List, Tuple, Optional
import traceback
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm, TwoSlopeNorm
import seaborn as sns
import rasterio
import geopandas as gpd
from collections import Counter


# import warnings
# # Suppress the specific OGR RuntimeWarning about inserting MULTIPOLYGON into POLYGON layers
# warnings.filterwarnings(
#     "ignore",
#     message=r"A geometry of type MULTIPOLYGON is inserted into layer .* of geometry type POLYGON.*",
#     category=RuntimeWarning,
#     module=r"osgeo\.ogr"
# )
# # Suppress other noisy osgeo RuntimeWarnings emitted during IO/cleanup
# warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"osgeo\..*")

L = hb.get_logger()
# gdal.UseExceptions()
ogr.DontUseExceptions()
gdal.DontUseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')

def reproject_lulc_rasters_to_equal_area(p):
    """
    Preprocessing step: Reproject all LULC rasters to Equal Earth projection.
    This creates a shared set of reprojected rasters that can be used by all InVEST models.
    """

    def get_lulc_paths(scenario_type, **kwargs):
        """Helper function to get LULC paths based on scenario type"""
        if scenario_type == 'baseline':
            year = kwargs.get('year')
            scenario_name = f'baseline_{year}'
            if hasattr(p, 'alt_base_lulc_path'):
                lulc_path = p.alt_base_lulc_path
            else:
                lulc_path = os.path.join(p.fine_processed_inputs_dir, 'lulc', 'esa', 'seals7', f'lulc_esa_seals7_{year}.tif')
        else:
            year = kwargs.get('year')
            scenario_name = f'{p.lulc_src_label}_{p.lulc_simplification_label}_{p.exogenous_label}_{p.climate_label}_{p.model_label}_{p.counterfactual_label}_{year}'
            lulc_path = os.path.join(p.stitched_lulc_simplified_scenarios_dir, f'lulc_{scenario_name}_clipped.tif')
        
        return lulc_path, scenario_name
    
    # Track which rasters have been processed to avoid duplicates
    processed_rasters = set()
    
    # Process all scenarios
    for index, row in p.scenarios_df.iterrows():
        seals_utils.assign_df_row_to_object_attributes(p, row)
        
        if p.scenario_type == 'baseline':
            lulc_path, scenario_name = get_lulc_paths('baseline', year=p.base_years[0])
            reprojected_path = os.path.join(p.cur_dir, f'lulc_{scenario_name}.tif')
            
            if scenario_name not in processed_rasters and not os.path.exists(reprojected_path):
                hb.log(f'Reprojecting baseline LULC: {scenario_name}')
                project_to_equal_area(lulc_path, reprojected_path)
                processed_rasters.add(scenario_name)
        else:
            for year in p.years:
                lulc_path, scenario_name = get_lulc_paths('future', year=year)
                reprojected_path = os.path.join(p.cur_dir, f'lulc_{scenario_name}.tif')
                
                if scenario_name not in processed_rasters and not os.path.exists(reprojected_path):
                    hb.log(f'Reprojecting future LULC: {scenario_name}')
                    project_to_equal_area(lulc_path, reprojected_path)
                    processed_rasters.add(scenario_name)


def project_to_equal_area(input_raster_path, output_raster_path, target_pixel_size=(300, -300)):
    """
    Reproject a raster to Equal Earth projection (EPSG:8857).
    
    Args:
        input_raster_path (str): Path to input raster
        output_raster_path (str): Path for output reprojected raster
        target_pixel_size (tuple): Target pixel size (x, y) in meters
    """
    # Define projection in Equal Earth (EPSG:8857)
    wkt_projection = CRS.from_epsg(8857).to_wkt()
    
    # Check if input file exists
    if not os.path.exists(input_raster_path):
        raise FileNotFoundError(f"Input raster not found: {input_raster_path}")
    
    # Reproject raster
    pygeoprocessing.warp_raster(
        base_raster_path=input_raster_path,
        target_pixel_size=target_pixel_size,
        target_raster_path=output_raster_path,
        resample_method='nearest',
        target_projection_wkt=wkt_projection,
    )


def run_invest_pollination(p):
    """
    Generate pollination sufficiency from Land Use/Land Cover (LULC) data
    using NatCap's pollination module.
    Uses pre-reprojected LULC rasters from the reprojected_lulc directory.
    - Input raster: LULC (categorical)
    - Output raster: Pollination sufficiency index (scale 0-1)
    """
    
    def get_reprojected_lulc_path(scenario_type, **kwargs):
        """Helper function to get reprojected LULC paths"""
        if scenario_type == 'baseline':
            year = kwargs.get('year')
            scenario_name = f'baseline_{year}'
        else:
            year = kwargs.get('year')
            scenario_name = f'{p.lulc_src_label}_{p.lulc_simplification_label}_{p.exogenous_label}_{p.climate_label}_{p.model_label}_{p.counterfactual_label}_{year}'
        
        reprojected_path = os.path.join(p.reproject_lulc_rasters_to_equal_area_dir, f'lulc_{scenario_name}.tif')
        return reprojected_path, scenario_name

    def run_pollination_for_scenario(reprojected_path, scenario_name):
        """Helper function to run pollination analysis for a single scenario"""
        
        # Check that reprojected file exists
        if not os.path.exists(reprojected_path):
            raise FileNotFoundError(f"Reprojected LULC file not found: {reprojected_path}")
        
        p.pollination_args = {
            'farm_vector_path': '',
            'guild_table_path': p.get_path(os.path.join('global_invest', 'pollination', 'guild_table.csv')),
            'landcover_biophysical_table_path': p.get_path(os.path.join('global_invest', 'pollination', 'landcover_biophysical_table.csv')),
            'landcover_raster_path': '',
            'n_workers': '-1',
            'results_suffix': '',
            'workspace_dir': '',
        }
        
        # Define InVEST pollination model output directories
        pollination_workspace_dir = os.path.join(p.cur_dir, scenario_name)
        
        # Run InVEST pollination model if final output does not exist
        if not hb.path_exists(os.path.join(pollination_workspace_dir, 'total_pollinator_abundance_spring.tif')):
            p.pollination_args['workspace_dir'] = pollination_workspace_dir
            p.pollination_args['landcover_raster_path'] = reprojected_path
            pollination.execute(p.pollination_args)

    # Process all scenarios
    for index, row in p.scenarios_df.iterrows():
        seals_utils.assign_df_row_to_object_attributes(p, row)
        hb.log(f'Running InVEST pollination module for scenario {index+1} of {len(p.scenarios_df)}')
        
        if p.scenario_type == 'baseline':
            # Handle baseline scenario
            reprojected_path, scenario_name = get_reprojected_lulc_path('baseline', year=p.base_years[0])
            run_pollination_for_scenario(reprojected_path, scenario_name)
        else:
            # Handle future scenarios (loop through years)
            for year in p.years:
                reprojected_path, scenario_name = get_reprojected_lulc_path('future', year=year)
                run_pollination_for_scenario(reprojected_path, scenario_name)

def run_invest_carbon(p):
    """
    Generate carbon storage estimates from Land Use/Land Cover (LULC) data
    using NatCap's carbon storage module.
    Uses pre-reprojected LULC rasters from the reprojected_lulc directory.
    - Input raster: LULC (categorical)
    - Output raster: Carbon storage estimates
    """
    
    # Path to reprojected rasters (created by reproject_lulc_rasters function)
    def get_reprojected_lulc_path(scenario_type, **kwargs):
        """Helper function to get reprojected LULC paths"""
        if scenario_type == 'baseline':
            year = kwargs.get('year')
            scenario_name = f'baseline_{year}'
        else:
            year = kwargs.get('year')
            scenario_name = f'{p.lulc_src_label}_{p.lulc_simplification_label}_{p.exogenous_label}_{p.climate_label}_{p.model_label}_{p.counterfactual_label}_{year}'
        
        reprojected_path = os.path.join(p.reproject_lulc_rasters_to_equal_area_dir, f'lulc_{scenario_name}.tif')
        return reprojected_path, scenario_name

    def run_carbon_for_scenario(reprojected_path, scenario_name):
        """Helper function to run carbon analysis for a single scenario"""
        p.carbon_args = {
            'calc_sequestration': False,
            'carbon_pools_path': p.get_path(os.path.join('global_invest', 'carbon', 'seals_biophysical_table.csv')),
            'discount_rate': '',
            'do_redd': False,
            'do_valuation': False,
            'lulc_cur_path': '',
            'lulc_cur_year': '',
            'lulc_fut_path': '',
            'lulc_fut_year': '',
            'lulc_redd_path': '',
            'n_workers': '-1',
            'price_per_metric_ton_of_c': '',
            'rate_change': '',
            'results_suffix': '',
            'workspace_dir': '',
        }
        
        # Check that reprojected file exists
        if not os.path.exists(reprojected_path):
            raise FileNotFoundError(f"Reprojected LULC file not found: {reprojected_path}")
        
        # Define InVEST carbon model output directories
        carbon_workspace_dir = os.path.join(p.cur_dir, scenario_name)
        
        # Run InVEST carbon model if final output does not exist
        # Check for a specific carbon output file instead of just the directory
        carbon_output_file = os.path.join(carbon_workspace_dir, 'tot_c_cur.tif')
        if not hb.path_exists(carbon_output_file):
            p.carbon_args['workspace_dir'] = carbon_workspace_dir
            p.carbon_args['lulc_cur_path'] = reprojected_path
            carbon.execute(p.carbon_args)

    # Process all scenarios
    for index, row in p.scenarios_df.iterrows():
        seals_utils.assign_df_row_to_object_attributes(p, row)
        hb.log(f'Running InVEST carbon module for scenario {index+1} of {len(p.scenarios_df)}')
        
        if p.scenario_type == 'baseline':
            # Handle baseline scenario
            reprojected_path, scenario_name = get_reprojected_lulc_path('baseline', year=p.base_years[0])
            run_carbon_for_scenario(reprojected_path, scenario_name)
        else:
            # Handle future scenarios (loop through years)
            for year in p.years:
                reprojected_path, scenario_name = get_reprojected_lulc_path('future', year=year)
                run_carbon_for_scenario(reprojected_path, scenario_name)



def calculate_pollinator_adjusted_value(lulc, poll_suff, crop_value_max_lost, crop_value_baseline, year, sufficient_pollination_threshold, L):
    """
    Compute crop value adjusted for pollination sufficiency.
    
    - Inputs:
        - lulc: Land Use/Land Cover raster (categorical)
        - poll_suff: Pollination sufficiency raster (scale 0-1)
        - crop_value_max_lost: Maximum economic loss due to pollination loss ($/ha)
        - crop_value_baseline: Baseline crop value ($/ha)
    - Output:
        - Adjusted crop value raster array ($/ha)
    """
    L.info(f'Calculating pollination-adjusted crop value for {year}')
    return np.where(
        (crop_value_max_lost > 0) & (poll_suff < sufficient_pollination_threshold) & (lulc == 2),
        crop_value_baseline - crop_value_max_lost * (1 - (1 / sufficient_pollination_threshold) * poll_suff),
        np.where(
            (crop_value_max_lost > 0) & (poll_suff >= sufficient_pollination_threshold) & (lulc == 2),
            crop_value_baseline,
            -9999.
        )
    )


def calculate_crop_value_and_shock(p):
    """
    Calculate crop value adjusted for pollination sufficiency and compute regional shock values.
    """
    # Threshold for sufficient pollination
    sufficient_pollination_threshold = 0.3

    def get_scenario_name(scenario_type, **kwargs):
        """Generate consistent scenario names"""
        if scenario_type == 'baseline':
            year = kwargs.get('year')
            return f'baseline_{year}'
        else:
            year = kwargs.get('year')
            return f'{p.lulc_src_label}_{p.lulc_simplification_label}_{p.exogenous_label}_{p.climate_label}_{p.model_label}_{p.counterfactual_label}_{year}'
    
    def get_lulc_path(scenario_type, **kwargs):
        """Get LULC path based on scenario type"""
        if scenario_type == 'baseline':
            if hasattr(p, 'alt_base_lulc_path'):
                return p.alt_base_lulc_path
            else:
                year = kwargs.get('year')
                lulc_name = f'lulc_{p.lulc_src_label}_{p.lulc_simplification_label}_{year}'
                return os.path.join(p.fine_processed_inputs_dir, 'lulc', p.lulc_src_label, p.lulc_simplification_label, f'{lulc_name}.tif')
        else:
            year = kwargs.get('year')
            lulc_name = f'lulc_{p.lulc_src_label}_{p.lulc_simplification_label}_{p.exogenous_label}_{p.climate_label}_{p.model_label}_{p.counterfactual_label}_{year}'
            return os.path.join(p.stitched_lulc_simplified_scenarios_dir, f'{lulc_name}.tif')
    
    def resample_rasters_to_reference(raster_pairs, reference_path):
        """Resample multiple rasters to match reference raster properties"""
        raster_info = pygeoprocessing.get_raster_info(reference_path)
        target_pixel_size = raster_info['pixel_size']
        target_projection_wkt = raster_info['projection_wkt']
        target_bb = raster_info['bounding_box']
        
        for base_path, resampled_path in raster_pairs:
            if not os.path.exists(resampled_path):
                pygeoprocessing.warp_raster(
                    base_raster_path=base_path,
                    target_pixel_size=target_pixel_size,
                    target_raster_path=resampled_path,
                    resample_method='nearest',
                    target_projection_wkt=target_projection_wkt,
                    target_bb=target_bb,
                )
    
    # Step 1: Calculate baseline crop values (if not already done)
    ref_raster = p.get_path(os.path.join('crops', 'production', 'alfalfa_HarvAreaYield_Geotiff', 'alfalfa_Production.tif'))
    crop_value_baseline_path = os.path.join(p.cur_dir, 'crop_value_baseline.tif')
    crop_value_max_lost_path = os.path.join(p.cur_dir, 'crop_value_max_lost.tif')
    
    if not os.path.exists(crop_value_baseline_path) or not os.path.exists(crop_value_max_lost_path):
        # Load crop dependence data
        pollination_dependence_spreadsheet_input_path = os.path.join(p.base_data_dir, 'crops', 'rspb20141799supp3.xls')
        df_dependence = pd.read_excel(pollination_dependence_spreadsheet_input_path, sheet_name='Crop nutrient content')
        crop_names = list(df_dependence['Crop map file name'])[:-3]
        pollination_dependence = list(df_dependence['poll.dep'])
        
        # Initialize arrays
        ha_shape = hb.get_shape_from_dataset_path(ref_raster)
        crop_value_baseline = np.zeros(ha_shape)
        crop_value_no_pollination = np.zeros(ha_shape)
        
        # Calculate crop values
        for c, crop_name in enumerate(crop_names):
            # L.info(f'Calculating crop value for {crop_name} with pollination dependence {pollination_dependence[c]}')
            crop_yield_path = os.path.join(p.base_data_dir, 'crops', 'production', f'{crop_name}_HarvAreaYield_Geotiff', f'{crop_name}_Production.tif')
            crop_yield = hb.as_array(crop_yield_path)
            crop_yield = np.where(crop_yield > 0, crop_yield, 0.0)
            
            crop_value_baseline += crop_yield
            crop_value_no_pollination += crop_yield * (1 - float(pollination_dependence[c]))
        
        # Save results
        crop_value_max_lost = crop_value_baseline - crop_value_no_pollination
        hb.save_array_as_geotiff(crop_value_baseline, crop_value_baseline_path, ref_raster, ndv=-9999, data_type=6)
        hb.save_array_as_geotiff(crop_value_max_lost, crop_value_max_lost_path, ref_raster, ndv=-9999, data_type=6)
    
    # Step 2: Process scenarios
    combined_data = [] 
    alt_combined_data = []
    
    for index, row in p.scenarios_df.iterrows():
        seals_utils.assign_df_row_to_object_attributes(p, row)
        
        # Process both baseline and scenario types
        if p.scenario_type == 'baseline':
            continue
        L.info(f'Processing scenario: {p.counterfactual_label}')
        
        for year in p.years:
            # Get consistent naming
            baseline_name = get_scenario_name('baseline', year=p.base_years[0])
            scenario_name = get_scenario_name('scenario', year=year)
            
            # Define paths using consistent naming
            baseline_lulc_path = get_lulc_path('baseline', year=p.base_years[0])
            scenario_lulc_path = get_lulc_path('scenario', year=year)
            
            # Pollination paths
            poll_baseline_path = os.path.join(p.run_invest_pollination_dir, baseline_name, 'total_pollinator_abundance_spring.tif')
            poll_scenario_path = os.path.join(p.run_invest_pollination_dir, scenario_name, 'total_pollinator_abundance_spring.tif')
            
            # Check if required input files exist
            required_files = [baseline_lulc_path, scenario_lulc_path, poll_baseline_path, poll_scenario_path]
            missing_files = [f for f in required_files if not os.path.exists(f)]
            if missing_files:
                L.warning(f'Missing required files for year {year}: {missing_files}')
                continue
                
            # Check if output files already exist, if so, skip this year
            baseline_output_path = os.path.join(p.cur_dir, f'crop_value_adjusted_{baseline_name}.tif')
            scenario_output_path = os.path.join(p.cur_dir, f'crop_value_adjusted_{scenario_name}.tif')
            if os.path.exists(baseline_output_path) and os.path.exists(scenario_output_path):
                L.info(f"Skipping year {year} for scenario {p.counterfactual_label} — outputs already exist.")
                continue
            
            # Resampled paths
            resampled_crop_baseline_path = os.path.join(p.cur_dir, 'resampled_crop_value_baseline.tif')
            resampled_crop_max_lost_path = os.path.join(p.cur_dir, 'resampled_crop_value_max_lost.tif')
            resampled_poll_baseline_path = os.path.join(p.cur_dir, f'resampled_poll_{baseline_name}.tif')
            resampled_poll_scenario_path = os.path.join(p.cur_dir, f'resampled_poll_{scenario_name}.tif')
            
            # Resample all rasters to baseline LULC resolution
            raster_pairs = [
                (crop_value_baseline_path, resampled_crop_baseline_path),
                (crop_value_max_lost_path, resampled_crop_max_lost_path),
                (poll_baseline_path, resampled_poll_baseline_path),
                (poll_scenario_path, resampled_poll_scenario_path),
            ]
            resample_rasters_to_reference(raster_pairs, baseline_lulc_path)
            
            # Define output paths
            baseline_output_path = os.path.join(p.cur_dir, f'crop_value_adjusted_{baseline_name}.tif')
            scenario_output_path = os.path.join(p.cur_dir, f'crop_value_adjusted_{scenario_name}.tif')
            
            # Always calculate shock values (remove the file existence check)
            # Load arrays
            crop_value_max_lost = hb.as_array(resampled_crop_max_lost_path)
            crop_value_baseline = hb.as_array(resampled_crop_baseline_path)
            poll_baseline = hb.as_array(resampled_poll_baseline_path)
            poll_scenario = hb.as_array(resampled_poll_scenario_path)
            lulc_baseline = hb.as_array(baseline_lulc_path)
            lulc_scenario = hb.as_array(scenario_lulc_path)

            # Calculate pollination-adjusted crop values
            crop_value_baseline_adjusted = calculate_pollinator_adjusted_value(
                lulc_baseline, poll_baseline, crop_value_max_lost, crop_value_baseline, 
                'baseline', sufficient_pollination_threshold, L
            )

            crop_value_scenario_adjusted = calculate_pollinator_adjusted_value(
                lulc_scenario, poll_scenario, crop_value_max_lost, crop_value_baseline, 
                year, sufficient_pollination_threshold, L
            )

            # Save adjusted crop values (only if they don't exist)
            if not os.path.exists(baseline_output_path):
                hb.save_array_as_geotiff(crop_value_baseline_adjusted, baseline_output_path, baseline_lulc_path, ndv=-9999, data_type=6)
            if not os.path.exists(scenario_output_path):
                hb.save_array_as_geotiff(crop_value_scenario_adjusted, scenario_output_path, scenario_lulc_path, ndv=-9999, data_type=6)
        
            # Calculate regional shock values
            L.info('Calculating shock value by region')
            
            # Debug info for crop value arrays
            L.info(f'Crop value scenario array shape: {crop_value_scenario_adjusted.shape}')
            L.info(f'Crop value scenario array info:')
            L.info(pygeoprocessing.get_raster_info(scenario_output_path))
            
            # Load and check regions array
            L.info(f'Loading regions from: {p.regions_path}')
            region_array = hb.as_array(p.regions_path)
            L.info(f'Region array shape: {region_array.shape}')
            L.info(f'Region array info:')
            L.info(pygeoprocessing.get_raster_info(p.regions_path))
            
            unique_regions = np.unique(region_array)
            L.info(f'Unique regions found: {unique_regions}')

            for region in unique_regions:
                if region in (-1, 255):  # Skip nodata values
                    continue
                
                # mask = region_array == region
                # valid_data_mask = (crop_value_scenario_adjusted != -9999) & (crop_value_baseline_adjusted != -9999) & mask
                # scenario_sum = np.sum(crop_value_scenario_adjusted[valid_data_mask])
                # baseline_sum = np.sum(crop_value_baseline_adjusted[valid_data_mask])
                mask = region_array == region
                scenario_sum = np.sum(crop_value_scenario_adjusted[mask])
                baseline_sum = np.sum(crop_value_baseline_adjusted[mask])
                
                shock_value = np.nan if baseline_sum == 0 else scenario_sum / baseline_sum
                # Use full scenario name instead of just counterfactual_label
                scenario_name = f"{p.lulc_src_label}_{p.lulc_simplification_label}_{p.exogenous_label}_{p.climate_label}_{p.model_label}_{p.counterfactual_label}"
                combined_data.append([year, region, scenario_name, shock_value])
                
                # Add debug logging
                L.info(f'Region {region}: baseline_sum={baseline_sum}, scenario_sum={scenario_sum}, shock_value={shock_value}')

            # If alternative regions path exists, calculate shocks for those regions too
            if hasattr(p, 'alt_regions_path') and p.alt_regions_path:
                L.info('Calculating shock value by alternative regions')
                L.info(f'Loading alternative regions from: {p.alt_regions_path}')
                alt_regions_array = hb.as_array(p.alt_regions_path)
                L.info(f'Alternative region array shape: {alt_regions_array.shape}')
                L.info(f'Alternative region array info:')
                L.info(pygeoprocessing.get_raster_info(p.alt_regions_path))
                
                alt_unique_regions = np.unique(alt_regions_array)
                L.info(f'Unique alternative regions found: {alt_unique_regions}')
                L.info(f'Found alternative regions: {alt_unique_regions}')

                for region in alt_unique_regions:
                    if region in (-1, 255):  # Skip nodata values
                        continue
                    
                    # mask = alt_regions_array == region
                    # valid_data_mask = (crop_value_scenario_adjusted != -9999) & (crop_value_baseline_adjusted != -9999) & mask
                    # scenario_sum = np.sum(crop_value_scenario_adjusted[valid_data_mask])
                    # baseline_sum = np.sum(crop_value_baseline_adjusted[valid_data_mask])
                    mask = alt_regions_array == region
                    scenario_sum = np.sum(crop_value_scenario_adjusted[mask])
                    baseline_sum = np.sum(crop_value_baseline_adjusted[mask])

                    shock_value = np.nan if baseline_sum == 0 else scenario_sum / baseline_sum
                    # Use full scenario name instead of just counterfactual_label
                    scenario_name = f"{p.lulc_src_label}_{p.lulc_simplification_label}_{p.exogenous_label}_{p.climate_label}_{p.model_label}_{p.counterfactual_label}"
                    alt_combined_data.append([year, region, scenario_name, shock_value])

    # Save combined shock values outside the loops
    master_csv_path = os.path.join(p.cur_dir, f'pollination_shocks_{p.lulc_simplification_label}_{p.exogenous_label}_{p.climate_label}_{p.model_label}.csv')
    
    L.info(f'Total data points to write: {len(combined_data)}')
    
    with open(master_csv_path, 'w', newline='') as master_file:
        writer = csv.writer(master_file)
        writer.writerow(['Year', 'Region', 'Scenario', 'Shock_Value'])
        writer.writerows(combined_data)
    
    L.info(f'Combined shock values saved to: {master_csv_path}')
    
    # If alternative region data, save it
    if len(alt_combined_data) > 0:
        # Save alternative regions shock values
        alt_master_csv_path = os.path.join(p.cur_dir, f'pollination_shocks_alt_{p.lulc_simplification_label}_{p.exogenous_label}_{p.climate_label}_{p.model_label}.csv')
        L.info(f'Total alternative region data points to write: {len(alt_combined_data)}')
        with open(alt_master_csv_path, 'w', newline='') as alt_master_file:
            writer = csv.writer(alt_master_file)
            writer.writerow(['Year', 'Region', 'Scenario', 'Shock_Value'])
            writer.writerows(alt_combined_data)
        L.info(f'Alternative region shock values saved to: {alt_master_csv_path}')
    return combined_data


def calculate_biodiversity_index(p):
    """
    Calculate biodiversity index from LULC data.
    
    This function processes land use/land cover (LULC) data to calculate a comprehensive
    biodiversity index based on species richness, red list species, endemic species,
    key biodiversity areas (KBAs), and ecoregion diversity.
    
    Args:
        p: Parameter object containing configuration and file paths
        
    The function expects input rasters in p.base_data_dir/gtap-biodiversity/InputRasters/
    including species data for Amphibians, Birds, Mammals, and Reptiles.
    """
    
    # Configuration constants
    GLOBAL_RICHNESS = {
        'Amphibians': 6631,
        'Birds': 10424,
        'Mammals': 5709,
        'Reptiles': 6416
    }
    TAXA_TYPES = ['Amphibians', 'Birds', 'Mammals', 'Reptiles']
    NODATA_VALUE = -9999.0
    
    # Excluded LULC classes (urban, cropland, other)
    EXCLUDED_LULC_CLASSES = [1, 2, 7]
    
    # Biodiversity layer weights
    LAYER_WEIGHTS = {
        'species_richness': 1.0,
        'red_list': 1.0,
        'kba': 0.5,
        'endemic': 1.0,
        'ecoregion': 1.0
    }
    
    logging.info("Starting biodiversity index calculation")
    
    # Setup directories and file paths
    input_dir, output_dir, raster_paths = _setup_directories_and_paths(p, TAXA_TYPES)
    
    # Get template raster info for alignment
    aoi_path = os.path.join(p.project_aoi_dir, 'pyramids', 'aoi_ha_per_cell_fine.tif')
    target_info = _get_target_raster_info(aoi_path)
    
    # Align and resample all input rasters
    _align_rasters(raster_paths, output_dir, target_info)
    
    # Get file paths for processing
    file_paths = _get_aligned_file_paths(output_dir, TAXA_TYPES)
    
    # Process each scenario
    for index, row in p.scenarios_df.iterrows():
        seals_utils.assign_df_row_to_object_attributes(p, row)
            
        # Process each year in the scenario
        for year in p.years:
            scenario_name, lulc_path = _get_scenario_paths(p, year)
            
            if not os.path.exists(lulc_path):
                raise FileNotFoundError(f"LULC file not found: {lulc_path}")
                
            logging.info(f"Processing scenario: {scenario_name}")
            
            # Create scenario output directory
            scenario_dir = os.path.join(p.cur_dir, scenario_name)
            
            # Check if scenario already processed (look for final index file)
            final_index_path = os.path.join(scenario_dir, f'{scenario_name}_index.tif')
            if os.path.exists(final_index_path):
                logging.info(f"Skipping existing scenario: {scenario_name}")
                continue
            
            # Create directory for processing
            os.makedirs(scenario_dir, exist_ok=True)
            
            # Generate binary LULC suitability layer
            binary_path = _create_binary_lulc(lulc_path, scenario_dir, scenario_name, 
                                            EXCLUDED_LULC_CLASSES, NODATA_VALUE)
            
            # Calculate species-based indices
            species_paths = _calculate_species_indices(
                binary_path, file_paths, scenario_dir, scenario_name, 
                TAXA_TYPES, GLOBAL_RICHNESS, NODATA_VALUE
            )
            
            # Calculate KBA index
            kba_path = _calculate_kba_index(
                binary_path, file_paths['kba'], scenario_dir, scenario_name, NODATA_VALUE
            )
            
            # Calculate ecoregion index
            ecoregion_path = _calculate_ecoregion_index(
                binary_path, file_paths['ecoregion'], scenario_dir, scenario_name, NODATA_VALUE
            )
            
            # Calculate final weighted biodiversity index
            _calculate_final_index(
                binary_path, species_paths, kba_path, ecoregion_path,
                scenario_dir, scenario_name, LAYER_WEIGHTS, NODATA_VALUE
            )
    
    logging.info("Biodiversity index calculation completed")


def _setup_directories_and_paths(p, taxa_types: List[str]) -> Tuple[str, str, List[Tuple[str, str]]]:
    """Setup input/output directories and create list of raster paths to process."""
    input_dir = p.get_path(os.path.join('global_invest', 'biodiversity'))
    output_dir = os.path.join(p.cur_dir, 'clipped_biodiversity_inputs')
    
    raster_paths = []
    
    # Species pool files
    for taxa in taxa_types:
        for suffix in ['', 'RL', 'Endemic']:
            filename = f'{taxa}{suffix}.tif'
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            raster_paths.append((input_path, output_path))
    
    # Additional layers
    additional_files = ['KBAs.tif', 'ecoMaps.tif']
    for filename in additional_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        raster_paths.append((input_path, output_path))
    
    return input_dir, output_dir, raster_paths


def _get_target_raster_info(aoi_path: str) -> Dict:
    """Extract target raster information for alignment."""
    raster_info = pygeoprocessing.get_raster_info(aoi_path)
    return {
        'pixel_size': raster_info['pixel_size'],
        'projection_wkt': raster_info['projection_wkt'],
        'bounding_box': raster_info['bounding_box']
    }


def _align_rasters(raster_paths: List[Tuple[str, str]], output_dir: str, target_info: Dict):
    """Align and resample all input rasters to match target specifications."""
    os.makedirs(output_dir, exist_ok=True)
    
    for base_path, resampled_path in raster_paths:
        if os.path.exists(resampled_path):
            continue
            
        if not os.path.exists(base_path):
            logging.warning(f"Input raster not found: {base_path}")
            continue
            
        try:
            pygeoprocessing.warp_raster(
                base_raster_path=base_path,
                target_pixel_size=target_info['pixel_size'],
                target_raster_path=resampled_path,
                resample_method='nearest',
                target_projection_wkt=target_info['projection_wkt'],
                target_bb=target_info['bounding_box']
            )
        except Exception as e:
            logging.error(f"Failed to warp raster {base_path}: {e}")


def _get_aligned_file_paths(output_dir: str, taxa_types: List[str]) -> Dict[str, List[str]]:
    """Get organized file paths for aligned rasters."""
    return {
        'species_pool': [os.path.join(output_dir, f'{taxa}.tif') for taxa in taxa_types],
        'red_list': [os.path.join(output_dir, f'{taxa}RL.tif') for taxa in taxa_types],
        'endemic': [os.path.join(output_dir, f'{taxa}Endemic.tif') for taxa in taxa_types],
        'kba': os.path.join(output_dir, 'KBAs.tif'),
        'ecoregion': os.path.join(output_dir, 'ecoMaps.tif')
    }


def _get_scenario_paths(p, year) -> Tuple[str, str]:
    """Generate scenario name and LULC path for given year."""
    if p.scenario_type == 'baseline':
        if hasattr(p, 'alt_base_lulc_path'):
            scenario_name = f'baseline_{year}'
            lulc_path = p.alt_base_lulc_path
        else:
            scenario_name = f'baseline_{year}'
            lulc_path = os.path.join(p.fine_processed_inputs_dir, 'lulc', p.lulc_src_label, p.lulc_simplification_label, 
                                   f'lulc_{p.lulc_src_label}_{p.lulc_simplification_label}_{year}' + '.tif')
    else:
        scenario_name = (f'{p.lulc_src_label}_{p.lulc_simplification_label}_'
                        f'{p.exogenous_label}_{p.climate_label}_{p.model_label}_'
                        f'{p.counterfactual_label}_{year}')
        lulc_path = os.path.join(p.stitched_lulc_simplified_scenarios_dir, 
                                f'lulc_{scenario_name}' + '.tif')
    
    return scenario_name, lulc_path


def _create_binary_lulc(lulc_path: str, output_dir: str, scenario_name: str, 
                       excluded_classes: List[int], nodata: float) -> str:
    """Create binary LULC suitability layer."""
    binary_path = os.path.join(output_dir, f'{scenario_name}_binaryLULC.tif')
    nodata_lulc = pygeoprocessing.get_raster_info(lulc_path)['nodata'][0]
    
    def binary_lulc_func(lulc_array):
        """Convert LULC to binary suitability (1=suitable, 0=unsuitable)."""
        result = np.where(np.isin(lulc_array, excluded_classes), 0, 1)
        invalid_mask = (lulc_array == nodata_lulc)
        result[invalid_mask] = nodata
        return result.astype(np.float32)
    
    pygeoprocessing.raster_calculator(
        [(lulc_path, 1)],
        binary_lulc_func,
        binary_path,
        gdal.GDT_Float32,
        nodata
    )
    
    return binary_path


def _calculate_species_indices(binary_path: str, file_paths: Dict, output_dir: str, 
                             scenario_name: str, taxa_types: List[str], 
                             global_richness: Dict, nodata: float) -> Dict[str, str]:
    """Calculate species-based biodiversity indices."""
    
    def score_species_func(binary_array, *taxa_arrays):
        """Score species richness normalized by global richness."""
        result = np.full(binary_array.shape, nodata, dtype=np.float32)
        valid_mask = (binary_array != nodata)
        result[valid_mask] = 0
        
        global_values = [global_richness[taxa] for taxa in taxa_types]
        
        for taxa_array, global_val in zip(taxa_arrays, global_values):
            taxa_clean = np.maximum(taxa_array, 0)  # Remove negative values
            result[valid_mask] += (binary_array[valid_mask] * taxa_clean[valid_mask] / 
                                 (100 * global_val))
        
        return np.maximum(result, 0)  # Ensure non-negative
    
    species_layers = {
        'species_richness': file_paths['species_pool'],
        'red_list': file_paths['red_list'],
        'endemic': file_paths['endemic']
    }
    
    output_paths = {}
    for layer_name, raster_files in species_layers.items():
        output_path = os.path.join(output_dir, f"{scenario_name}_{layer_name}.tif")
        output_paths[layer_name] = output_path
        
        raster_list = [(binary_path, 1)] + [(f, 1) for f in raster_files]
        
        pygeoprocessing.raster_calculator(
            raster_list,
            score_species_func,
            output_path,
            gdal.GDT_Float32,
            nodata
        )
    
    return output_paths


def _calculate_kba_index(binary_path: str, kba_file: str, output_dir: str, 
                        scenario_name: str, nodata: float) -> str:
    """Calculate Key Biodiversity Areas index."""
    kba_path = os.path.join(output_dir, f'{scenario_name}_kba.tif')
    
    def score_kba_func(binary_array, kba_array):
        """Score KBA coverage."""
        result = np.full(binary_array.shape, nodata, dtype=np.float32)
        valid_mask = (binary_array != nodata)
        result[valid_mask] = 0
        
        kba_clean = np.maximum(kba_array, 0)
        result[valid_mask] = binary_array[valid_mask] * kba_clean[valid_mask]
        
        return np.maximum(result, 0)
    
    pygeoprocessing.raster_calculator(
        [(binary_path, 1), (kba_file, 1)],
        score_kba_func,
        kba_path,
        gdal.GDT_Float32,
        nodata
    )
    
    return kba_path


def _calculate_ecoregion_index(binary_path: str, ecoregion_file: str, output_dir: str, 
                              scenario_name: str, nodata: float) -> str:
    """Calculate ecoregion diversity index."""
    ecoregion_path = os.path.join(output_dir, f'{scenario_name}_ecoregion.tif')
    nodata_eco = pygeoprocessing.get_raster_info(ecoregion_file)['nodata'][0]
    
    def score_ecoregion_func(binary_array, eco_array):
        """Score ecoregion diversity (inverse relationship)."""
        result = np.full(binary_array.shape, nodata, dtype=np.float32)
        valid_mask = np.logical_and(binary_array != nodata, eco_array != nodata_eco)
        result[valid_mask] = 0
        
        eco_clean = np.maximum(eco_array, 1)  # Avoid division by zero
        result[valid_mask] = binary_array[valid_mask] / eco_clean[valid_mask]
        
        return result
    
    pygeoprocessing.raster_calculator(
        [(binary_path, 1), (ecoregion_file, 1)],
        score_ecoregion_func,
        ecoregion_path,
        gdal.GDT_Float32,
        nodata
    )
    
    return ecoregion_path


def _get_raster_min_max(raster_path: str) -> Tuple[float, float]:
    """Get minimum and maximum values from raster for normalization."""
    try:
        with gdal.OpenEx(raster_path) as ds:
            band = ds.GetRasterBand(1)
            band.ComputeStatistics(0)
            return band.GetMinimum(), band.GetMaximum()
    except Exception as e:
        logging.error(f"Error computing statistics for {raster_path}: {e}")
        return 0.0, 1.0


def _calculate_final_index(binary_path: str, species_paths: Dict[str, str], 
                          kba_path: str, ecoregion_path: str, output_dir: str, 
                          scenario_name: str, weights: Dict[str, float], nodata: float):
    """Calculate final weighted biodiversity index."""
    
    # Get normalization values
    norm_values = {}
    for layer_name in ['species_richness', 'red_list', 'endemic', 'ecoregion']:
        path = species_paths.get(layer_name, ecoregion_path if layer_name == 'ecoregion' else None)
        if path:
            norm_values[layer_name] = _get_raster_min_max(path)
    
    def weight_layers_func(binary, species_rich, red_list, kba, endemic, ecoregion):
        """Combine all biodiversity layers with weighted average."""
        result = np.full(binary.shape, nodata, dtype=np.float32)
        valid_mask = (binary != nodata)
        
        # Clean negative values
        layers = [species_rich, red_list, kba, endemic, ecoregion]
        for layer in layers:
            layer[layer < 0] = 0
        
        # Normalize layers to [0,1]
        def normalize(arr, layer_name):
            if layer_name in norm_values:
                min_val, max_val = norm_values[layer_name]
                if max_val > min_val:
                    return np.clip((arr - min_val) / (max_val - min_val), 0, 1)
            return np.clip(arr, 0, 1)
        
        species_norm = normalize(species_rich, 'species_richness')
        red_list_norm = normalize(red_list, 'red_list')
        endemic_norm = normalize(endemic, 'endemic')
        ecoregion_norm = normalize(ecoregion, 'ecoregion')
        kba_norm = np.clip(kba, 0, 1)  # Assume KBA already normalized
        
        # Calculate weighted average
        total_weight = sum(weights.values())
        result[valid_mask] = (
            weights['species_richness'] * species_norm[valid_mask] +
            weights['red_list'] * red_list_norm[valid_mask] +
            weights['kba'] * kba_norm[valid_mask] +
            weights['endemic'] * endemic_norm[valid_mask] +
            weights['ecoregion'] * ecoregion_norm[valid_mask]
        ) / total_weight
        
        return result
    
    final_path = os.path.join(output_dir, f'{scenario_name}_index.tif')
    
    pygeoprocessing.raster_calculator(
        [
            (binary_path, 1),
            (species_paths['species_richness'], 1),
            (species_paths['red_list'], 1),
            (kba_path, 1),
            (species_paths['endemic'], 1),
            (ecoregion_path, 1)
        ],
        weight_layers_func,
        final_path,
        gdal.GDT_Float32,
        nodata
    )







# ========================= PLOTTING =========================
# ========================= CONFIGURATION =========================
class ServiceConfig:
    """Configuration for different ecosystem services"""
    
    # County to AEZ mapping
    COUNTY_AEZ_MAPPING = {
        'Baringo': 'AN', 'Bomet': 'HR', 'Bungoma': 'HR', 'Busia': 'HR',
        'Elgeyo-Marakwet': 'HR', 'Embu': 'MN', 'Garissa': 'AS', 'Homa Bay': 'HR',
        'Isiolo': 'AN', 'Kajiado': 'MS', 'Kakamega': 'HR', 'Kericho': 'HR',
        'Kiambu': 'HR', 'Kilifi': 'CO', 'Kirinyaga': 'HR', 'Kisii': 'HR',
        'Kisumu': 'HR', 'Kitui': 'MS', 'Kwale': 'CO', 'Laikipia': 'MN',
        'Lamu': 'CO', 'Machakos': 'HR', 'Makueni': 'MS', 'Mandera': 'AN',
        'Marsabit': 'AN', 'Meru': 'HR', 'Migori': 'HR', 'Mombasa': 'MO',
        "Murang'a": 'HR', 'Nairobi': 'NA', 'Nakuru': 'HR', 'Nandi': 'HR',
        'Narok': 'MS', 'Nyamira': 'HR', 'Nyandarua': 'HR', 'Nyeri': 'HR',
        'Samburu': 'AN', 'Siaya': 'HR', 'Taita Taveta': 'MS', 'Tana River': 'AS',
        'Tharaka-Nithi': 'MN', 'Trans Nzoia': 'HR', 'Turkana': 'AN',
        'Uasin Gishu': 'HR', 'Vihiga': 'HR', 'Wajir': 'AN', 'West Pokot': 'MN'
    }
    
    AEZ_ORDER = ['AN', 'AS', 'CO', 'HR', 'MN', 'MO', 'MS', 'NA']

    # Consistent AEZ color scheme
    AEZ_COLORS = {
        'AN': '#d7191c', 
        'AS': '#2c7bb6',  
        'CO': '#31a354',  
        'HR': '#756bb1',  
        'MN': '#fe9929',  
        'MO': "#f781bf",  
        'MS': '#b15928', 
        'NA': "#e1c10b"   
    }
    
    CARBON_PRICE = 130  # USD per ton
    
    SERVICES = {
        'biodiversity': {
            'raster_path_template': lambda p, scenario: os.path.join(p.calculate_biodiversity_index_dir, scenario, f'{scenario}_index.tif'),
            'colormap': 'Greens',
            'colormap_change': 'RdBu',
            'colorbar_label': 'Mean Biodiversity Index',
            'colorbar_label_change': '% Change in Mean Biodiversity',
            'resample_method': 'near',
            'map_ranges': {'raster': (0, 0.8), 'vector': (0, 0.6)},
            'change_ranges': {'raster': (-20, 20)},
            'y_ranges': {'aez': (-15, 15), 'county': (-15, 15)},
            'create_vector_map': False,
            'use_percent_change': True
        },
        'carbon': {
            'raster_path_template': lambda p, scenario: os.path.join(p.run_invest_carbon_dir, scenario, 'tot_c_cur.tif'),
            'colormap': 'Purples',
            'colormap_change': 'RdBu',
            'colorbar_label': 'Mean Carbon Storage (t C/ha)',
            'colorbar_label_change': '% Change in Mean Carbon (t C/ha)',
            'resample_method': 'bilinear',
            'map_ranges': {'raster': (0, 4000), 'vector': (0, 1600)},
            'change_ranges': {'raster': (-20, 20)},
            'y_ranges': {'aez': (-250, 250), 'county': (-250, 250)},
            'create_vector_map': False,
            'use_percent_change': True
        },
        'carbon_total': {
            'raster_path_template': lambda p, scenario: os.path.join(p.run_invest_carbon_dir, scenario, 'tot_c_cur.tif'),
            'colormap': 'Purples',
            'colormap_change': 'RdBu',
            'colorbar_label': 'Total Carbon Storage (t C)',
            'colorbar_label_change': 'Change in Total Carbon (t C)',
            'resample_method': 'bilinear',
            'map_ranges': {'raster': (0, 4000), 'vector': (0, 1600)},
            'change_ranges': {'raster': (-500, 500)},
            'y_ranges': {'aez': (-5000000, 5000000), 'county': (-2000000, 2000000)},
            'create_vector_map': False,
            'use_total': True,
            'use_percent_change': False
        },
        'carbon_value': {
            'raster_path_template': lambda p, scenario: os.path.join(p.run_invest_carbon_dir, scenario, 'tot_c_cur.tif'),
            'colormap': 'Purples',
            'colormap_change': 'RdBu',
            'colorbar_label': 'Total Carbon Value ($)',
            'colorbar_label_change': 'Change in Carbon Value ($)',
            'resample_method': 'bilinear',
            'map_ranges': {'raster': (0, 4000), 'vector': (0, 1600)},
            'change_ranges': {'raster': (-500, 500)},
            'y_ranges': {'aez': (-650000000, 650000000), 'county': (-260000000, 260000000)},
            'create_vector_map': False,
            'use_total': True,
            'use_value': True,
            'use_percent_change': False
        },
        'pollination': {
            'raster_path_template': lambda p, scenario: os.path.join(p.run_invest_pollination_dir, scenario, 'total_pollinator_abundance_spring.tif'),
            'colormap': 'Oranges',
            'colormap_change': 'RdBu',
            'colorbar_label': 'Mean Pollinator Abundance per Pixel',
            'colorbar_label_change': '% Change in Mean Pollinator Abundance',
            'resample_method': 'bilinear',
            'map_ranges': {'raster': (0, 0.15), 'vector': (0, 0.1)},
            'change_ranges': {'raster': (-30, 30)},
            'y_ranges': {'aez': (-30, 30), 'county': (-30, 30)},
            'create_vector_map': False,
            'use_percent_change': True
        },
        'lulc': {
            'raster_path_template': lambda p, scenario: get_lulc_clipped_path(p, scenario),
            'colormap': 'custom_lulc',
            'colorbar_label': 'LULC Class',
            'resample_method': 'near',
            'y_ranges': {'aez': (-2000000, 2000000), 'county': (-800000, 800000)},
            'classes': {1: 'Urban', 2: 'Cropland', 3: 'Grassland', 4: 'Forest', 5: 'Other Natural', 6: 'Water', 7: 'Other'},
            'colors': ['#FF0000', '#FFA500', '#BDB76B', '#006400', '#66CDAA', '#4682B4', '#D3D3D3'],
            'create_vector_map': False,
            'skip_change_for_bau': True  # LULC always shows absolute, not change
        }
    }

    # Font configuration
    FONT_SETTINGS = {
        'title_size': 24,
        'label_size': 18,
        'legend_size': 18,
        'tick_size': 18,
        'axis_label_size': 18,
        'colorbar_label_size': 18
    }

def get_lulc_path(p, scenario):
    """Get LULC raster path based on scenario type"""
    if 'baseline' in scenario:
        if hasattr(p, 'alt_base_lulc_path') and os.path.exists(p.alt_base_lulc_path):
            return p.alt_base_lulc_path
        elif hasattr(p, 'base_years') and p.base_years:
            baseline_year = p.base_years[0]
            return os.path.join(
                p.fine_processed_inputs_dir, 'lulc', p.lulc_src_label, 
                p.lulc_simplification_label, 
                f'lulc_{p.lulc_src_label}_{p.lulc_simplification_label}_{baseline_year}.tif'
            )
    else:
        return os.path.join(p.stitched_lulc_simplified_scenarios_dir, f'lulc_{scenario}.tif')

def get_lulc_clipped_path(p, scenario):
    """Get clipped LULC raster path"""
    if 'baseline' in scenario:
        return os.path.join(p.cur_dir, 'lulc_clipped', 'baseline_lulc_clipped.tif')
    else:
        return os.path.join(p.cur_dir, 'lulc_clipped', f'{scenario}_clipped.tif')

# ========================= DATA PROCESSING =========================
class RasterProcessor:
    """Handles raster preprocessing and clipping operations"""
    
    def __init__(self, project):
        self.project = project
        
    def clip_all_rasters(self):
        """Pre-clip all rasters to AOI and save in service subfolders"""
        if not hasattr(self.project, 'aoi_path') or not os.path.exists(self.project.aoi_path):
            print("Warning: No AOI path found, skipping clipping")
            return {}
        
        clipped_dirs = self._create_service_directories()
        raster_paths = self._collect_raster_paths(clipped_dirs)
        
        if not raster_paths['base_rasters']:
            print("No rasters found to process!")
            return clipped_dirs
        
        if not all(os.path.exists(path) for path in raster_paths['target_rasters']):
            self._execute_clipping(raster_paths)
        
        return clipped_dirs
    
    def _create_service_directories(self):
        """Create directories for clipped rasters"""
        clipped_dirs = {}
        for service in ServiceConfig.SERVICES.keys():
            # Skip carbon_total and carbon_value - they use same rasters as carbon
            if service in ['carbon_total', 'carbon_value']:
                continue
            clipped_dir = os.path.join(self.project.cur_dir, f'{service}_clipped')
            os.makedirs(clipped_dir, exist_ok=True)
            clipped_dirs[service] = clipped_dir
        # Map carbon_total and carbon_value to carbon directory
        clipped_dirs['carbon_total'] = clipped_dirs['carbon']
        clipped_dirs['carbon_value'] = clipped_dirs['carbon']
        return clipped_dirs
    
    def _collect_raster_paths(self, clipped_dirs):
        """Collect all raster paths for processing"""
        base_rasters, target_rasters, resample_methods = [], [], []
        baseline_added = False
        
        for index, row in self.project.scenarios_df.iterrows():
            seals_utils.assign_df_row_to_object_attributes(self.project, row)
            
            if not baseline_added:
                baseline_added = self._add_baseline_lulc(base_rasters, target_rasters, resample_methods, clipped_dirs)
            
            years, template = self._get_scenario_config()
            for year in years:
                scenario_year = template.format(year)
                self._add_service_rasters(scenario_year, base_rasters, target_rasters, resample_methods, clipped_dirs)
        
        return {
            'base_rasters': base_rasters,
            'target_rasters': target_rasters, 
            'resample_methods': resample_methods
        }
    
    def _get_scenario_config(self):
        """Get years and naming template for current scenario"""
        if self.project.scenario_type == 'baseline':
            return self.project.base_years, 'baseline_{}'
        else:
            template = f'{self.project.lulc_src_label}_{self.project.lulc_simplification_label}_{self.project.exogenous_label}_{self.project.climate_label}_{self.project.model_label}_{self.project.counterfactual_label}_{{}}'
            return self.project.years, template
    
    def _add_baseline_lulc(self, base_rasters, target_rasters, resample_methods, clipped_dirs):
        """Add baseline LULC raster to processing list"""
        clipped_base_lulc_path = os.path.join(clipped_dirs['lulc'], 'baseline_lulc_clipped.tif')
        
        if hasattr(self.project, 'alt_base_lulc_path') and os.path.exists(self.project.alt_base_lulc_path):
            base_rasters.append(self.project.alt_base_lulc_path)
            target_rasters.append(clipped_base_lulc_path)
            resample_methods.append('near')
            return True
        elif hasattr(self.project, 'base_years') and self.project.base_years:
            baseline_year = self.project.base_years[0]
            standard_lulc_path = os.path.join(
                self.project.fine_processed_inputs_dir, 'lulc', 
                self.project.lulc_src_label, self.project.lulc_simplification_label, 
                f'lulc_{self.project.lulc_src_label}_{self.project.lulc_simplification_label}_{baseline_year}.tif'
            )
            
            if os.path.exists(standard_lulc_path):
                base_rasters.append(standard_lulc_path)
                target_rasters.append(clipped_base_lulc_path)
                resample_methods.append('near')
                return True
        
        return False
    
    def _add_service_rasters(self, scenario_year, base_rasters, target_rasters, resample_methods, clipped_dirs):
        """Add service rasters for a scenario year"""
        for service_name, config in ServiceConfig.SERVICES.items():
            # Skip carbon_total and carbon_value - they use carbon rasters
            if service_name in ['carbon_total', 'carbon_value']:
                continue
                
            if service_name == 'lulc':
                if 'baseline' not in scenario_year:
                    lulc_source_path = os.path.join(self.project.stitched_lulc_simplified_scenarios_dir, f'lulc_{scenario_year}.tif')
                    if os.path.exists(lulc_source_path):
                        clipped_path = os.path.join(clipped_dirs['lulc'], f'{scenario_year}_clipped.tif')
                        base_rasters.append(lulc_source_path)
                        target_rasters.append(clipped_path)
                        resample_methods.append(config['resample_method'])
                continue
                
            raster_path = config['raster_path_template'](self.project, scenario_year)
            if os.path.exists(raster_path):
                clipped_path = os.path.join(clipped_dirs[service_name], f'{scenario_year}_{service_name}_clipped.tif')
                base_rasters.append(raster_path)
                target_rasters.append(clipped_path)
                resample_methods.append(config['resample_method'])
    
    def _execute_clipping(self, raster_paths):
        """Execute the clipping operation"""
        pygeoprocessing.align_and_resize_raster_stack(
            base_raster_path_list=raster_paths['base_rasters'],
            target_raster_path_list=raster_paths['target_rasters'],
            resample_method_list=raster_paths['resample_methods'],
            target_pixel_size=pygeoprocessing.get_raster_info(self.project.aoi_ha_per_cell_fine_path)['pixel_size'],
            bounding_box_mode='intersection',
            base_vector_path_list=[self.project.aoi_path],
            target_projection_wkt=pygeoprocessing.get_vector_info(self.project.aoi_path)['projection_wkt'],
            vector_mask_options={'mask_vector_path': self.project.aoi_path},
        )

# ========================= ZONAL STATISTICS =========================
class ZonalStatsCalculator:
    """Handles zonal statistics calculations"""
    
    def __init__(self, project):
        self.project = project
    
    def calculate_stats(self, scenario_name, service_type, vector_path):
        """Calculate zonal statistics for a service and scenario"""
        config = ServiceConfig.SERVICES[service_type]
        raster_path = config['raster_path_template'](self.project, scenario_name)
        
        if not os.path.exists(raster_path):
            return None
        
        include_counts = (service_type == 'lulc')
        
        return pygeoprocessing.zonal_statistics(
            base_raster_path_band=(raster_path, 1),
            aggregate_vector_path=vector_path,
            aggregate_layer_name=None,
            ignore_nodata=True,
            polygons_might_overlap=False,
            include_value_counts=include_counts,
            working_dir=None
        )

# ========================= DATA FORMATTING =========================
class DataFormatter:
    """Handles conversion of statistics to DataFrames"""
    
    @staticmethod
    def stats_to_dataframe(stats_dict, scenario_name, year, region_mapping, is_lulc=False):
        """Convert statistics dictionary to DataFrame"""
        rows = []
        
        for region_id, stats in stats_dict.items():
            try:
                base_row = DataFormatter._create_base_row(stats, region_id, scenario_name, year, region_mapping)
                
                if is_lulc and 'value_counts' in stats and stats['value_counts']:
                    for lulc_class, pixel_count in stats['value_counts'].items():
                        lulc_row = base_row.copy()
                        lulc_row.update({
                            'lulc_class': int(lulc_class),
                            'pixel_count': DataFormatter._safe_convert(pixel_count, int),
                            'percentage': (DataFormatter._safe_convert(pixel_count, int) / base_row['count'] * 100) if base_row['count'] > 0 else 0
                        })
                        rows.append(lulc_row)
                else:
                    rows.append(base_row)
                    
            except Exception as e:
                rows.append(DataFormatter._create_fallback_row(region_id, scenario_name, year, region_mapping))
        
        return pd.DataFrame(rows)
    
    @staticmethod
    def _create_base_row(stats, region_id, scenario_name, year, region_mapping):
        """Create base statistics row"""
        count = DataFormatter._safe_convert(stats['count'], int)
        sum_val = DataFormatter._safe_convert(stats['sum'], float)
        
        return {
            'region_id': region_id,
            'region_label': region_mapping.get(region_id, f'Region_{region_id}'),
            'scenario': scenario_name,
            'year': year,
            'sum_value': sum_val,
            'mean_value': sum_val / count if count > 0 else 0.0,
            'min_value': DataFormatter._safe_convert(stats['min'], float),
            'max_value': DataFormatter._safe_convert(stats['max'], float),
            'count': count
        }
    
    @staticmethod
    def _safe_convert(value, target_type):
        """Safely convert NumPy types to Python types"""
        if hasattr(value, 'item'):
            return target_type(value.item())
        return target_type(value)
    
    @staticmethod
    def _create_fallback_row(region_id, scenario_name, year, region_mapping):
        """Create fallback row for failed conversions"""
        return {
            'region_id': region_id,
            'region_label': region_mapping.get(region_id, f'Region_{region_id}'),
            'scenario': scenario_name,
            'year': year,
            'sum_value': 0.0,
            'mean_value': 0.0,
            'min_value': 0.0,
            'max_value': 0.0,
            'count': 0
        }

# ========================= VISUALIZATION =========================
class MapVisualizer:
    """Handles map creation and visualization"""
    
    def __init__(self, clipped_dirs, font_settings=None):
        self.clipped_dirs = clipped_dirs
        self.project = None 

        # Set default font settings
        default_fonts = {
            'title_size': 16,
            'label_size': 16,
            'legend_size': 16,
            'tick_size': 14,
            'axis_label_size': 10,
            'colorbar_label_size': 12
        }
        
        # Update with provided settings
        self.font_settings = default_fonts.copy()
        if font_settings:
            self.font_settings.update(font_settings)
        
    def create_combined_visualization(self, scenario_data, baseline_data, output_dir, service_name, 
                                     scenario_name, project, aez_mapping, county_mapping, is_bau=False):
        """Create combined 3-panel visualization with raster maps and two bar charts"""
        
        self.project = project
        # Use unique years only
        scenario_years = sorted(list(set([year for _, _, year in scenario_data])))
        baseline_year = baseline_data[0][2] if baseline_data else None
        
        if not scenario_years or baseline_year is None:
            return
        
        base_service = self._get_base_service_name(service_name)
        config = ServiceConfig.SERVICES.get(base_service, {})
        
        # Create full 3-panel figure
        combined_filename = f'{scenario_name}_{service_name}_combined.png'
        combined_path = os.path.join(output_dir, combined_filename)
        
        if os.path.exists(combined_path):
            return
        
        # Setup figure with 3 vertical panels
        fig = plt.figure(figsize=(24, 18))
        
        # Panel 1: Raster maps (top)
        self._create_raster_maps_panel(fig, scenario_data, baseline_data, service_name, scenario_name, is_bau)
        
        # Panel 2: AEZ bar chart (middle)
        ax_aez = fig.add_subplot(3, 1, 2)
        self._create_bar_chart_panel(ax_aez, scenario_data, baseline_data, service_name, 
                                     project, 'aez', aez_mapping, is_bau)
        
        # Panel 3: County bar chart (bottom)
        ax_county = fig.add_subplot(3, 1, 3)
        self._create_bar_chart_panel(ax_county, scenario_data, baseline_data, service_name,
                                     project, 'county', county_mapping, is_bau)
        
        plt.tight_layout()
        fig.savefig(combined_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _get_base_service_name(self, service_name):
        """Extract base service name from composite names"""
        if 'carbon' in service_name:
            if 'total' in service_name:
                return 'carbon_total'
            elif 'value' in service_name:
                return 'carbon_value'
            else:
                return 'carbon'
        return service_name.split('_')[0]
    
    def _create_raster_maps_panel(self, fig, scenario_data, baseline_data, service_name, scenario_name, is_bau):
        """Create raster maps in the top panel"""
        # Use unique years only (avoid duplicates from aez/county processing)
        scenario_years = sorted(list(set([year for _, _, year in scenario_data])))
        baseline_year = baseline_data[0][2] if baseline_data else None
        
        # Get service configuration
        is_lulc = 'lulc' in service_name.lower()
        base_service = self._get_base_service_name(service_name)
        config = ServiceConfig.SERVICES.get(base_service, ServiceConfig.SERVICES['biodiversity'])
        
        # Determine if we show change maps
        show_change = not is_bau and not config.get('skip_change_for_bau', False)
        
        # Determine number of maps and layout
        if show_change:
            n_maps = len(scenario_years)  # No baseline for change maps
        else:
            n_maps = len(scenario_years) + 1  # Include baseline
        n_cols = min(4, n_maps)
        
        # Setup colormap and ranges
        if is_lulc:
            cmap = ListedColormap(config['colors'])
            bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
            norm = BoundaryNorm(bounds, cmap.N)
            vmin, vmax = None, None
        else:
            if show_change:
                cmap = config.get('colormap_change', 'RdBu')
                norm = TwoSlopeNorm(vmin=config['change_ranges']['raster'][0], 
                                   vcenter=0, 
                                   vmax=config['change_ranges']['raster'][1])
                vmin, vmax = None, None
            else:
                cmap = config['colormap']
                norm = None
                if 'map_ranges' in config:
                    vmin, vmax = config['map_ranges']['raster']
                else:
                    vmin, vmax = self._calculate_value_range(scenario_data, baseline_data, service_name, is_lulc)
        
        # Create grid spec for maps in top panel
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(3, 1, figure=fig, height_ratios=[1.5, 1, 1])
        gs_maps = gs[0].subgridspec(1, n_cols)
        
        map_idx = 0
        
        # Load baseline data for change calculation
        baseline_data_array = None
        shared_extent = None
        if show_change or not show_change:  # Load baseline in both cases
            baseline_data_array, shared_extent = self._load_raster_for_map(baseline_year, service_name, is_baseline=True)
        
        # Plot baseline if not showing change
        if not show_change and baseline_data_array is not None:
            ax = fig.add_subplot(gs_maps[0, map_idx])
            im = self._plot_single_map(ax, baseline_data_array, shared_extent, f'{baseline_year}', 
                                     cmap, norm, vmin, vmax, is_lulc)
            map_idx += 1
        
        # Plot scenario years (as change or absolute)
        for year in scenario_years:
            data_array, extent = self._load_raster_for_map(year, service_name, scenario_name, is_baseline=False)
            if data_array is not None:
                if show_change and baseline_data_array is not None:
                    # Calculate change (percent for biodiversity/pollination, absolute for others)
                    if config.get('use_percent_change', False):
                        combined_mask = baseline_data_array.mask | data_array.mask
                        with np.errstate(divide='ignore', invalid='ignore'):
                            change_array = np.where(baseline_data_array != 0, 
                                                   ((data_array - baseline_data_array) / baseline_data_array) * 100,
                                                   0)
                            # Apply the combined mask plus any invalid values
                            change_array = np.ma.masked_where(
                                combined_mask | np.isnan(change_array) | np.isinf(change_array), 
                                change_array
                            )
                    else:
                        # Absolute change
                        change_array = data_array - baseline_data_array
                    
                    ax = fig.add_subplot(gs_maps[0, map_idx])
                    im = self._plot_single_map(ax, change_array, shared_extent, f'{year}', 
                                             cmap, norm, vmin, vmax, False)
                else:
                    ax = fig.add_subplot(gs_maps[0, map_idx])
                    im = self._plot_single_map(ax, data_array, shared_extent, f'{year}', 
                                             cmap, norm, vmin, vmax, is_lulc)
                map_idx += 1

        
        # Add colorbar
        if 'im' in locals():
            if is_lulc:
                # Place colorbar to the right of all maps
                cbar_ax = fig.add_axes([0.88, 0.75, 0.015, 0.15])
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.set_label(config['colorbar_label'],
                               fontsize=self.font_settings['colorbar_label_size'])
                lulc_classes = list(range(1, 8))
                cbar.set_ticks(lulc_classes)
                cbar.set_ticklabels([config['classes'][i] for i in lulc_classes],
                    fontsize=self.font_settings['tick_size'])
                cbar.ax.invert_yaxis()
                # cbar.set_ticklabels([config['classes'][i] for i in range(7, 0, -1)],
                #                     fontsize=self.font_settings['tick_size'])
            else:
                # Place colorbar to the right of all maps
                cbar_ax = fig.add_axes([0.93, 0.75, 0.015, 0.15])
                cbar = fig.colorbar(im, cax=cbar_ax)
                label = config.get('colorbar_label_change', config['colorbar_label']) if show_change else config['colorbar_label']
                cbar.set_label(label, fontsize=self.font_settings['colorbar_label_size'])
                cbar.ax.tick_params(labelsize=self.font_settings['tick_size'])

    def _calculate_value_range(self, scenario_data, baseline_data, service_name, is_lulc):
        """Calculate shared value range for consistent visualization"""
        if is_lulc:
            return None, None
        
        all_arrays = []
        
        # Load baseline
        baseline_year = baseline_data[0][2] if baseline_data else None
        if baseline_year:
            baseline_array, _ = self._load_raster_for_map(baseline_year, service_name, is_baseline=True)
            if baseline_array is not None:
                all_arrays.append(baseline_array)
        
        # Load scenario years (use unique years only)
        scenario_years = sorted(list(set([year for _, _, year in scenario_data])))
        scenario_name = scenario_data[0][1] if scenario_data else None
        
        for year in scenario_years:
            data_array, _ = self._load_raster_for_map(year, service_name, scenario_name, is_baseline=False)
            if data_array is not None:
                all_arrays.append(data_array)
        
        if all_arrays:
            all_values = [arr.compressed() for arr in all_arrays]
            valid_values = [vals for vals in all_values if len(vals) > 0]
            if valid_values:
                vmin = min(np.min(vals) for vals in valid_values)
                vmax = max(np.max(vals) for vals in valid_values)
                return vmin, vmax
        
        return None, None
    def _load_raster_for_map(self, year, service_name, scenario_name=None, is_baseline=False):
        """Load a raster for mapping"""
        if is_baseline:
            scenario_year = f'baseline_{year}'
        else:
            scenario_year = f'{scenario_name}_{year}' if scenario_name else f'baseline_{year}'
        
        base_service = self._get_base_service_name(service_name)
        service_dir = self.clipped_dirs.get(base_service)
        
        if not service_dir:
            return None, None
        
        # Construct filename
        if base_service == 'lulc':
            filename = 'baseline_lulc_clipped.tif' if is_baseline else f'{scenario_year}_clipped.tif'
        elif base_service in ['carbon_total', 'carbon_value']:
            # Use the carbon raster
            filename = f'{scenario_year}_carbon_clipped.tif'
        else:
            filename = f'{scenario_year}_{base_service}_clipped.tif'
        
        raster_path = os.path.join(service_dir, filename)
        
        if not os.path.exists(raster_path):
            return None, None
        
        try:
            with rasterio.open(raster_path) as src:
                data = src.read(1)
                bounds = src.bounds
                extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
                
                # Mask nodata values
                nodata_mask = self._create_nodata_mask(data, src.nodata)
                data_masked = np.ma.masked_where(nodata_mask, data)
                
                return data_masked, extent
                
        except Exception as e:
            return None, None
    
    def _create_nodata_mask(self, data, nodata):
        """Create mask for nodata values"""
        if nodata is not None:
            return data == nodata
        else:
            return (
                (data == -9999) | 
                (data == -3.4028235e+38) |
                np.isnan(data) |
                np.isinf(data)
            )
    
    def _plot_single_map(self, ax, data_array, extent, title, cmap, norm, vmin, vmax, is_lulc):
        """Plot a single map on an axis with AEZ boundaries"""
        if is_lulc:
            im = ax.imshow(data_array, cmap=cmap, norm=norm, interpolation='nearest',
                          extent=extent, aspect='equal')
        else:
            im = ax.imshow(data_array, cmap=cmap, interpolation='nearest',
                          extent=extent, aspect='equal', vmin=vmin, vmax=vmax, norm=norm)
        
        # Add AEZ boundaries (black lines)
        aez_gdf = gpd.read_file(self.project.regions_vector_path_epsg8857)
        aez_gdf.to_crs('EPSG:4326').boundary.plot(ax=ax, color='black', linewidth=0.7, alpha=1.0)
        
        ax.set_title(title, fontsize=self.font_settings['title_size'])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        return im
    
    def _create_bar_chart_panel(self, ax, scenario_data, baseline_data, service_name, 
                                project, region_type, region_mapping, is_bau=False):
        """Create bar chart panel for a region type"""
        is_lulc = 'lulc' in service_name.lower()
        
        if is_lulc:
            self._create_lulc_stacked_bars(ax, scenario_data, baseline_data, service_name,
                                          project, region_type, region_mapping, is_bau)
        else:
            self._create_grouped_bars(ax, scenario_data, baseline_data, service_name,
                                     project, region_type, region_mapping, is_bau)
    
    def _create_grouped_bars(self, ax, scenario_data, baseline_data, service_name,
                            project, region_type, region_mapping, is_bau=False):
        """Create grouped bar chart for continuous services"""
        base_service = self._get_base_service_name(service_name)
        config = ServiceConfig.SERVICES.get(base_service)

        # Determine if we show change
        show_change = not is_bau
        use_percent = config.get('use_percent_change', False) and show_change
        
        # Collect all years including baseline
        baseline_year = baseline_data[0][2] if baseline_data else None
        scenario_years = sorted(list(set([year for _, _, year in scenario_data])))
        
        if show_change:
            all_years = scenario_years  # No baseline in display
        else:
            all_years = [baseline_year] + scenario_years if baseline_year else scenario_years
        
        # Collect baseline stats for change calculation
        baseline_stats = None
        if show_change and baseline_year:
            baseline_stats = self._calculate_regional_stats_for_bars(
                baseline_year, service_name, project, region_type, 'baseline', True
            )
        
        # Collect regional data for all years
        regional_data = {}  # {region_name: {year: value}}
        
        for year in all_years:
            is_baseline = (year == baseline_year) and not show_change
            scenario_name = 'baseline' if is_baseline else scenario_data[0][1]
            
            year_stats = self._calculate_regional_stats_for_bars(
                year, service_name, project, region_type, scenario_name, is_baseline
            )
            
            for region_name, value in year_stats.items():
                if region_name not in regional_data:
                    regional_data[region_name] = {}
                
                # Calculate change if needed
                if show_change and baseline_stats and region_name in baseline_stats:
                    baseline_val = baseline_stats[region_name]
                    change_val = value - baseline_val
                    if use_percent and baseline_val != 0:
                        regional_data[region_name][year] = (change_val / baseline_val) * 100
                    else:
                        regional_data[region_name][year] = change_val
                else:
                    regional_data[region_name][year] = value
        
        if not regional_data:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Sort regions appropriately
        if region_type.lower() == 'county':
            regions = self._sort_counties_by_aez(list(regional_data.keys()))
        else:
            regions = sorted(regional_data.keys())
        
        n_regions = len(regions)
        n_years = len(all_years)
        
        x = np.arange(n_regions)
        width = 0.8 / n_years
        
        # Get colors from the service's colormap
        if config and config.get('colormap'):
            cmap = plt.get_cmap(config['colormap'])
            colors = [cmap(0.3 + 0.6 * i / (n_years - 1)) if n_years > 1 else cmap(0.6) 
                     for i in range(n_years)]
        else:
            colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_years))
        
        # Plot bars for each year
        for i, year in enumerate(all_years):
            values = [regional_data[region].get(year, 0) for region in regions]
            offset = (i - n_years/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=str(year), color=colors[i])
        
        # Configure axes
        if region_type.lower() == 'aez':
            region_label = region_type.upper()
        else:
            region_label = region_type.title()
        
        # Set ylabel based on whether showing change
        if show_change:
            if use_percent:
                ylabel = f'% Change in\n{config["colorbar_label"]}'
            else:
                ylabel = f'Change in\n{config["colorbar_label"]}'
        else:
            if service_name == 'carbon_total' or service_name == 'carbon_value':
                ylabel = f'{config["colorbar_label"]}'
            else:
                ylabel = f'{config["colorbar_label"]}'

        def carbon_million_formatter(x, _):
            if abs(x) >= 1e6:
                return f"{x/1e6:}M"
            elif abs(x) >= 1e3:
                return f"{int(x):,}"
            else:
                return f"{x:.0f}"
        def carbon_million_formatter(x, _):
            if abs(x) >= 1e6:
                return f"{x/1e6:.0f}M"
            elif abs(x) >= 1e3:
                return f"{int(x):,}"
            else:
                return f"{x:.0f}"
        
        ax.set_ylabel(ylabel, fontsize=self.font_settings['label_size'])
        ax.set_title(f'{region_label} Regions', 
                     fontsize=self.font_settings['title_size'], pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(regions, rotation=45, ha='right', fontsize=self.font_settings['tick_size'])
        for label, region in zip(ax.get_xticklabels(), regions):
            if region_type.lower() == 'county':
                aez = ServiceConfig.COUNTY_AEZ_MAPPING.get(region)
            else:
                aez = region  # already an AEZ name
            color = ServiceConfig.AEZ_COLORS.get(aez, 'black')
            label.set_color(color)
        ax.tick_params(axis='y', labelsize=self.font_settings['tick_size'])
        # ax.yaxis.set_major_formatter(mticker.FuncFormatter(yaxis_tick_formatter))
        if base_service in ['biodiversity', 'pollination'] and is_bau:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
        elif base_service in ['carbon_total', 'carbon_value']:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(carbon_million_formatter))
        elif use_percent:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}%"))
        else:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.legend(title='Year', loc='center left', bbox_to_anchor=(1, 0.5),
                  fontsize=self.font_settings['legend_size'], 
                  title_fontsize=self.font_settings['legend_size'])
        ax.grid(True, alpha=0.3, axis='y')

    
    def _sort_counties_by_aez(self, counties):
        """Sort counties by AEZ region"""
        def get_aez_order(county):
            aez = ServiceConfig.COUNTY_AEZ_MAPPING.get(county, 'ZZ')
            aez_idx = ServiceConfig.AEZ_ORDER.index(aez) if aez in ServiceConfig.AEZ_ORDER else 999
            return (aez_idx, county)
        
        return sorted(counties, key=get_aez_order)

    
    def _create_lulc_stacked_bars(self, ax, scenario_data, baseline_data, service_name,
                                 project, region_type, region_mapping, is_bau=False):
        """Create stacked bar chart for LULC classes"""
        # For LULC, show change in hectares for non-BAU scenarios
        show_change = not is_bau
        
        # Collect all years including baseline
        baseline_year = baseline_data[0][2] if baseline_data else None
        scenario_years = sorted(list(set([year for _, _, year in scenario_data])))
        
        if show_change:
            all_years = scenario_years  # No baseline in display
        else:
            all_years = [baseline_year] + scenario_years if baseline_year else scenario_years
        
        # Collect baseline stats for change calculation
        baseline_stats = None
        if show_change and baseline_year:
            baseline_stats = self._calculate_lulc_stats_for_bars(
                baseline_year, service_name, project, region_type, 'baseline', True
            )
        
        # Collect regional LULC data for all years
        regional_lulc_data = {}  # {region_name: {year: {class: hectares}}}
        
        for year in all_years:
            is_baseline = (year == baseline_year) and not show_change
            scenario_name = 'baseline' if is_baseline else scenario_data[0][1]
            
            year_stats = self._calculate_lulc_stats_for_bars(
                year, service_name, project, region_type, scenario_name, is_baseline
            )
            
            for region_name, class_dict in year_stats.items():
                if region_name not in regional_lulc_data:
                    regional_lulc_data[region_name] = {}
                
                # Calculate change if needed
                if show_change and baseline_stats and region_name in baseline_stats:
                    change_dict = {}
                    baseline_dict = baseline_stats[region_name]
                    for lulc_class in range(1, 8):
                        baseline_ha = baseline_dict.get(lulc_class, 0)
                        scenario_ha = class_dict.get(lulc_class, 0)
                        change_dict[lulc_class] = scenario_ha - baseline_ha
                        # if baseline_ha != 0:
                        #     change_dict[lulc_class] = ((scenario_ha - baseline_ha) / baseline_ha) * 100
                        # else:
                        #     change_dict[lulc_class] = 0
                    regional_lulc_data[region_name][year] = change_dict
                else:
                    regional_lulc_data[region_name][year] = class_dict
        
        if not regional_lulc_data:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            return
    
        # Get LULC configuration
        config = ServiceConfig.SERVICES['lulc']
        lulc_classes = list(range(1, 8))
        lulc_colors = config['colors']
        
        # Sort regions appropriately
        if region_type.lower() == 'county':
            regions = self._sort_counties_by_aez(list(regional_lulc_data.keys()))
        else:
            regions = sorted(regional_lulc_data.keys())
        
        n_regions = len(regions)
        n_years = len(all_years)
        
        x = np.arange(n_regions)
        bar_width = 0.8 / n_years
        
        for year_idx, year in enumerate(all_years):
            bottoms_pos = np.zeros(n_regions)
            bottoms_neg = np.zeros(n_regions)
            offset = (year_idx - n_years/2 + 0.5) * bar_width
            for class_idx, lulc_class in enumerate(lulc_classes):
                values = []
                for region in regions:
                    ha_value = regional_lulc_data[region].get(year, {}).get(lulc_class, 0)
                    values.append(ha_value)
                pos_values = [v if v > 0 else 0 for v in values]
                neg_values = [v if v < 0 else 0 for v in values]
                # Stack positive bars
                ax.bar(x + offset, pos_values, bar_width, bottom=bottoms_pos,
                       color=lulc_colors[class_idx], alpha=0.8,
                       label=f'{config["classes"][lulc_class]}' if year_idx == 0 else "")
                # Stack negative bars
                ax.bar(x + offset, neg_values, bar_width, bottom=bottoms_neg,
                       color=lulc_colors[class_idx], alpha=0.8)
                # Update bottoms for next class
                bottoms_pos += np.array(pos_values)
                bottoms_neg += np.array(neg_values)
        
        # Configure axes
        if region_type.lower() == 'aez':
            region_label = region_type.upper()
        else:
            region_label = region_type.title()
        
        ylabel = 'Change in Area (hectares)' if show_change else 'Area (hectares)'
        ax.set_ylabel(ylabel, fontsize=self.font_settings['label_size'])
        ax.set_title(f'{region_label} LULC Composition', 
                     fontsize=self.font_settings['title_size'], pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(regions, rotation=45, ha='right', fontsize=self.font_settings['tick_size'])
        for label, region in zip(ax.get_xticklabels(), regions):
            if region_type.lower() == 'county':
                aez = ServiceConfig.COUNTY_AEZ_MAPPING.get(region)
            else:
                aez = region  # already an AEZ name
            color = ServiceConfig.AEZ_COLORS.get(aez, 'black')
            label.set_color(color)
        ax.tick_params(axis='y', labelsize=self.font_settings['tick_size'])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.legend(title='LULC Class', loc='center left', bbox_to_anchor=(1, 0.5),
                  fontsize=self.font_settings['legend_size'],
                  title_fontsize=self.font_settings['legend_size'])
        ax.grid(True, alpha=0.3, axis='y')

    def _calculate_regional_stats_for_bars(self, year, service_name, project, region_type, 
                                          scenario_name, is_baseline):
        """Calculate regional mean values for bar charts"""
        base_service = self._get_base_service_name(service_name)
        config = ServiceConfig.SERVICES.get(base_service)
        
        try:
            # Get raster data
            raster_data, _ = self._load_raster_for_map(year, service_name, scenario_name, is_baseline)
            regions_data = self._load_regions_raster(project, region_type)
            
            if raster_data is None or regions_data is None:
                return {}
            
            # For carbon_value, multiply by price per ton
            if base_service == 'carbon_value':
                raster_data = raster_data * ServiceConfig.CARBON_PRICE
            
            # Get region mapping
            vector_path = (project.alt_regions_vector_path_epsg8857 if region_type.lower() == 'county'
                          else project.regions_vector_path_epsg8857)
            id_column = (project.alt_region_id_column if region_type.lower() == 'county'
                        else project.region_id_column)
            label_column = (project.alt_region_label_column if region_type.lower() == 'county'
                           else project.region_label_column)
            
            regions_gdf = gpd.read_file(vector_path)
            region_mapping = {}
            for _, row in regions_gdf.iterrows():
                region_id = int(row[id_column])
                region_label = row[label_column]
                region_mapping[region_id] = region_label
            
            # Calculate statistic for each region
            regional_stats = {}
            use_total = config and config.get('use_total', False)
            
            for region_id, region_name in region_mapping.items():
                if region_id in [0, 255]:
                    continue
                
                region_mask = regions_data == int(region_id)
                if not np.any(region_mask):
                    continue
                
                region_values = raster_data[region_mask]
                if region_values.size > 0:
                    if use_total:
                        # For carbon_total and carbon_value, sum instead of mean
                        regional_stats[region_name] = np.sum(region_values)
                    else:
                        regional_stats[region_name] = np.mean(region_values)
            
            return regional_stats
            
        except Exception as e:
            return {}
    
    def _calculate_lulc_stats_for_bars(self, year, service_name, project, region_type,
                                      scenario_name, is_baseline):
        """Calculate LULC class counts for bar charts (returns hectares)"""
        try:
            # Get raster data
            raster_data, _ = self._load_raster_for_map(year, service_name, scenario_name, is_baseline)
            regions_data = self._load_regions_raster(project, region_type)
            
            if raster_data is None or regions_data is None:
                return {}
            
            # Get region mapping
            vector_path = (project.alt_regions_vector_path_epsg8857 if region_type.lower() == 'county'
                          else project.regions_vector_path_epsg8857)
            id_column = (project.alt_region_id_column if region_type.lower() == 'county'
                        else project.region_id_column)
            label_column = (project.alt_region_label_column if region_type.lower() == 'county'
                           else project.region_label_column)
            
            regions_gdf = gpd.read_file(vector_path)
            region_mapping = {}
            for _, row in regions_gdf.iterrows():
                region_id = int(row[id_column])
                region_label = row[label_column]
                region_mapping[region_id] = region_label
            
            # Calculate pixel size for hectare conversion (assuming 300m pixels)
            pixel_area_ha = 9  # 300m x 300m = 90,000 m² = 9 ha
            
            # Calculate class counts for each region
            regional_lulc = {}
            for region_id, region_name in region_mapping.items():
                if region_id in [0, 255]:
                    continue
                
                region_mask = regions_data == int(region_id)
                if not np.any(region_mask):
                    continue
                
                region_lulc = raster_data[region_mask]
                # Handle masked arrays properly
                if isinstance(region_lulc, np.ma.MaskedArray):
                    valid_values = region_lulc.compressed()
                    if len(valid_values) > 0:
                        class_counts = Counter(valid_values.flatten())
                    else:
                        class_counts = Counter()
                else:
                    valid_mask = ~np.isnan(region_lulc) & (region_lulc != -9999)
                    valid_values = region_lulc[valid_mask]
                    class_counts = Counter(valid_values.flatten())
                
                # Convert to dict with all classes (1-7) in hectares
                class_dict = {}
                for lulc_class in range(1, 8):
                    pixel_count = class_counts.get(lulc_class, 0)
                    class_dict[lulc_class] = pixel_count * pixel_area_ha
                
                regional_lulc[region_name] = class_dict
            
            return regional_lulc
            
        except Exception as e:
            print(f"Exception in _calculate_lulc_stats_for_bars: {e}")
            traceback.print_exc()
            return {}
    
    def _load_regions_raster(self, project, region_type):
        """Load regions raster data"""
        regions_path = (project.alt_regions_clipped_path if region_type.lower() == 'county' 
                       else project.regions_clipped_path)
        
        if not regions_path or not os.path.exists(regions_path):
            return None
        
        try:
            with rasterio.open(regions_path) as src:
                return src.read(1)
        except Exception:
            return None

# ========================= SUMMARY VISUALIZATION =========================
class SummaryPlot:
    """Creates summary comparison heatmaps across scenarios"""
    
    def __init__(self, project, font_settings=None):
        self.project = project
        self.font_settings = font_settings or ServiceConfig.FONT_SETTINGS

    def create_grouped_summary(self, base_dir):
        """Create vertically stacked grouped bar charts (one per service) for the final year across scenarios."""
        services_to_plot = ['biodiversity', 'carbon', 'pollination']
        final_year = 2050  # or make this dynamic if you prefer

        # Read data from existing CSVs
        summary_data = self._read_data_from_working_csvs(services_to_plot, [final_year])

        if not summary_data or final_year not in summary_data:
            print("No data found for grouped summary")
            return

        # Collect all scenarios
        all_scenarios = set()
        for service_data in summary_data[final_year].values():
            all_scenarios.update(service_data.keys())
        scenarios = sorted([s for s in all_scenarios if 'bau' not in s.lower()])

        if not scenarios:
            print("No non-BAU scenarios found")
            return

        # Define colors for scenarios using viridis colormap
        n_scenarios = len(scenarios)
        colors = plt.cm.Set2(np.linspace(0.2, 0.9, n_scenarios))
        scenario_colors = {scenario: colors[i] for i, scenario in enumerate(scenarios)}

        # Create 3 vertical panels
        fig, axes = plt.subplots(len(services_to_plot), 1, figsize=(14, 12), sharex=False)

        for ax, service in zip(axes, services_to_plot):
            if service not in summary_data[final_year]:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=self.font_settings['label_size'])
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            service_data = summary_data[final_year][service]

            # Build DataFrame for plotting
            data = []
            for scenario, region_vals in service_data.items():
                if isinstance(region_vals, dict):
                    for region, val in region_vals.items():
                        data.append((region, scenario, val))
                else:
                    data.append(('Overall', scenario, region_vals))

            df_plot = pd.DataFrame(data, columns=['region_label', 'scenario', 'mean_change'])
            regions = sorted(df_plot['region_label'].unique())

            # Compute x positions
            x = np.arange(len(regions))
            width = 0.12 if n_scenarios > 4 else 0.15
            offsets = np.linspace(-width * (n_scenarios - 1) / 2, width * (n_scenarios - 1) / 2, n_scenarios)

            # Plot grouped bars
            for i, scenario in enumerate(scenarios):
                scenario_df = df_plot[df_plot['scenario'] == scenario]
                yvals = [scenario_df[scenario_df['region_label'] == r]['mean_change'].mean()
                         if r in scenario_df['region_label'].values else 0 for r in regions]
                ax.bar(x + offsets[i], yvals, width=width,
                       color=scenario_colors[scenario],
                       label=scenario.split('_')[-1], alpha=0.9)

            # Formatting
            ax.axhline(0, color='black', lw=0.8, alpha=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(regions, rotation=45, ha='right',
                               fontsize=self.font_settings['tick_size'])
            ax.tick_params(axis='y', labelsize=self.font_settings['tick_size'])
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
            service_config = ServiceConfig.SERVICES[service]
            title = service_config.get('colorbar_label', service.title())
            ax.set_title(f"{title}",
                            fontsize=self.font_settings['title_size'], pad=8)
            # ax.set_title(f"{service.title()}",
            #              fontsize=self.font_settings['title_size'], pad=8)
            ax.grid(True, axis='y', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Add shared y-axis label
        fig.text(0.01, 0.5, f'% Change from Baseline to Final Year ({final_year})',
                 va='center', rotation='vertical',
                 fontsize=self.font_settings['label_size'])

        # Create single shared legend on the right
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, title="Scenario",
                   loc='center left', bbox_to_anchor=(0.93, 0.5),
                   fontsize=self.font_settings['tick_size'],
                   title_fontsize=self.font_settings['label_size'],
                   frameon=False)

        # Adjust layout — add space between panels
        plt.subplots_adjust(left=0.1, right=0.92, top=0.92, bottom=0.12, hspace=0.6)

        # Save figure
        output_dir = os.path.join(base_dir, 'summary_visualizations')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'summary_grouped_vertical_{final_year}.png')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Grouped summary saved to: {output_path}")

    def _read_data_from_working_csvs(self, services, target_years):
        """Read data from the same CSV files that the working bar charts use"""
        summary_data = {}
        
        for service in services:
            csv_path = os.path.join(self.project.cur_dir, f'{service}_combined', f'{service}_aez_stats.csv')
            print(f"Reading CSV: {csv_path}")
            
            if not os.path.exists(csv_path):
                print(f"CSV not found: {csv_path}")
                continue
            
            try:
                df = pd.read_csv(csv_path)
                print(f"Loaded {service} CSV: {len(df)} rows")
                print(f"Available scenarios: {df['scenario'].unique()}")
                print(f"Available years: {df['year'].unique()}")
                
                # Find baseline data - look for explicit baseline scenarios first
                baseline_df = df[df['scenario'].str.contains('baseline', case=False, na=False)]
                
                if baseline_df.empty:
                    print(f"No explicit baseline for {service}, checking other patterns...")
                    # Try other baseline detection methods
                    for pattern in ['base', '2017', '2015', '2020']:
                        potential_baseline = df[df['scenario'].str.contains(pattern, case=False, na=False)]
                        if not potential_baseline.empty:
                            baseline_df = potential_baseline
                            print(f"Found baseline using pattern '{pattern}'")
                            break
                    
                    # If still no baseline, find the BAU scenario (which acts as baseline for comparison)
                    if baseline_df.empty:
                        bau_baseline = df[df['scenario'].str.contains('bau', case=False, na=False)]
                        if not bau_baseline.empty:
                            # Use BAU from earliest year as baseline
                            earliest_year = bau_baseline['year'].min()
                            baseline_df = bau_baseline[bau_baseline['year'] == earliest_year]
                            print(f"Using BAU from year {earliest_year} as baseline")
                        else:
                            # Last resort: use any scenario from earliest year
                            earliest_year = df['year'].min()
                            earliest_scenarios = df[df['year'] == earliest_year]
                            if not earliest_scenarios.empty:
                                # Take just one scenario from earliest year
                                first_scenario = earliest_scenarios['scenario'].iloc[0]
                                baseline_df = earliest_scenarios[earliest_scenarios['scenario'] == first_scenario]
                                print(f"Using {first_scenario} from year {earliest_year} as baseline")
                
                if baseline_df.empty:
                    print(f"No baseline data found for {service}")
                    continue
                
                print(f"Baseline data: {len(baseline_df)} rows")
                print(f"Baseline scenario(s): {baseline_df['scenario'].unique()}")
                print(f"Baseline year(s): {baseline_df['year'].unique()}")
                
                # Calculate baseline means by region
                baseline_means = baseline_df.groupby('region_label')['mean_value'].mean()
                print(f"Baseline means for {len(baseline_means)} regions")
                
                # Get ALL scenario data for target years (don't exclude baseline scenarios)
                # We want to compare all scenarios in target years against the baseline
                scenario_df = df[df['year'].isin(target_years)]
                
                print(f"Scenario data: {len(scenario_df)} rows for years {target_years}")
                print(f"Scenario data scenarios: {scenario_df['scenario'].unique()}")
                
                # Calculate percent changes for ALL scenarios (including BAU if it exists)
                for (scenario, year), group in scenario_df.groupby(['scenario', 'year']):
                    print(f"Processing {scenario} - {year}")
                    
                    regional_changes = []
                    for _, row in group.iterrows():
                        region_name = row['region_label']
                        scenario_value = row['mean_value']
                        
                        if region_name in baseline_means:
                            baseline_val = baseline_means[region_name]
                            if baseline_val != 0:
                                pct_change = ((scenario_value - baseline_val) / baseline_val) * 100
                                regional_changes.append(pct_change)
                    
                    if regional_changes:
                        # Store region-level results instead of collapsing
                        if year not in summary_data:
                            summary_data[year] = {}
                        if service not in summary_data[year]:
                            summary_data[year][service] = {}
                        if scenario not in summary_data[year][service]:
                            summary_data[year][service][scenario] = {}

                        for _, row in group.iterrows():
                            region_name = row['region_label']
                            scenario_value = row['mean_value']
                            if region_name in baseline_means and baseline_means[region_name] != 0:
                                pct_change = ((scenario_value - baseline_means[region_name]) /
                                              baseline_means[region_name]) * 100
                                summary_data[year][service][scenario][region_name] = pct_change

                        mean_change = np.mean(list(summary_data[year][service][scenario].values()))
                        print(f"  {scenario} {year}: {mean_change:.1f}% change (region-level stored)")

                        
            except Exception as e:
                print(f"Error processing {service}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\nFinal summary data:")
        for year in summary_data:
            print(f"  {year}: {len(summary_data[year])} services")
            for service in summary_data[year]:
                print(f"    {service}: {len(summary_data[year][service])} scenarios")
        
        return summary_data

# ========================= MAIN ORCHESTRATOR =========================
class ScenarioProcessor:
    """Main orchestrator for processing scenarios"""
    
    def __init__(self, project, font_settings=None):
        self.project = project
        self.raster_processor = RasterProcessor(project)
        self.stats_calculator = ZonalStatsCalculator(project)
        self.visualizer = None
        self.font_settings = font_settings or ServiceConfig.FONT_SETTINGS
    
    def process_all_scenarios(self):
        """Main entry point - process all scenarios"""
        
        # Pre-clip all rasters
        clipped_dirs = self.raster_processor.clip_all_rasters()
        if not clipped_dirs:
            return
        
        # Pass font settings to visualizer
        self.visualizer = MapVisualizer(clipped_dirs, self.font_settings)
        self.visualizer.project = self.project
        
        # Get region mappings
        aez_mapping = self._get_region_mapping('aez')
        county_mapping = self._get_region_mapping('county')
        
        # Process scenarios and collect results
        results = self._process_scenarios()
        
        # Save results and create combined visualizations
        self._save_results(results, aez_mapping, county_mapping)
        
        # Create summary heatmap from existing CSV files
        summary_viz = SummaryPlot(self.project, self.font_settings)
        # summary_viz.create_faceted_summary(self.project.cur_dir)  
        summary_viz.create_grouped_summary(self.project.cur_dir)  
    
    def _get_region_mapping(self, region_type):
        """Get region ID to label mapping"""
        if region_type == 'aez':
            vector_path = self.project.regions_vector_path_epsg8857
            id_column = self.project.region_id_column
            label_column = self.project.region_label_column
        else:
            vector_path = self.project.alt_regions_vector_path_epsg8857
            id_column = self.project.alt_region_id_column
            label_column = self.project.alt_region_label_column
        
        try:
            regions_gdf = gpd.read_file(vector_path)
            mapping = {}
            for _, row in regions_gdf.iterrows():
                region_id = int(row[id_column])
                region_label = row[label_column]
                mapping[region_id] = region_label
            return mapping
        except Exception:
            return {}
    
    def _process_scenarios(self):
        """Process all scenarios and collect statistics"""
        results = {service: {'stats': [], 'baseline': []} 
                  for service in ServiceConfig.SERVICES.keys()}
        
        for index, row in self.project.scenarios_df.iterrows():
            seals_utils.assign_df_row_to_object_attributes(self.project, row)
            
            years, template, scenario_name, is_baseline = self._get_scenario_info()
            
            for year in years:
                scenario_year = template.format(year)
                
                for service_type in ServiceConfig.SERVICES.keys():
                    # Calculate stats for both AEZ and County
                    for region_type in ['aez', 'county']:
                        vector_path = (self.project.regions_vector_path_epsg8857 if region_type == 'aez'
                                      else self.project.alt_regions_vector_path_epsg8857)
                        
                        stats_dict = self.stats_calculator.calculate_stats(
                            scenario_year, service_type, vector_path
                        )
                        
                        if stats_dict:
                            target_list = results[service_type]['baseline' if is_baseline else 'stats']
                            target_list.append((stats_dict, scenario_name, year))
        
        return results
    
    def _get_scenario_info(self):
        """Get scenario information"""
        if self.project.scenario_type == 'baseline':
            return (
                self.project.base_years, 
                'baseline_{}',
                'baseline',
                True
            )
        else:
            template = f'{self.project.lulc_src_label}_{self.project.lulc_simplification_label}_{self.project.exogenous_label}_{self.project.climate_label}_{self.project.model_label}_{self.project.counterfactual_label}_{{}}'
            scenario_name = f'{self.project.lulc_src_label}_{self.project.lulc_simplification_label}_{self.project.exogenous_label}_{self.project.climate_label}_{self.project.model_label}_{self.project.counterfactual_label}'
            return (
                self.project.years,
                template, 
                scenario_name,
                False
            )
    
    def _save_results(self, results, aez_mapping, county_mapping):
        """Save results and create combined visualizations"""
        for service_type, data in results.items():
            output_dir = os.path.join(self.project.cur_dir, f'{service_type}_combined')
            os.makedirs(output_dir, exist_ok=True)
            
            # Save CSVs for both region types
            for region_type, region_mapping in [('aez', aez_mapping), ('county', county_mapping)]:
                self._save_csv_results(data['stats'], data['baseline'], output_dir, 
                                      service_type, region_type, region_mapping)
            
            # Create combined visualizations
            if data['stats']:
                scenarios_dict = {}
                for stats_dict, sname, year in data['stats']:
                    if sname not in scenarios_dict:
                        scenarios_dict[sname] = []
                    scenarios_dict[sname].append((stats_dict, sname, year))
                
                for scenario_name, scenario_data in scenarios_dict.items():
                    # Determine if BAU
                    is_bau = 'bau' in scenario_name.lower()
                    
                    # Create visualization
                    self.visualizer.create_combined_visualization(
                        scenario_data, data['baseline'], output_dir, 
                        service_type, scenario_name, self.project,
                        aez_mapping, county_mapping, is_bau
                    )
    
    def _save_csv_results(self, stats_data, baseline_data, output_dir, service_type, 
                         region_type, region_mapping):
        """Save CSV results for a region type"""
        if not stats_data:
            return
        
        is_lulc = service_type == 'lulc'
        
        # Convert to DataFrames
        all_data = []
        for stats_dict, scenario_name, year in stats_data:
            df = DataFormatter.stats_to_dataframe(stats_dict, scenario_name, year, 
                                                 region_mapping, is_lulc)
            all_data.append(df)
        
        if all_data:
            master_df = pd.concat(all_data, ignore_index=True)
            
            # --- FILTER: keep only regions that are in the provided mapping ---
            # mapping may be id->label. Build allowed sets for both id and label.
            allowed_labels = set(region_mapping.values())
            allowed_ids = set(region_mapping.keys())
            
            # If DataFormatter produced numeric region_id column, prefer that filter
            if 'region_id' in master_df.columns:
                before = len(master_df)
                master_df = master_df[master_df['region_id'].isin(allowed_ids)]
                after = len(master_df)
                if after < before:
                    L.info(f"Filtered {before-after} rows not in {region_type} mapping (region_id filter).")
            else:
                # Fallback to filter by region_label (string)
                before = len(master_df)
                master_df = master_df[master_df['region_label'].isin(allowed_labels)]
                after = len(master_df)
                if after < before:
                    removed = set(master_df['region_label'].unique()) ^ allowed_labels
                    L.info(f"Filtered {before-after} rows not in {region_type} mapping (region_label filter).")
                    L.debug(f"Removed labels (sample): {list(removed)[:10]}")
            # -----------------------------------------------------------------
            
            csv_filename = f'{service_type}_{region_type}_stats.csv'
            csv_path = os.path.join(output_dir, csv_filename)
            master_df.to_csv(csv_path, index=False)
# ...existing code...

# ========================= MAIN FUNCTION =========================
def summarize_and_visualize_multi_vector(p):
    """Main function to process all scenarios"""
    from osgeo import gdal, ogr
    # suppress all OGR/GDAL warnings
    gdal.DontUseExceptions()
    ogr.DontUseExceptions()

    processor = ScenarioProcessor(p)
    processor.process_all_scenarios()


