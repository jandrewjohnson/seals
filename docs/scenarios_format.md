# scenarios.csv Format

The `scenarios.csv` file defines your SEALS scenarios. Each row represents one scenario.

## Column Reference

| Column | Description | Example |
|--------|-------------|---------|
| `scenario_label` | Unique scenario identifier | `baseline`, `ssp2_rcp45` |
| `scenario_type` | `baseline` or `policy` | `baseline` |
| `aoi` | Area of interest | `global` or `from_regional_projections_input_path` |
| `exogenous_label` | Exogenous scenario name | `baseline` |
| `climate_label` | Climate scenario | `rcp45` |
| `model_label` | Model source | `gtap`, `magpie`, `luh2` |
| `counterfactual_label` | Counterfactual reference | `baseline` |
| `years` | Space-separated projection years | `2030 2050` |
| `baseline_reference_label` | Reference scenario for policy | `baseline` (blank for baseline) |
| `base_years` | Base year(s) | `2017` |
| `key_base_year` | Key base year for LULC | `2017` |
| `comparison_counterfactual_labels` | For comparisons | (usually blank) |
| `time_dim_adjustment` | Time adjustment | `add2015` |
| `coarse_projections_input_path` | Path to LUH2 data | `luh2/raw_data/rcp45_ssp2/...nc` |
| `lulc_src_label` | LULC source | `esa` |
| `lulc_simplification_label` | LULC class scheme | `seals7` |
| `lulc_correspondence_path` | LULC correspondence file | `seals/default_inputs/esa_seals7_correspondence.csv` |
| `coarse_src_label` | Coarse data source | `luh2-14` |
| `coarse_simplification_label` | Coarse class scheme | `seals7` |
| `coarse_correspondence_path` | Coarse correspondence file | `seals/default_inputs/luh2-14_seals7_correspondence.csv` |
| `lc_class_varname` | Land class variable | `all_variables` |
| `dimensions` | Dimensions | `time` |
| `calibration_parameters_source` | Calibration file | `seals/default_inputs/default_global_coefficients.csv` |
| `base_year_lulc_path` | Base year LULC raster | `lulc/esa/lulc_esa_2017.tif` |
| `regional_projections_input_path` | Regional projections CSV (optional) | `regional_projections/my_projections.csv` |
| `regions_vector_path` | Regions shapefile | `cartographic/countries_iso3_with_label.gpkg` |
| `regions_column_label` | Region column name | `iso3_label` |

## Example

```csv
scenario_label,scenario_type,aoi,exogenous_label,climate_label,model_label,counterfactual_label,years,baseline_reference_label,base_years,key_base_year,comparison_counterfactual_labels,time_dim_adjustment,coarse_projections_input_path,lulc_src_label,lulc_simplification_label,lulc_correspondence_path,coarse_src_label,coarse_simplification_label,coarse_correspondence_path,lc_class_varname,dimensions,calibration_parameters_source,base_year_lulc_path,regional_projections_input_path,regions_vector_path,regions_column_label
baseline,baseline,global,baseline,rcp45,luh2,baseline,2017,,2017,2017,,add2015,luh2/raw_data/rcp45_ssp2/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-MESSAGE-ssp245-2-1-f_gn_2015-2100.nc,esa,seals7,seals/default_inputs/esa_seals7_correspondence.csv,luh2-14,seals7,seals/default_inputs/luh2-14_seals7_correspondence.csv,all_variables,time,seals/default_inputs/default_global_coefficients.csv,lulc/esa/lulc_esa_2017.tif,,cartographic/countries_iso3_with_label.gpkg,iso3_label
ssp2_rcp45,policy,global,ssp2,rcp45,luh2,ssp2_rcp45,2030 2050,baseline,2017,2017,,add2015,luh2/raw_data/rcp45_ssp2/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-MESSAGE-ssp245-2-1-f_gn_2015-2100.nc,esa,seals7,seals/default_inputs/esa_seals7_correspondence.csv,luh2-14,seals7,seals/default_inputs/luh2-14_seals7_correspondence.csv,all_variables,time,seals/default_inputs/default_global_coefficients.csv,lulc/esa/lulc_esa_2017.tif,,cartographic/countries_iso3_with_label.gpkg,iso3_label
```
