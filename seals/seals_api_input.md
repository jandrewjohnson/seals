# Setup parameters

-   Description: This file defines the input to a standard SEALS run. The markdown notation here is parsed to a csv file saved at scenario_definitions.csv, which follows the v2 scenario definitions schema specified in earth_economy_devstack/docs/scenario_definitions.qmd. Parsing conventions from v2 apply — an empty cell is the only null representation (never a "nan" or "none" literal), lists are single-space-separated within a cell, and paths always use forward slashes. If you have just installed SEALS, be sure to run the "run_test_standard.py" file to generate an example of the CSV for you to inspect, edit, or use as a template.

## aoi

-   Description: Sets the area of interest. If set as a region_label (e.g., a country-ISO3 codes), all data will be generated based that regions boundaries (as defined in the regions_vector_path). Other options include setting it to "global" or a specific shapefile, or iso3 code. Good small examples include RWA, BTN. Can be either 3-letter ISO code, keyword "global", or is a path.
-   Type: string, path
-   Default: "RWA"
-   Required: True
-   Examples: "RWA", "BDI", "global", "cartographic/ee/seals/default_inputs/rwa_bdi.shp"

## regions_vector_path

-   Description: Path to the vector file of the regions present. This should be a geopackage (gpkg) file with a set of polygons paired with a unique region_label. This label should be defined in some regional correspondence file, ideally in the ee_r264_correspondence or the eemarine_r566_correspondence in the base_data. If regional projections are used, the region_labels must correspond to those defined in regional_projections_input_path.
-   Type: path
-   Default: "cartographic/ee/ee_r264_correspondence.gpkg"
-   Required: True
-   Examples: "cartographic/ee/ee_r264_correspondence.gpkg", "cartographic/ee/eemarine_r566_correspondence.gpkg"
-   Note: Renamed from regional_boundaries_input_path to match the v2 column name.

## regions_column_label

-   Description: Column in regions_vector_path whose values are the unique region labels used to join regional data.
-   Type: string
-   Default: "ee_r264_label"
-   Required: True
-   Examples: "ee_r264_label", "ee_r50_aez18_id"

## calibration_parameters_path

-   Description: Path to a csv which contains all of the pretrained regressor variables. Can also be "from_calibration" indicating that this run will actually create the calibration or it can be from a tile-designated file of location-specific regressor variables.
-   Type: path
-   Default: "seals/default_inputs/default_global_coefficients.csv"
-   Required: True
-   Examples: "seals/default_inputs/default_global_coefficients.csv", "from_calibration", "tile_designated"
-   Note: Renamed from calibration_parameters_source in v2 so the _path suffix types it correctly.

## base_year_lulc_path

-   Description: Path to the LULC map for the canonical base_year. This is the data that will be used to determine the resolution, extent, and projection of the fine resolution LULC data. On an observed row this anchors the scenario set. It must exist at load, or be a cat-ears-templated product of a depends_on row (a constructed anchor — see the v2 spec).
-   Type: path
-   Default: "seals/default_inputs/esa_2017.tif"
-   Required: True
-   Examples: "seals/default_inputs/esa_2017.tif", "seals/default_inputs/esa_1992.tif", "seals/default_inputs/esa_2020.tif"

# Projection inputs

-   Description: SEALS supports 2 types of projection inputs: regional and coarse. Regional projections, defined via a vector file (gpkg) and some distribution algorithm (e.g. proportional allocation). Coarse are coarse-resolution gridded data, such as the 30km resolution outputs of IAMs.

## regional_projections_input_path

-   Description: Path to the regional land-use change data. This should be a table (csv) file with a region_label corresponding to a polygon in regions_vector_path. In this table, changes are reported in hectarage values for each region_label for each of the changing_lulc_classes.
    -   This is not required, but if it is not provided, the coarse projections must be a raster path or raster-producing task. This is because the other coarse input option is an integer resolution for the proportional allocation of the regional projections on the coarse grid of the integer's resolution (See below).
-   Type: path
-   Default: "cartographic/ee/seals/default_inputs/rwa_bdi_regional_changes.csv"
-   Required: False
-   Examples: "cartographic/ee/seals/default_inputs/rwa_bdi_regional_changes.csv"

## region_to_coarse_algorithm

-   Description: Algorithm used to distribute regional (vector) projections onto the coarse grid.
-   Type: string
-   Default: "covariate_sum_shift"
-   Required: False
-   Examples: "covariate_sum_shift"

## coarse_projections_input_path

-   Description: Designates how to handle the coarse land-use change projections. The coarse projections are always gridded, but are much coarser than the output LULC's fine resolution. Can be a path to a coarse-gridded set of rasters for each land-use change and time, can be an integer indicating the resolution of the coarse projection (but assuming a simple smooth proportional allocation of the regional projections to the coarse projections), or can be a string indicating the name of a task in the ProjectFlow object that will generate the coarse projections.
    -   If pointing to a raster, it should be (referred) a directory of geotiffs where directories follow the scenario_structure defined in exogenous_label, climate_label, model_label, counterfactual_label, and years. The other type is a single netcdf file that contains all of the data. If it is a netcdf, you may optionally use the time_dim_adjustment parameter to adjust the time dimension to match the desired format.
        -   If it is a path, it should ideally be a ref_path (meaning that it is included in the base_data and is relative to the base_data_dir and can be found via p.get_path())
    -   If it is an integer, it should express a pyramid-compatible integer of arcsecond resolution, specifically 10, 30, 150, 300, 900, 1800, 3600, 7200, 14400, 36000 (though anything less than 300 might be weirdly small)
    -   If is a task, it must be an algorithm that produces a directory of geotiffs, as discussed above
-   Type: path, Integer, task_name
-   Default: "luh2/raw_data/rcp45_ssp2/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-MESSAGE-ssp245-2-1-f_gn_2015-2100.nc"
-   Required: True
-   Examples: "luh2/raw_data/rcp45_ssp2/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-MESSAGE-ssp245-2-1-f_gn_2015-2100.nc", "luh2/raw_data/rcp19_ssp1/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-IMAGE-ssp119-2-1-f_gn_2015-2100.nc", "luh2/raw_data/rcp26_ssp1/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-IMAGE-ssp126-2-1-f_gn_2015-2100.nc", "luh2/raw_data/rcp70_ssp3/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-AIM-ssp370-2-1-f_gn_2015-2100.nc", "luh2/raw_data/rcp34_ssp4/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-GCAM-ssp434-2-1-f_gn_2015-2100.nc", "luh2/raw_data/rcp60_ssp4/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-GCAM-ssp460-2-1-f_gn_2015-2100.nc", "luh2/raw_data/rcp34_ssp5/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-MAGPIE-ssp534-2-1-f_gn_2015-2100.nc", "luh2/raw_data/rcp85_ssp5/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-MAGPIE-ssp585-2-1-f_gn_2015-2100.nc", "900", "1800", "coarse_simplified_projected_ha_difference_from_previous_year",

# Scenario structure

-   Description: Every row of the scenarios csv is either an observed anchor or a runnable scenario. An observed row defines the historical anchor (the canonical base year, the base-year LULC map, calibration inputs) and runs no projection years. A bau or policy row is a trajectory to simulate. Three concerns that v1 mixed together are now separate columns — where is my anchor data (observed_reference_label), what must run before me (depends_on), and what am I compared against (compared_to). Output directories compose as exogenous_label/climate_label/model_label/counterfactual_label/year.

## scenario_label

-   Description: String that uniquely identifies the scenario. Will be referenced by other scenarios via observed_reference_label, depends_on and compared_to. A label starting with # marks the whole row as a comment and it is skipped entirely.
-   Type: string
-   Default: None
-   Required: True
-   Examples: "ssp2_rcp45_luh2-message_bau",

## scenario_type

-   Description: Closed vocabulary of observed, bau, or policy. An observed row is the historical anchor (data preparation, not a counterfactual). A bau row is the business-as-usual trajectory. A policy row is a counterfactual compared against the scenarios in compared_to. Anything else is a load-time error. Replaces the v1 vocabulary in which "baseline" conflated the observed anchor with the bau trajectory.
-   Type: string
    -   One of \["observed", "bau", "policy"\]
-   Default: None
-   Required: True
-   Examples: "bau"

## run

-   Description: Include this row in execution. Replaces commenting-out or deleting rows. Keep the full run and a fast test configuration as two rows and flip these flags.
-   Type: bool
-   Default: 1
-   Required: False
-   Examples: 1, 0

## ignore_dependencies

-   Description: Skip validation of and waiting on cross-scenario upstream artifacts (assume present or regenerate). This is the boolean that v1 smuggled into scenario_type as "baseline_ignore_dependencies".
-   Type: bool
-   Default: 0
-   Required: False
-   Examples: 0, 1

## tags

-   Description: Free-form grouping labels (e.g. stress_test) for filtering and reporting. Never read by run logic.
-   Type: space delimited list of strings
-   Default: None
-   Required: False
-   Examples: "stress_test ngfs_phase4"

## observed_reference_label

-   Description: Which observed row anchors this scenario (base-year data and calibration lookup). Required on bau and policy rows. Replaces v1 baseline_reference_label.
-   Type: label in scenario_label
-   Default: None
-   Required: True
-   Examples: "observed_esa_2017"

## depends_on

-   Description: Scenarios that must complete before this row runs. Defines the execution DAG. The parser topologically sorts and cycles are a load-time error. Also allowed on observed rows whose anchor is produced by another row (constructed anchors — see the v2 spec).
-   Type: space delimited list of labels
-   Default: None
-   Required: False
-   Examples: "gtap_initial"

## compared_to

-   Description: Scenario(s) this row is differenced against in reporting and comparison tasks. Required on policy rows. Replaces v1 comparison_counterfactual_labels.
-   Type: space delimited list of labels
-   Default: None
-   Required: True
-   Examples: "bau"

## exogenous_label

-   Description: Exogenous label references some set of exogenous drivers like population, TFP growth, LUH2 pattern, SSP database etc
-   Type: string
-   Default: None
-   Required: True
-   Examples: "ssp2"

## climate_label

-   Description: One of the climate RCPs
-   Type: string
-   Default: None
-   Required: True
-   Examples: "rcp45"

## model_label

-   Description: Indicator of which model led to the coarse projection
-   Type: string
-   Default: None
-   Required: True
-   Examples: "luh2-image", "luh2-remind", "luh2-aim", "luh2-gcam", "luh2-magi", "luh2-message", "luh2-merlin", "luh2-orchidee", "luh2-plum", "luh2-sleuth", "luh2-terra", "luh2-wasp"

## counterfactual_label

-   Description: AKA policy scenario, or a label of something that you have tweaked to assess it"s efficacy
-   Type: string
-   Default: None
-   Required: True
-   Examples: "bau", "no_policy", "policy1", "policy2", "policy3"

## base_year

-   Description: The single canonical anchor year of the project clock, always a scalar. It must refer to a year the anchoring observed row can supply data for — either truly observed LULC (with ESACCI that means 2022 or earlier) or a constructed anchor produced by an upstream row. It must never appear in years. Replaces v1 key_base_year (and the v1 scalar-versus-list asymmetry is gone).
-   Type: integer
-   Default: 2017
-   Required: True
-   Examples: 2017

## observed_years

-   Description: Additional observed or historical years available for calibration or validation. Defaults to just base_year. Replaces v1 base_years.
-   Type: space delimited list of integers
-   Default: 2017
-   Required: False
-   Examples: 2017, 2000 2005 2010 2015

## years

-   Description: The canonical years to simulate and report — the clock the whole project exchanges data on. Must never include base_year (that is the anchor, never a simulated step). May contain years before and/or after base_year (hindcasting). Empty on observed rows. In multi-stage projects, other model stages may declare their own native clocks with stage-prefixed columns (e.g. gtap_base_year, gtap_timestep) that the framework harmonizes to these canonical years — SEALS resolves the plain columns and is typically the canonical clock itself. See the harmonization section of the v2 spec.
-   Type: space delimited list of integers
-   Default: 2030 2050
-   Required: True
-   Examples: 2030 2050, 2025 2030 2035 2040 2045 2050

# Correspondences

-   Description: There are two different correspondences that are typically used to map inputs to that required by seals. First is an LULC correspondence that maps the input, e.g., ESA CCI 37 classes, to some simpler classification for which there exists a SEALS calibration file, e.g., seals7. The second is the coarse-gridded land-use projection that maps the input, e.g., LUH2-14, to some simpler classification for which there exists a SEALS calibration file, e.g., seals7.

## lulc_src_label

-   Description: Label of the LULC data being reclassified into the simplified form.
-   Type: string
-   Default: None
-   Required: True
-   Examples: "esa"

## lulc_simplification_label

-   Description: Label of the new LULC simplified classification
-   Type: string
-   Default: None
-   Required: True
-   Examples: "seals7"

## lulc_correspondence_path

-   Description: Path to a csv that will map the a many-to-one reclassification of the src LULC map to a simplified version
-   Type: path
-   Default: None
-   Required: True
-   Examples: "seals/default_inputs/esa_seals7_correspondence.csv"
-   Note: Becasue this is a correspondence file, it should either have columns for at least src_id, dst_id, src_label, dst_label. Or, it should follow the multicorrespondence format, such as in ee_r264_correspondence.csv that specifies domain, attribute and size (but still follows multiple nested many-to-one relationships)

## nonchanging_class_indices

-   Description: To speed up processing, select which classes you know won"t change. For default seals7, this is the urban classes, the water classes, and the bare land class.
-   Type: list of integers
-   Default: \[0, 6, 7\]
-   Required: True
-   Examples: \[0, 6, 7\]

## coarse_src_label

-   Description: Label of the coarse LUC data that will be reclassified to the required coarse gridded projection data used as an input to the coarse gridded allocation.
-   Type: string
-   Default: None
-   Required: True
-   Examples: "luh2-14"

## coarse_simplification_label

-   Description: Label of the simplified coarse LUC data that matches the simplified lulc classification and has a corresponding calibration file.
-   Type: string
-   Default: None
-   Required: True
-   Examples: "seals7"

## coarse_correspondence_path

-   Description: Path to a csv that includes at least src_id, dst_id, src_label, dst_label
-   Type: path
-   Default: None
-   Required: True
-   Examples: "seals/default_inputs/luh2-14_seals7_correspondence.csv"
-   Note: Becasue this is a correspondence file, it should either have columns for at least src_id, dst_id, src_label, dst_label. Or, it should follow the multicorrespondence format, such as in ee_r264_correspondence.csv that specifies domain, attribute and size (but still follows multiple nested many-to-one relationships)

# Coarse projections preprocessing

-   Description: These keywords allow for automatically extracting the correct parts of potentially malformed netcdfs and harmonizing their time axis with the canonical clock.

## time_dim_adjustment

-   Description: Often NetCDF files can have the time dimension in something other than just the year. This string allows for doing operations on the time dimension to match what is desired. e.g., multiply5 add2015. This is calendar alignment only — filling years the source lacks is governed by coarse_interpolation.
-   Type: string
-   Default: None
-   Required: True
-   Examples: "add2015"

## lc_class_varname

-   Description: Because different NetCDF files have different arrangements (e.g. time is in the dimension versus LU_class is in the dimension), this option allows you to specify where in the input NC the information is. If "all_variables", assumes the LU classes will be the different variables named otherwise it can be a subset of variables, otherwise, if it is a named variable, e.g. LC_area_share then assume that the lc_class variable is
-   Type: string
-   Default: "all_variables"
-   Required: True
-   Examples: "all_variables"

## dimensions

-   Description: Lists which dimensions are stored in the netcdf in addition to lat and lon. Ideally this is just time but sometimes there are more. \# From the csv, this is a space-separated list.
-   Type: string
-   Default: None
-   Required: True
-   Examples: "time"

## coarse_interpolation

-   Description: How to fill requested years that are missing from the coarse projection source. With MAgPIE or LUH2 5-year bands, a requested year between bands is filled from the neighboring bands instead of being silently skipped. One of linear, nearest, hold, or none (with none, a missing year is a load-time error).
-   Type: string
-   Default: "linear"
-   Required: False
-   Examples: "linear"
