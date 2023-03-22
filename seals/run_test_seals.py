import os
import hazelbean as hb

main = ''
if __name__ == '__main__':

    #### These next three lines should be the only computer-specific things to set. Everything is relative to these (or the source code dir)

    # A ProjectFlow object is created from the Hazelbean library to organize directories and enable parallel processing.
    # project-level variables are assigned as attributes to the p object (such as in p.base_data_dir = ... below)
    p = hb.ProjectFlow('..\\..\\projects\\test_seals_jaj_workstation2022')

    # Set the base data dir. The model will check here to see if it has everything it needs to run.
    # If anything is missing, it will download it. You can use the same base_data dir across multiple projects.
    p.base_data_dir = os.path.join('C:\\', 'Users', 'jajohns', 'Files', 'Research', 'base_data') # This is where the minimum set of downloaded files goes.

    # In order for SEALS to download using the google_cloud_api service, you need to have a valid credentials JSON file.
    # Identify its location here. If you don't have one, email jajohns@umn.edu. The data are freely available but are very, very large
    # (and thus expensive to host), so I limit access via credentials.
    p.data_credentials_path = '..\\api_key_credentials.json'

    # There are different versions of the base_data in gcloud, but best open-source one is 'seals_public_2022-03-01'
    p.input_bucket_name = 'seals_public_2022-03-01'

    # Set the area of interest. If set as a country-ISO3 code, all data will be generated based
    # that countries boundaries (as defined in the base data). Other options include setting it to
    # 'global' or a specific shapefile.
    p.aoi = 'RWA'

    # Set the training start year and end year. These years will be used for calibrating the model. Once calibrated, project forward
    # from the base_year (which could be the same as the training_end_year but not necessarily).
    p.training_start_year = 2000
    p.training_end_year = 2015
    p.base_year = 2015

    # For GTAP-enabled runs, we project the economy from the latest GTAP reference year to the year in which a
    # policy is made so that we can apply the policy to a future date. Set that policy year here. (Only affects runs if p.is_gtap_run is True)
    p.policy_base_year = 2021

    # Define terminal year for simulation. This is only partly implemented but will be finished when switching to GTAP-AEZ-RD.
    p.scenario_years = [2050]

    # Define which meso-level LUC and Climate scenarios will be used.
    p.luh_scenario_labels = ['rcp45_ssp2']

    # SEALS has two resolutions: fine and coarse. In most applications, fine is 10 arcseconds (~300m at equator, based on ESACCI)
    # and coarse is based on IAM results that are 900 arcseconds (LUH2) or 1800 arcseconds (MAgPIE). Note that there is a coarser-yet
    # scale possible from e.g. GTAP-determined endogenous LUC. This is excluded in the base SEALS config.
    p.fine_resolution_arcseconds = 10.0 # MUST BE FLOAT
    p.coarse_resolution_arcseconds = 900.0 # MUST BE FLOAT

    # To run a much faster version for code-testing purposes, enable test_mode. Selects a much smaller set of scenarios and spatial tiles.
    p.test_mode = True

    # In order to apply this code to the magpie model, I set this option to either
    # use the GTAP-shited LUH data (as was done in the WB feedback model)
    # or to instead use the outputs of some other extraction functions with
    # no shifting logic. This could be scaled to different interfaces
    # when models have different input points.
    p.is_magpie_run = False

    # For the intitial magpie run, we enforced that the amount of ag land in ESA had to match that in Magpie. This enables that option.
    p.adjust_baseline_to_match_magpie_2015 = False

    p.is_gtap1_run = False
    p.is_calibration_run = True
    p.is_standard_seals_run = False


    ############# Below here shouldn't need editing, but it may explain what's happening ###########

    # Configure the logger that captures all the information generated.
    p.L = hb.get_logger('test_run_seals')

    p.L.info('Created ProjectFlow object at ' + p.project_dir + '\n    '
           'from script ' +  p.calling_script + '\n    '
           'with base_data set at ' + p.base_data_dir)

    import seals_main
    import seals_generate_base_data
    import seals_process_luh2
    import config

    # initialize and set all basic variables. Sadly this is still needed even for a SEALS run until it's extracted.
    seals_main.initialize_paths(p)


    # Run configuration options that affect performance.
    p.num_workers = 14 # None sets it to max available.
    p.reporting_level = 15
    p.calibration_reporting_level = 0
    p.output_writing_level = 5 # >=2 writes chunk-baseline lulc
    p.build_overviews_and_stats = 0 # For later fast-viewing, this can be enabled to write ovr files and geotiff stats files.
    p.write_projected_coarse_change_chunks = 1 # in the SEALS allocation, for troubleshooting, it can be useful to see what was the coarse allocation input.

    # Sometimes runs fail mid run. This checks for that and picks up where there is a completed file for that zone.
    p.skip_created_downscaling_zones = 0

    # As it calibrates, optionally write each calibration allocation step
    p.write_calibration_generation_arrays = 1

    # TODOO: Define a task tree consistent both with magpie needing esa-magpie 2015 calibration AND
    # gtap needing an extra base-year of 2021 AND gtap being a 3-layer allocation with SSP2 (which should be made interchangeable with Magpie)
    if p.is_gtap1_run:
        p.base_years = [p.base_year, p.policy_base_year]
    elif p.is_magpie_run:
        p.base_years = [p.base_year]
    elif p.is_calibration_run:
        p.base_years = [p.base_year]

    # Before training and running the model, the LULC map is simplified to fewer classes. Each one currently has a label
    # such as 'seals7', defined in the config code.
    p.lulc_remap_labels = {}
    p.lulc_simplification_label = 'seals7'
    p.lulc_label = 'lulc_esa'
    p.lulc_simplified_label = 'lulc_esa_' + p.lulc_simplification_label
    p.lulc_simplification_remap = config.esa_to_seals7_correspondence

    # THIS CODE IS REDUNDANT AND NEEDS TO BE REFACTORED to draw from the simplifcation remap or a more robust type of input CSV.
    # SEALS-simplified classes are defined here, which can be iterated over. We also define what classes are shifted by GTAP's endogenous land-calcualtion step.
    p.class_indices = [1, 2, 3, 4, 5] # These are the indices of classes THAT CAN EXPAND/CONTRACT
    p.nonchanging_class_indices = [6, 7] # These add other lulc classes that might have an effect on LUC but cannot change themselves (e.g. water, barren)
    p.regression_input_class_indices = p.class_indices + p.nonchanging_class_indices

    p.class_labels = ['urban', 'cropland', 'grassland', 'forest', 'nonforestnatural',]
    p.nonchanging_class_labels = ['water', 'barren_and_other']
    p.regression_input_class_labels = p.class_labels + p.nonchanging_class_labels

    p.shortened_class_labels = ['urban', 'crop', 'past', 'forest', 'other',]

    p.class_indices_that_differ_between_ssp_and_gtap = [2, 3, 4,]
    p.class_labels_that_differ_between_ssp_and_gtap = ['cropland', 'grassland', 'forest',]

    # Specifies which sigmas should be used in a gaussian blur of the class-presence tifs in order to regress on adjacency.
    # Note this will have a huge impact on performance as full-extent gaussian blurs for each class will be generated for
    # each sigma.
    p.gaussian_sigmas_to_test = [1, 5]

    # Change how many generations of training to allow. A generation is an exhaustive search so relatievely few generations are required to get to a point
    # where no more improvements can be found.
    p.num_generations = 1

    # Provided by GTAP team.
    # TODOO This is still based on the file below, which was from Purdue. It is a vector of 300sec gridcells and should be replaced with continuous vectors
    p.gtap37_aez18_input_vector_path = os.path.join(p.base_data_dir, "pyramids", "GTAP37_AEZ18.gpkg")
    p.use_calibration_from_zone_centroid_tile = 1
    p.use_calibration_created_coefficients = 1
    p.calibration_zone_polygons_path = os.path.join(p.gtap37_aez18_input_vector_path)  # Only needed if use_calibration_from_zone_centroid_tile us True.

    #### Here is very ugly code that makes sure the different run configs point to the right files. This need to be fixed as an override of a default.
    # TODOO For magpie integration: make this an override.
    p.baseline_labels = ['baseline']
    p.baseline_coarse_state_paths = {}
    p.baseline_coarse_state_paths['baseline'] = {}
    p.baseline_coarse_state_paths['baseline'][p.base_year] = os.path.join(p.input_dir, "SSP2_BiodivPol_LPJmL5_2021-05-21_15.08.06", "cell.land_0.5_share_to_seals_SSP2_BiodivPol_LPJmL5.nc")

    # These are the POLICY scenarios. The model will iterate over these as well.
    p.gtap_combined_policy_scenario_labels = ['BAU', 'BAU_rigid', 'PESGC', 'SR_Land', 'PESLC', 'SR_RnD_20p', 'SR_Land_PESGC', 'SR_PESLC',  'SR_RnD_20p_PESGC', 'SR_RnD_PESLC', 'SR_RnD_20p_PESGC_30']
    p.gtap_just_bau_label = ['BAU']
    p.gtap_bau_and_30_labels = ['BAU', 'SR_RnD_20p_PESGC_30']
    p.luh_labels = ['no_policy']

    p.magpie_policy_scenario_labels = [
        'SSP2_BiodivPol_LPJmL5',
        'SSP2_BiodivPol_ClimPol_LPJmL5',
        'SSP2_BiodivPol_ClimPol_NCPpol_LPJmL5',
        'SSP2_ClimPol_LPJmL5',
        'SSP2_NPI_base_LPJmL5',
    ]

    p.magpie_test_policy_scenario_labels = [
        # 'SSP2_BiodivPol_LPJmL5',
        # 'SSP2_BiodivPol_ClimPol_LPJmL5',
        'SSP2_BiodivPol_ClimPol_NCPpol_LPJmL5',
        # 'SSP2_ClimPol_LPJmL5',
        # 'SSP2_NPI_base_LPJmL5',
    ]

    # Scenarios are defined by a combination of meso-level focusing layer that defines coarse LUC and Climate with the policy scenarios (or just scenarios) below.
    p.magpie_scenario_coarse_state_paths = {}
    p.magpie_scenario_coarse_state_paths['rcp45_ssp2'] = {}
    p.magpie_scenario_coarse_state_paths['rcp45_ssp2'][2050] = {}
    p.magpie_scenario_coarse_state_paths['rcp45_ssp2'][2050]['SSP2_BiodivPol_LPJmL5'] = os.path.join(p.input_dir, "SSP2_BiodivPol_LPJmL5_2021-05-21_15.08.06", "cell.land_0.5_share_to_seals_SSP2_BiodivPol_LPJmL5.nc")
    p.magpie_scenario_coarse_state_paths['rcp45_ssp2'][2050]['SSP2_BiodivPol_ClimPol_LPJmL5'] = os.path.join(p.input_dir, "SSP2_BiodivPol+ClimPol_LPJmL5_2021-05-21_15.09.32", "cell.land_0.5_share_to_seals_SSP2_BiodivPol+ClimPol_LPJmL5.nc")
    p.magpie_scenario_coarse_state_paths['rcp45_ssp2'][2050]['SSP2_BiodivPol_ClimPol_NCPpol_LPJmL5'] = os.path.join(p.input_dir, "SSP2_BiodivPol+ClimPol+NCPpol_LPJmL5_2021-05-21_15.10.54", "cell.land_0.5_share_to_seals_SSP2_BiodivPol+ClimPol+NCPpol_LPJmL5.nc")
    p.magpie_scenario_coarse_state_paths['rcp45_ssp2'][2050]['SSP2_ClimPol_LPJmL5'] = os.path.join(p.input_dir, "SSP2_ClimPol_LPJmL5_2021-05-21_15.12.19", "cell.land_0.5_share_to_seals_SSP2_ClimPol_LPJmL5.nc")
    p.magpie_scenario_coarse_state_paths['rcp45_ssp2'][2050]['SSP2_NPI_base_LPJmL5'] = os.path.join(p.input_dir, "SSP2_NPI_base_LPJmL5_2021-05-21_15.05.56", "cell.land_0.5_share_to_seals_SSP2_NPI_base_LPJmL5.nc")

    p.gtap_scenario_coarse_state_paths = {}
    p.gtap_scenario_coarse_state_paths['rcp45_ssp2'] = {}
    p.gtap_scenario_coarse_state_paths['rcp45_ssp2'][2030] = {}
    p.gtap_scenario_coarse_state_paths['rcp45_ssp2'][2050] = {}
    p.gtap_scenario_coarse_state_paths['rcp45_ssp2'][2030]['BAU'] = p.luh_scenario_states_paths['rcp45_ssp2']
    p.gtap_scenario_coarse_state_paths['rcp45_ssp2'][2050]['BAU'] = p.luh_scenario_states_paths['rcp45_ssp2']

    p.luh_scenario_coarse_state_paths = {}
    p.luh_scenario_coarse_state_paths['rcp45_ssp2'] = {}
    p.luh_scenario_coarse_state_paths['rcp85_ssp5'] = {}
    p.luh_scenario_coarse_state_paths['rcp45_ssp2'][2030] = {}
    p.luh_scenario_coarse_state_paths['rcp45_ssp2'][2050] = {}
    p.luh_scenario_coarse_state_paths['rcp85_ssp5'][2030] = {}
    p.luh_scenario_coarse_state_paths['rcp85_ssp5'][2050] = {}
    p.luh_scenario_coarse_state_paths['rcp45_ssp2'][2030]['no_policy'] = p.luh_scenario_states_paths['rcp45_ssp2']
    p.luh_scenario_coarse_state_paths['rcp45_ssp2'][2050]['no_policy'] = p.luh_scenario_states_paths['rcp45_ssp2']
    p.luh_scenario_coarse_state_paths['rcp85_ssp5'][2030]['no_policy'] = p.luh_scenario_states_paths['rcp85_ssp5']
    p.luh_scenario_coarse_state_paths['rcp85_ssp5'][2050]['no_policy'] = p.luh_scenario_states_paths['rcp85_ssp5']


    # TODOO: figure out is_calibration_run vs p.calibrate. I need to specify the different run achetypes (just luh2, alternative to luh2, shapefile ON TOP of luh2)

    if p.is_magpie_run:
        p.scenario_coarse_state_paths = p.magpie_scenario_coarse_state_paths
    elif p.is_gtap1_run:
        p.scenario_coarse_state_paths = p.gtap_scenario_coarse_state_paths
    elif p.is_calibration_run:
        p.scenario_coarse_state_paths = p.luh_scenario_coarse_state_paths

    if p.test_mode:
        if p.is_magpie_run:
            p.policy_scenario_labels = p.magpie_test_policy_scenario_labels
        elif p.is_gtap1_run:
            p.policy_scenario_labels = p.gtap_bau_and_30_labels
        elif p.is_calibration_run:
            p.policy_scenario_labels = p.luh_labels
    else:
        if p.is_magpie_run:
            p.policy_scenario_labels = p.magpie_policy_scenario_labels
        elif p.is_gtap1_run:
            p.policy_scenario_labels = p.gtap_combined_policy_scenario_labels
        elif p.is_calibration_run:
            p.policy_scenario_labels = p.luh_labels

    if p.is_gtap1_run:
        # HACK, because I don't yet auto-generate the cmf files and other GTAP modelled inputs, and instead just take the files out of the zipfile Uris
        # provides, I still have to follow his naming scheme. This list comprehension converts a policy_scenario_label into a gtap1 or gtap2 label.
        p.gtap1_scenario_labels = [str(p.policy_base_year) + '_' + str(p.scenario_years[0])[2:] + '_' + i + '_noES' for i in p.policy_scenario_labels]
        p.gtap2_scenario_labels = [str(p.policy_base_year) + '_' + str(p.scenario_years[0])[2:] + '_' + i + '_allES' for i in p.policy_scenario_labels]



    # This is a zipfile I received from URIS that has all the packaged GTAP files ready to run. Extract these to a project dir.
    p.gtap_aez_invest_release_string = '04_20_2021_GTAP_AEZ_INVEST'
    p.gtap_aez_invest_zipfile_path = os.path.join(p.base_data_dir, 'gtap_aez_invest_releases', p.gtap_aez_invest_release_string + '.zip')
    p.gtap_aez_invest_code_dir = os.path.join(p.script_dir, 'gtap_aez', p.gtap_aez_invest_release_string)

    # Associate each luh, year, and policy scenario with a set of seals input parameters. This can be used if, for instance, the policy you
    # are analyzing involves focusing land-use change into certain types of gridcells.
    p.gtap_pretrained_coefficients_path_dict = {}
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'] = {}
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030] = {}
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2050] = {}
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['BAU'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2050]['BAU'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['PESGC'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['RnD'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_Land'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['PESLC'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_RnD'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_Land_PESGC'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_PESLC'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_RnD_PESGC'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_RnD_PESLC'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_RnD_20p'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_RnD_20p_PESGC'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_RnD_PESGC_30'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints_and_protected_areas.xlsx')
    p.gtap_pretrained_coefficients_path_dict['rcp45_ssp2'][2030]['SR_RnD_20p_PESGC_30'] = os.path.join(p.base_data_dir, 'default_inputs', 'gtap_trained_coefficients_combined_with_constraints_and_protected_areas.xlsx')

    p.magpie_pretrained_coefficients_path_dict = {}
    p.magpie_pretrained_coefficients_path_dict['baseline'] = {}
    p.magpie_pretrained_coefficients_path_dict['baseline'][2015] = {}
    p.magpie_pretrained_coefficients_path_dict['baseline'][2015]['baseline'] = {}
    p.magpie_pretrained_coefficients_path_dict['baseline'][2015]['baseline'] = os.path.join(p.base_data_dir, 'input', 'trained_coefficients_combined_with_constraints.xlsx')

    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'] = {}
    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'][2050] = {}
    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'][2050]['SSP2_BiodivPol_LPJmL5'] = os.path.join(p.base_data_dir, 'input', 'trained_coefficients_combined_with_constraints.xlsx')
    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'][2050]['SSP2_BiodivPol_ClimPol_LPJmL5'] = os.path.join(p.base_data_dir, 'input', 'trained_coefficients_combined_with_constraints.xlsx')
    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'][2050]['SSP2_BiodivPol_ClimPol_NCPpol_LPJmL5'] = os.path.join(p.base_data_dir, 'input', 'trained_coefficients_combined_with_constraints.xlsx')
    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'][2050]['SSP2_ClimPol_LPJmL5'] = os.path.join(p.base_data_dir, 'input', 'trained_coefficients_combined_with_constraints.xlsx')
    p.magpie_pretrained_coefficients_path_dict['rcp45_ssp2'][2050]['SSP2_NPI_base_LPJmL5'] = os.path.join(p.base_data_dir, 'input', 'trained_coefficients_combined_with_constraints.xlsx')

    # Define exact floating point representations of arcdegrees
    p.fine_resolution_degrees = hb.pyramid_compatible_resolutions[p.fine_resolution_arcseconds]
    p.coarse_resolution_degrees = hb.pyramid_compatible_resolutions[p.coarse_resolution_arcseconds]
    p.fine_resolution = p.fine_resolution_degrees
    p.coarse_resolution = p.coarse_resolution_degrees

    p.coarse_ha_per_cell_path = p.ha_per_cell_paths[p.coarse_resolution_arcseconds]
    p.coarse_match_path = p.coarse_ha_per_cell_path

    # A little awkward, but I used to use integers and list counting to keep track of the actual lulc class value. Now i'm making it expicit with dicts.
    p.class_indices_to_labels_correspondence = dict(zip(p.class_indices, p.class_labels))
    p.class_labels_to_indices_correspondence = dict(zip(p.class_labels, p.class_indices))

    p.calibrate = 1 # UNUSED EXCEPT IN Development features


    if p.is_gtap1_run:
        p.pretrained_coefficients_path_dict = p.gtap_pretrained_coefficients_path_dict
    elif p.is_magpie_run:
        p.pretrained_coefficients_path_dict = p.magpie_pretrained_coefficients_path_dict
    elif p.is_calibration_run:
        p.pretrained_coefficients_path_dict = 'use_generated' # TODOO Make this point somehow to the generated one.

    p.static_regressor_paths = {}
    p.static_regressor_paths['sand_percent'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'sand_percent.tif')
    p.static_regressor_paths['silt_percent'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'silt_percent.tif')
    p.static_regressor_paths['soil_bulk_density'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'soil_bulk_density.tif')
    p.static_regressor_paths['soil_cec'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'soil_cec.tif')
    p.static_regressor_paths['soil_organic_content'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'soil_organic_content.tif')
    p.static_regressor_paths['strict_pa'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'strict_pa.tif')
    p.static_regressor_paths['temperature_c'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'temperature_c.tif')
    p.static_regressor_paths['travel_time_to_market_mins'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'travel_time_to_market_mins.tif')
    p.static_regressor_paths['wetlands_binary'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'wetlands_binary.tif')
    p.static_regressor_paths['alt_m'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'alt_m.tif')
    p.static_regressor_paths['carbon_above_ground_mg_per_ha_global'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'carbon_above_ground_mg_per_ha_global.tif')
    p.static_regressor_paths['clay_percent'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'clay_percent.tif')
    p.static_regressor_paths['ph'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'ph.tif')
    p.static_regressor_paths['pop'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'pop.tif')
    p.static_regressor_paths['precip_mm'] = os.path.join(p.base_data_dir, 'seals', 'static_regressors', 'precip_mm.tif')

    p.global_esa_lulc_paths_by_year = {}
    p.global_esa_lulc_paths_by_year[p.training_start_year] = os.path.join(p.base_data_dir, "lulc", "esa", "lulc_esa_" + str(p.training_start_year) + '.tif' )
    p.global_esa_lulc_paths_by_year[p.base_year] = os.path.join(p.base_data_dir, "lulc", "esa", "lulc_esa_" + str(p.base_year) + '.tif' )

    p.training_start_year_lulc_path = p.global_esa_lulc_paths_by_year[p.training_start_year]
    p.training_end_year_lulc_path = p.global_esa_lulc_paths_by_year[p.base_year]
    p.base_year_lulc_path = p.global_esa_lulc_paths_by_year[p.base_year]

    p.global_esa_seals7_lulc_paths_by_year = {}

    # START HERE, finish adding seals7 base data to google cloud andor fixing the paths.
    p.global_esa_seals7_lulc_paths_by_year[p.training_start_year] = os.path.join(p.base_data_dir, "lulc", "esa", "seals7", "lulc_esa_seals7_" + str(p.training_start_year) + '.tif' )
    p.global_esa_seals7_lulc_paths_by_year[p.base_year] = os.path.join(p.base_data_dir, "lulc", "esa", "seals7", "lulc_esa_seals7_" + str(p.base_year) + '.tif' )

    p.training_start_year_seals7_lulc_path = p.global_esa_seals7_lulc_paths_by_year[p.training_start_year]
    p.training_end_year_seals7_lulc_path = p.global_esa_seals7_lulc_paths_by_year[p.base_year]
    p.base_year_seals7_lulc_path = p.global_esa_seals7_lulc_paths_by_year[p.base_year]

    p.base_year_lulc_path = p.global_esa_lulc_paths_by_year[p.base_year]

    # SEALS results will be tiled on top of output_base_map_path, filling in areas potentially outside of the zones run (e.g., filling in small islands that were skipped_
    p.base_year_simplified_lulc_path = p.global_esa_seals7_lulc_paths_by_year[p.base_year]
    p.lulc_training_start_year_10sec_path = p.global_esa_seals7_lulc_paths_by_year[p.base_year]
    p.output_base_map_path = p.base_year_simplified_lulc_path

    if p.test_mode:
        p.stitch_tiles_to_global_basemap = 0
        if p.is_gtap1_run:
            run_1deg_subset = 1
            run_5deg_subset = 0
            magpie_subset = 0
        elif p.is_magpie_run:
            run_1deg_subset = 0
            run_5deg_subset = 0
            magpie_subset = 1
        elif p.is_calibration_run:
            run_1deg_subset = 1
            run_5deg_subset = 0
            magpie_subset = 0

    else:
        p.stitch_tiles_to_global_basemap = 1
        run_1deg_subset = 0
        run_5deg_subset = 0
        magpie_subset = 0

    # If a a subset is defined, set its tiles here.
    if run_1deg_subset:
        p.processing_block_size = 1.0  # arcdegrees
        p.subset_of_blocks_to_run = [15526]  # mn. Has urban displaced by natural error.
        p.subset_of_blocks_to_run = [15708]  # slightly more representative  yet heterogenous zone.
        p.subset_of_blocks_to_run = [15526, 15708]  # combined. # DEFINED GLOBALLY
        p.subset_of_blocks_to_run = [0, 1, 2]  # Now defined wrt clipped aoi

        # p.subset_of_blocks_to_run = [
        #     15526, 15526 + 180 * 1, 15526 + 180 * 2,
        #     15527, 15527 + 180 * 1, 15527 + 180 * 2,
        #     15528, 15528 + 180 * 1, 15528 + 180 * 2,
        # ]  # 3x3 mn tiles
        p.force_to_global_bb = False
    elif run_5deg_subset:
        p.processing_block_size = 5.0  # arcdegrees
        p.subset_of_blocks_to_run = [476, 476 + 1 + (36 * 2), 476 + 3 + (36 * 4), 476 + 9 + (36 * 8), 476 + 1 + (36 * 25)]  # Montana
        p.force_to_global_bb = False
    elif magpie_subset:
        p.processing_block_size = 5.0  # arcdegrees
        # p.subset_of_blocks_to_run = [476, 476 + 1 + (36 * 2)]
        # p.subset_of_blocks_to_run = [476 + 9 + (36 * 8)]  # Montana
        p.subset_of_blocks_to_run = [476]
        p.force_to_global_bb = False
    else:
        p.subset_of_blocks_to_run = None
        p.processing_block_size = 5.0  # arcdegrees
        p.force_to_global_bb = True


    # Define which paths need to be in the base_data. Missing paths will be downloaded.
    p.required_base_data_paths = {}
    p.required_base_data_paths['global_countries_iso3_path'] = p.countries_iso3_path

    p.required_base_data_paths['coarse_state_paths'] = p.scenario_coarse_state_paths
    p.required_base_data_paths['pyramids'] = os.path.join(p.base_data_dir, 'pyramids', 'ha_per_cell_300sec.tif')
    p.required_base_data_paths['pyramids2'] = os.path.join(p.base_data_dir, 'pyramids', 'ha_per_cell_900sec.tif') # TODOO I made an idiotic choice to make this a nested dict which requires unique ids.... Eventually i have this flatten to a list but here i need to give it a temporary unique index.
    p.required_base_data_paths['pyramids3'] = os.path.join(p.base_data_dir, 'pyramids', 'ha_per_cell_10sec.tif') # TODOO I made an idiotic choice to make this a nested dict which requires unique ids.... Eventually i have this flatten to a list but here i need to give it a temporary unique index.
    p.required_base_data_paths['pyramids4'] = p.match_paths[3600.0] # TODOO I made an idiotic choice to make this a nested dict which requires unique ids.... Eventually i have this flatten to a list but here i need to give it a temporary unique index.
    p.required_base_data_paths['pretrained_coefficients_paths'] = p.pretrained_coefficients_path_dict

    p.required_base_data_paths['gtap37_aez18_path'] = p.gtap37_aez18_input_vector_path

    if p.is_calibration_run:
        p.required_base_data_paths['static_regressor_paths'] = p.static_regressor_paths

        p.required_base_data_paths['training_start_year_lulc_path'] = p.training_start_year_lulc_path
        p.required_base_data_paths['training_end_year_lulc_path'] = p.training_end_year_lulc_path
        p.required_base_data_paths['base_year_lulc_path'] = p.base_year_lulc_path

        p.required_base_data_paths['training_start_year_seals7_lulc_path'] = p.training_start_year_seals7_lulc_path
        p.required_base_data_paths['training_end_year_seals7_lulc_path'] = p.training_end_year_seals7_lulc_path
        p.required_base_data_paths['base_year_seals7_lulc_path'] = p.base_year_seals7_lulc_path


    # Sometimes runs fail mid run. This checks for that and picks up where there is a completed file for that zone.
    p.skip_created_downscaling_zones = 1

    ## ADD TASKS to project_flow task tree, then below set if they should run and/or be skipped if existing.
    p.base_data_task = p.add_task(seals_process_luh2.download_base_data, creates_dir=False,                                                                             run=1, skip_if_dir_exists=0)
    p.regressors_starting_values_task = p.add_task(seals_generate_base_data.regressors_starting_values,                                                                 run=1, skip_if_dir_exists=0)
    p.generated_data_task = p.add_task(seals_generate_base_data.generated_data,                                                                                         run=1, skip_if_dir_exists=0)
    p.aoi_vector_task = p.add_task(seals_generate_base_data.aoi_vector, parent=p.generated_data_task, creates_dir=False,                                                run=1, skip_if_dir_exists=0)
    p.lulc_clip_task = p.add_task(seals_generate_base_data.lulc_clip, parent=p.generated_data_task,creates_dir=False,                                                   run=1, skip_if_dir_exists=0)
    p.lulc_simplifications_task = p.add_task(seals_generate_base_data.lulc_simplifications, parent=p.generated_data_task, creates_dir=False,                            run=1, skip_if_dir_exists=0)
    p.lulc_binaries_task = p.add_task(seals_generate_base_data.lulc_binaries, parent=p.generated_data_task, creates_dir=False,                                          run=1, skip_if_dir_exists=0)
    p.generated_kernels_task = p.add_task(seals_generate_base_data.generated_kernels, parent=p.generated_data_task, creates_dir=False,                                  run=1, skip_if_dir_exists=0)
    p.lulc_convolutions_task = p.add_task(seals_generate_base_data.lulc_convolutions, parent=p.generated_data_task, creates_dir=False,                                  run=1, skip_if_dir_exists=0)
    p.local_data_regression_starting_values_task = p.add_task(seals_generate_base_data.local_data_regressors_starting_values,                                           run=1, skip_if_dir_exists=0)
    p.luh2_extraction_task = p.add_task(seals_process_luh2.luh2_extraction,                                                                                             run=1, skip_if_dir_exists=0)
    p.luh2_difference_from_base_year_task = p.add_task(seals_process_luh2.luh2_difference_from_base_year,                                                               run=1, skip_if_dir_exists=0)
    p.luh2_as_seals7_proportion_task = p.add_task(seals_process_luh2.luh2_as_seals7_proportion,                                                                         run=1, skip_if_dir_exists=0)
    p.seals7_difference_from_base_yea_task = p.add_task(seals_process_luh2.seals7_difference_from_base_year,                                                            run=1, skip_if_dir_exists=0)
    p.calibration_generated_inputs_task = p.add_task(seals_main.calibration_generated_inputs,                                                                           run=0, skip_if_dir_exists=0)
    p.calibration_task = p.add_iterator(seals_main.calibration, run_in_parallel=1,                                                                                      run=0, skip_if_dir_exists=0)
    p.calibration_prepare_lulc_task = p.add_task(seals_main.calibration_prepare_lulc, parent=p.calibration_task,                                                        run=0, skip_if_dir_exists=0)
    p.calibration_change_matrix_task = p.add_task(seals_main.calibration_change_matrix, parent=p.calibration_task,                                                      run=0, skip_if_dir_exists=0)
    p.calibration_zones_task = p.add_task(seals_main.calibration_zones, parent=p.calibration_task,                                                                      run=0, skip_if_dir_exists=0, logging_level=20)
    p.calibration_plots_task = p.add_task(seals_main.calibration_plots, parent=p.calibration_task,                                                                      run=0, skip_if_dir_exists=0)
    p.luh_allocations_task = p.add_iterator(seals_main.scenarios, run_in_parallel=0,                                                                                    run=0, skip_if_dir_exists=0)
    p.allocation_zones_task = p.add_iterator(seals_main.allocation_zones, parent=p.luh_allocations_task, run_in_parallel=1,                                             run=0, skip_if_dir_exists=0)
    p.prepare_lulc_task = p.add_task(seals_main.prepare_lulc, parent=p.allocation_zones_task,                                                                           run=0, skip_if_dir_exists=0)
    p.allocation_change_matrix_task = p.add_task(seals_main.allocation_change_matrix, parent=p.allocation_zones_task,                                                   run=0, skip_if_dir_exists=0)
    p.allocation_task = p.add_task(seals_main.allocation, parent=p.allocation_zones_task,                                                                               run=0, skip_if_dir_exists=0)
    p.allocation_exclusive_task = p.add_task(seals_main.allocation_exclusive, parent=p.allocation_zones_task,                                                           run=0, skip_if_dir_exists=0)
    p.allocation_from_change_matrix_task = p.add_task(seals_main.allocation_from_change_matrix, parent=p.allocation_zones_task,                                         run=0, skip_if_dir_exists=0)
    p.change_pngs_task = p.add_task(seals_main.change_pngs, parent=p.allocation_zones_task,                                                                             run=0, skip_if_dir_exists=0)
    p.change_exclusive_pngs_task = p.add_task(seals_main.change_exclusive_pngs, parent=p.allocation_zones_task,                                                         run=0, skip_if_dir_exists=0)
    p.change_from_change_matrix_pngs_task = p.add_task(seals_main.change_from_change_matrix_pngs, parent=p.allocation_zones_task,                                       run=0, skip_if_dir_exists=0)
    p.stitched_lulcs_task = p.add_task(seals_main.stitched_lulc_simplified_scenarios,                                                                                   run=1, skip_if_dir_exists=0)
    p.map_esa_simplified_back_to_esa_task = p.add_task(seals_main.stitched_lulc_esa_scenarios)



    p.execute()


