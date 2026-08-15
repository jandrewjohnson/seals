"""ASPIRATIONAL / NOT WIRED IN.

This module is a planned, future model-spec-driven parsing layer (e.g. parse a
markdown model spec into project attributes). It is NOT part of the live code
path: it is not exported from seals/__init__.py and the only thing that imports
it is seals_tests/manual_t_api_utils.py.

Production parsing currently lives in seals_utils.py (assign_df_row_to_object_attributes,
set_derived_attributes, generate_scenarios_csv_and_put_in_input_dir). The copies
here are an intentional parking spot for the future model_spec design, not a
maintained second implementation -- do not treat them as a variant to reconcile,
and do not import from here in production code until this is actually built out
(parse_model_spec_md is still a stub).
"""

import os
import hazelbean as hb
import pandas as pd


def assign_defaults_from_model_spec(input_object, model_spec_dict):
    # Helper function that takes an input object, like a ProjectFlow p variable
    # and a dictionary of default values. If the attribute is not already set, it will
    # set the attribute to the default value.    
    for k, v in model_spec_dict.items():
        if not hasattr(input_object, k):
            setattr(input_object, k, v)
            
def parse_input_api_md(input_md_path):
    lines = hb.read_path_as_list(input_md_path)
    d = {}
    level = 0
    for line in lines:
        line = line.replace('\n', '')
        if len(line.strip()) == 0:
            continue
        
        if line.startswith('# '):
            level = 1
            k1 = line.replace('#', '').strip()
            d[k1] = {}
            
        elif line.startswith('## '):
            level = 2
            k2 = line.replace('##', '').strip()
            d[k1][k2] = {}
            
        # Test if has a key
        elif line.startswith('- '):
            if len(line.replace('- ', '').strip().split(':')[0].split(' ')) == 1:        

                # Then its a key value
                key = line.replace('-', '').strip().split(':')[0].strip()
                value = line.replace('-', '').strip().split(':')[1].strip()
                
                if key == "Examples":
                    value = [i.strip() for i in value.split(',')]
                    
                

                if level == 1:
                    
                    if 'Type' in d[k1].keys():
                        current_type = d[k1]['Type']
                        if key == 'Default':
                            if current_type == 'int':
                                value = int(value)
                            elif current_type == 'float':
                                value = float(value)
                            elif current_type == 'list':
                                value = [i.strip() for i in value.split(',')]
                            elif current_type == 'str':
                                value = str(value)
                            elif current_type == 'path':
                                value = str(value)
                            elif current_type == 'bool':
                                value = bool(value)
                            else:
                                value = value
                    d[k1][key] = value
                    
                elif level == 2:
                    
                    if 'Type' in d[k1][k2].keys():
                        current_type = d[k1][k2]['Type']
                        if key == 'Default':
                            if current_type == 'int':
                                value = int(value)
                            elif current_type == 'float':
                                value = float(value)
                            elif current_type == 'list':
                                value = [i.strip() for i in value.split(',')]
                            elif current_type == 'space delimited list of integers':
                                value = [int(i.strip()) for i in value.split(' ')]
                            elif current_type == 'str':
                                value = str(value)
                            elif current_type == 'path':
                                value = str(value)
                            elif current_type == 'bool':
                                value = bool(value)
                            else:
                                value = value
                    d[k1][k2][key] = value
                    
                            
    hb.print_iterable(d)
    
    return d
            
            
def api_dict_to_df(api_dict):
    # Flatten the dict so only ## level 2 are there
    flat_dict = {}
    for k1, v1 in api_dict.items():
        for k2, v2 in v1.items():
            flat_dict[k2] = v2
            
    parsed_dict = {}
    for k, v in flat_dict.items():  
        if type(v) is dict:
            parsed_dict[k] = v['Default']     
    
    df = pd.DataFrame(parsed_dict)
    
          
# Make this follow model spec for all pre-processing and validation
def assign_df_row_to_object_attributes(input_object, input_row):
    # Thin shim. The per-row value grammar is owned by hazelbean now (shared across
    # seals, gtappy and the vertical parser). See hb.assign_to_object. NOTE: this
    # module's parser is not referenced by any caller; seals_utils' version is the
    # production one. Kept as a shim for consistency.
    hb.assign_row_to_object_attributes(input_object, input_row)

    model_spec = {}
    model_spec['regional_projections_input_path'] = ''
    assign_defaults_from_model_spec(input_object, model_spec)


# Old approach. Literally writes a csv.
def generate_scenarios_csv_and_put_in_input_dir(p):
    # In the event that the scenarios csv was not set, this currently wouldn't yet be a scenarios_df
    # Yet, I still want to be able to iterate over it. So thus, I need to GENERATE the scenarios_df from the project_flow
    # attributes
    list_of_attributes_to_write = [
        	
        'scenario_label',	
        'scenario_type',	
        'aoi',	
        'exogenous_label',	
        'climate_label',	
        'model_label',	
        'counterfactual_label',	
        'years',	
        'baseline_reference_label',	
        'base_years',	
        'key_base_year',
        'comparison_counterfactual_labels',	
        'time_dim_adjustment',	
        'coarse_projections_input_path',	
        'lulc_src_label',	
        'lulc_simplification_label',	
        'lulc_correspondence_path',	
        'coarse_src_label',	
        'coarse_simplification_label',	
        'coarse_correspondence_path',	
        'lc_class_varname',	
        'dimensions',	
        'calibration_parameters_source',	
        'base_year_lulc_path',
    ]


    data = {i: [] for i in list_of_attributes_to_write}

    # Add a baseline row. For the next scenario specific row it will actually take from the p attributes
    # however for the baseline row we have to override a few things (like setting years to the base years)
    data['scenario_label'].append('baseline_' + p.model_label)
    data['scenario_type'].append('baseline')
    data['aoi'].append(p.aoi)
    data['exogenous_label'].append('baseline')
    data['climate_label'].append(''	)
    data['model_label'].append(p.model_label)
    data['counterfactual_label'].append('')
    data['years'].append(' '.join([str(int(i)) for i in p.base_years]))
    data['baseline_reference_label'].append('' )
    data['base_years'].append(' '.join([str(int(i)) for i in p.base_years]))
    data['key_base_year'].append(p.key_base_year)
    data['comparison_counterfactual_labels'].append('')
    data['time_dim_adjustment'].append('add2015')
    data['coarse_projections_input_path'].append(p.coarse_projections_input_path)
    data['lulc_src_label'].append('esa')
    data['lulc_simplification_label'].append('seals7')
    data['lulc_correspondence_path'].append('seals/default_inputs/esa_seals7_correspondence.csv')
    data['coarse_src_label'].append('luh2-14')
    data['coarse_simplification_label'].append('seals7')
    data['coarse_correspondence_path'].append('seals/default_inputs/luh2-14_seals7_correspondence.csv')
    data['lc_class_varname'].append('all_variables')
    data['dimensions'].append('time')
    data['calibration_parameters_source'].append('seals/default_inputs/default_global_coefficients.csv')
    data['base_year_lulc_path'].append(p.base_year_lulc_path)


    # Add non baseline. Now that the baseline was added, we can now just iterate over the existing attributes
    for i in list_of_attributes_to_write:
        current_attribute = getattr(p, i)
        if type(current_attribute) is str:
            if current_attribute.startswith('['):
                current_attribute = ' '.join(list(current_attribute))
            elif os.path.isabs(current_attribute):
                current_attribute = hb.path_split_at_dir(current_attribute, os.path.split(p.base_data_dir)[1])[2].replace('\\\\', '\\').replace('\\', '/') # NOTE Awkward hack of assuming there is only 1 dir with the same name as base_data_dir

        elif type(current_attribute) is list:
            current_attribute = ' '.join([str(i) for i in current_attribute])

        data[i].append(current_attribute)

    p.scenarios_df = pd.DataFrame(data=data, columns=list_of_attributes_to_write)

    hb.create_directories(p.scenario_definitions_path)
    p.scenarios_df.to_csv(p.scenario_definitions_path, index=False)



# Not SURE if this is modelspec, but maybe it would allow custom functions after an input?
def set_derived_attributes(p):
    
    # Resolutions come from the fine and coarse maps
    p.fine_resolution = hb.get_cell_size_from_path(p.base_year_lulc_path)
    p.fine_resolution_arcseconds = hb.pyramid_compatible_resolution_to_arcseconds[p.fine_resolution]
    
    if hb.is_path_gdal_readable(p.coarse_projections_input_path):
        p.coarse_resolution = hb.get_cell_size_from_path(p.coarse_projections_input_path)
        p.coarse_resolution_arcseconds = hb.pyramid_compatible_resolution_to_arcseconds[p.coarse_resolution] 
    else:
        p.coarse_resolution_arcseconds = float(p.coarse_projections_input_path)
        p.coarse_resolution = hb.pyramid_compatible_resolutions[p.coarse_resolution_arcseconds]    
     
    p.fine_resolution_degrees = hb.pyramid_compatible_resolutions[p.fine_resolution_arcseconds]
    p.coarse_resolution_degrees = hb.pyramid_compatible_resolutions[p.coarse_resolution_arcseconds]
    p.fine_resolution = p.fine_resolution_degrees
    p.coarse_resolution = p.coarse_resolution_degrees
    
       
    
    # Set the derived-attributes too whenever the core attributes are set
    p.lulc_correspondence_path = p.get_path(p.lulc_correspondence_path)
    # p.lulc_correspondence_path = hb.get_first_extant_path(p.lulc_correspondence_path, [p.input_dir, p.base_data_dir])
    p.lulc_correspondence_dict = hb.utils.get_reclassification_dict_from_df(p.lulc_correspondence_path, 'src_id', 'dst_id', 'src_label', 'dst_label')
    
    
    p.coarse_correspondence_path = p.get_path(p.coarse_correspondence_path)
    # p.coarse_correspondence_path = hb.get_first_extant_path(p.coarse_correspondence_path, [p.input_dir, p.base_data_dir])
    p.coarse_correspondence_dict = hb.utils.get_reclassification_dict_from_df(p.coarse_correspondence_path, 'src_id', 'dst_id', 'src_label', 'dst_label')

    ## Load the indices and labels from the COARSE correspondence. We need this go get waht calsses are changing.
    if p.coarse_correspondence_dict is not None:

        coarse_dst_ids = p.coarse_correspondence_dict['dst_ids']
        p.coarse_correspondence_class_indices = sorted([int(i) for i in coarse_dst_ids])

        coarse_dst_ids_to_labels = p.coarse_correspondence_dict['dst_ids_to_labels']
        p.coarse_correspondence_class_labels = [str(coarse_dst_ids_to_labels[i]) for i in p.coarse_correspondence_class_indices]


    if p.lulc_correspondence_dict is not None:
        lulc_dst_ids = p.lulc_correspondence_dict['dst_ids']
        p.lulc_correspondence_class_indices = sorted([int(i) for i in lulc_dst_ids])

        lulc_dst_ids_to_labels = p.lulc_correspondence_dict['dst_ids_to_labels']
        p.lulc_correspondence_class_labels = [str(lulc_dst_ids_to_labels[i]) for i in p.lulc_correspondence_class_indices]

    # Define the nonchanging class indices as anything in the lulc simplification classes that is not in the coarse simplification classes
    p.nonchanging_class_indices = [int(i) for i in p.lulc_correspondence_class_indices if i not in p.coarse_correspondence_class_indices] # These are the indices of classes THAT CANNOT EXPAND/CONTRACT

    # Classes a project wants protected beyond the non-changing ones: classes nothing may
    # expand onto even though they do have a coarse demand. Empty by default. The no-demand
    # classes are derived above; anything further is a scenario statement rather than a
    # property of the correspondences, so it has to be named. Urban is the standing example,
    # since it expands but built land is not un-built.

    p.changing_coarse_correspondence_class_indices = [int(i) for i in p.coarse_correspondence_class_indices if i not in p.nonchanging_class_indices] # These are the indices of classes THAT CAN EXPAND/CONTRACT
    p.changing_coarse_correspondence_class_labels = [str(p.coarse_correspondence_dict['dst_ids_to_labels'][i]) for i in p.changing_coarse_correspondence_class_indices if i not in p.nonchanging_class_indices]
    p.changing_lulc_correspondence_class_indices = [int(i) for i in p.lulc_correspondence_class_indices if i not in p.nonchanging_class_indices] # These are the indices of classes THAT CAN EXPAND/CONTRACT
    p.changing_lulc_correspondence_class_labels = [str(p.lulc_correspondence_dict['dst_ids_to_labels'][i]) for i in p.changing_lulc_correspondence_class_indices if i not in p.nonchanging_class_indices]
       
    # From the changing/nonchanging class sets as defined in the lulc correspondence AND the coarse correspondence.
    p.changing_class_indices = p.changing_coarse_correspondence_class_indices + [i for i in p.changing_lulc_correspondence_class_indices if i not in p.changing_coarse_correspondence_class_indices] 
    p.changing_class_labels = p.changing_coarse_correspondence_class_labels + [i for i in p.changing_lulc_correspondence_class_labels if i not in p.changing_coarse_correspondence_class_labels]
    
    p.all_class_indices = p.coarse_correspondence_class_indices + [i for i in p.lulc_correspondence_class_indices if i not in p.coarse_correspondence_class_indices] 
    p.all_class_labels = p.coarse_correspondence_class_labels + [i for i in p.lulc_correspondence_class_labels if i not in p.coarse_correspondence_class_labels]
    p.class_labels = p.all_class_labels
    
    # Check if processing_resolution exists
    if not hasattr(p, 'processing_resolution'):
        p.processing_resolution = 1.0
    
    p.processing_resolution_arcseconds = p.processing_resolution * 3600.0 # MUST BE FLOAT
    
    def parse_model_spec_md(input_md_path):
        out = {}
        return out