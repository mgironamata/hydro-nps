
filepath = r"C:\Users\Sior AMD-4\Downloads\daymet_data_seed05.csv"

# basins_file = r'../../../ealstm_regional_modelling/ealstm_regional_modeling/data/basin_list.txt'
basins_file = r"C:\Users\Sior AMD-4\Downloads\basin_list.txt"

# path = '../../data/camels_processed/attibutes.csv'
path = r"C:\Users\Sior AMD-4\Downloads\attibutes.csv"


# Fields configuration
fields = [
    "OBS_RUN",
    "prcp(mm/day)",
    "dayl(s)",
    "srad(W/m2)",
    "swe(mm)",
    "tmax(C)",
    "tmin(C)",
    "vp(Pa)",
    "year"
]

# Attributes configuration
attributes = [
                #'gauge_id', 
                'p_mean', 
                'pet_mean', 
                'p_seasonality', 
                'frac_snow',
                'aridity', 
                'high_prec_freq', 
                'high_prec_dur', 
                #'high_prec_timing',
                'low_prec_freq', 
                'low_prec_dur', 
                #'low_prec_timing', 
                # 'geol_1st_class', 
                # 'glim_1st_class_frac', 
                # 'geol_2nd_class', 
                # 'glim_2nd_class_frac',
                'carbonate_rocks_frac', 
                #'geol_porostiy', 
                'geol_permeability', 
                #'q_mean',
                # 'runoff_ratio', 
                # 'slope_fdc', 
                # 'baseflow_index', 
                # 'stream_elas', 
                # 'q5',
                # 'q95', 
                # 'high_q_freq', 
                # 'high_q_dur', 
                # 'low_q_freq', 
                # 'low_q_dur',
                # 'zero_q_freq', 
                # 'hfd_mean', 
                # 'huc_02', 
                # 'gauge_name',
                'soil_depth_pelletier', 
                'soil_depth_statsgo', 
                'soil_porosity',
                'soil_conductivity', 
                'max_water_content', 
                'sand_frac', 
                'silt_frac', 
                'clay_frac', 
                #'water_frac', 
                # 'organic_frac', 
                # 'other_frac', 
                #'gauge_lat',
                #'gauge_lon', 
                'elev_mean', 
                'slope_mean', 
                #'area_gages2',
                'area_geospa_fabric', 
                'frac_forest', 
                'lai_max', 
                'lai_diff', 
                'gvf_max',
                'gvf_diff', 
                #'dom_land_cover_frac', 
                # 'dom_land_cover', 
                # 'root_depth_50',
                # 'root_depth_99', 
                # 'hru08'
            ]

s_date_tr = '1980-10-01'
e_date_tr = '1995-09-30'

s_date_te = '1995-10-01'
e_date_te = '2010-09-30'

context_channels = ['OBS_RUN',
                    #'dayl(s)',
                    'doy_cos',
                    'doy_sin',
                    'prcp(mm/day)', 
                    'srad(W/m2)',  
                    'tmax(C)',
                    'tmin(C)', 
                    'vp(Pa)',
                   ]

target_channels = context_channels
target_val_channel = ['OBS_RUN_mean']

context_mask = [0,1,1,1,1,1,1,1]
target_mask = [0,1,1,1,1,1,1,1]

list_to_drop = ['MNTH','DY','hru02','hru04','RAIM','TAIR','PET','ET','SWE','swe(mm)','PRCP','seed','id_lag','HR']

observed_at_target_flag = True
feature_embedding_flag = True
feature_embedding_key_flag = True
extrapolate_flag = False

timeslice = 200

min_train_points = 100
max_train_points = 120
min_test_points = 40
max_test_points = 60

dynamic_embedding_dims = 10
static_embedding_dims = 5

concat_static_features = False
static_feature_embedding = True

static_embedding_location = "after_encoder" 
static_feature_missing_data = False # True
static_masking_rate = 0 # 0.25

encoder_out_channels = 8

rho_in_channels = encoder_out_channels if static_embedding_location != "after_encoder" else 8 + static_embedding_dims
static_embedding_in_channels = 2 if static_feature_missing_data else len(attributes)