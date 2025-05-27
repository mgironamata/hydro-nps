filepath = r"C:\Users\Sior AMD-4\Downloads\daymet_data_seed05.csv"

# basins_file = r'../../../ealstm_regional_modelling/ealstm_regional_modeling/data/basin_list.txt'
basins_file = r"C:\Users\Sior AMD-4\Downloads\basin_list.txt"

# path = '../../data/camels_processed/attibutes.csv'
path = r"C:\Users\Sior AMD-4\Downloads\attibutes.csv"

# 'PRCP(mm/day)', 
# 'Dayl(s)', 
# 'SRAD(W/m2)',
# 'SWE(mm)', 
# 'Tmax(C)', 
# 'Tmin(C)', 
# 'Vp(Pa)', 
# 'QObs(mm/d)',
# 'Year'

# Fields configuration
fields = [
    "QObs(mm/d)",
    'PRCP(mm/day)', 
    'Dayl(s)', 
    'SRAD(W/m2)',
    'SWE(mm)', 
    'Tmax(C)', 
    'Tmin(C)', 
    'Vp(Pa)', 
    'Year'
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

context_channels = ['QObs(mm/d)',
                    #'dayl(s)',
                    # 'doy_cos',
                    # 'doy_sin',
                    'PRCP(mm/day)', 
                    # 'Dayl(s)', 
                    'SRAD(W/m2)',
                    'Tmax(C)', 
                    'Tmin(C)', 
                    'Vp(Pa)',
                   ]

target_channels = context_channels
target_val_channel = ['QObs(mm/d)_mean']

context_mask = [0,1,1,1,1,1,1,1]
target_mask = [0,1,1,1,1,1,1,1]

list_to_drop = ['MNTH','DY','hru02','hru04','RAIM','TAIR','PET','ET','SWE','swe(mm)','PRCP','seed','id_lag','HR']

observed_at_target_flag = False
feature_embedding_flag = True
feature_embedding_key_flag = True
extrapolate_flag = True

timeslice = 365

min_train_points = 355
max_train_points = 355
min_test_points = 7
max_test_points = 10

dynamic_embedding_dims = 8
static_embedding_dims = 8

concat_static_features = False
static_feature_embedding = True

static_embedding_location = "after_encoder" # "after_encoder" or "after_rho"
static_feature_missing_data = False # True
static_masking_rate = 0 # 0.25

points_per_unit = timeslice # 64*8

encoder_out_channels = 16

rho_in_channels = encoder_out_channels if static_embedding_location != "after_encoder" else 8 + static_embedding_dims
static_embedding_in_channels = 2 if static_feature_missing_data else len(attributes)

transform = False # "LOG"
scaling = "STANDARD"