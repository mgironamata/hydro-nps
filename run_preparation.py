import os
from numpy import loadtxt
import pandas as pd
from transformations import standardise, log_transform, boxcox_transform, scaling
import config as C
import pickle

def load_data(path):
    df = pd.read_pickle(path)
    return df

def load_selected_basins(basins_file):
    return loadtxt(basins_file, comments="#", delimiter=",", unpack=False, dtype="int")

def preprocess_data(df_raw, selected_basins):
    df_raw = df_raw[df_raw['basin'].isin(selected_basins)]
    print("Length of dataframe for selected basin: ", len(df_raw))
    
    df_raw.drop(['MOD_RUN'], axis=1, inplace=True)
    df_raw.drop_duplicates(inplace=True)
    print("Length of dataframe after dropping duplicates: ", len(df_raw))
    
    df_raw = df_raw[df_raw['OBS_RUN'] >= 0]
    print("Length of dataframe after filtering out error values: ", len(df_raw))
    
    df_raw['year'] = df_raw['YR']
    return df_raw

def process_attributes(path):
    df_att = pd.read_csv(path)
    numeric_attributes = df_att.select_dtypes('float64').columns.tolist()
    
    for att in numeric_attributes:
        df_att[att] = standardise(df_att[att])
    
    return df_att

def apply_transformations(df_raw, df, log_transformation=False, boxcox_transformation=False):
    if log_transformation:
        target_fields = ['prcp(mm/day)']
        for f in target_fields:
            df[f] = log_transform(df_raw[f], 1)
            print(f"log of {f}")
    
    if boxcox_transformation:
        target_fields = ['prcp(mm/day)']
        for f in target_fields:
            df[f], lambda_val = boxcox_transform(df_raw[f], 1e-6)
            
            for ch in ['(', ')', '/']:
                f = f.replace(ch, "_")
            
            exec(f"lambda_{f} = {lambda_val}") # save lambda value to a variable
            print(f"lambda_{f} = ", lambda_val)
    
    return df

def scale_and_shift_data(df, dist, C):
    df, q_mu, q_sigma = scaling(df, C.fields, method='std')
    
    if dist == 'gamma':
        Q_shift = abs(df['OBS_RUN'].min()) + 1
        df['OBS_RUN'] = df['OBS_RUN'] + Q_shift
        print(f'Shifted by {Q_shift}')
    else:
        print('No shift')
    
    df['OBS_RUN_mean'] = df.groupby('hru08')['OBS_RUN'].transform(lambda x: x.mean())
    return df, q_mu, q_sigma

def split_dataframes(df, C, tr, te):
    df_train = df[(df['DATE'] >= C.s_date_tr) & (df['DATE'] <= C.e_date_tr) & (df['basin'].isin(tr))].copy()
    df_test_both = df[(df['DATE'] >= C.s_date_te) & (df['DATE'] <= C.e_date_te) & (df['basin'].isin(te))].copy()
    df_test_catchment = df[(df['DATE'] >= C.s_date_tr) & (df['DATE'] <= C.e_date_tr) & (df['basin'].isin(te))].copy()
    df_test_temporal = df[(df['DATE'] >= C.s_date_te) & (df['DATE'] <= C.e_date_te) & (df['basin'].isin(tr))].copy()
    
    for df in [df_train, df_test_both, df_test_catchment, df_test_temporal]:
        df.drop(C.list_to_drop, axis=1, inplace=True)
        df.reset_index(drop=True, inplace=True)
    
    return df_train, df_test_both, df_test_catchment, df_test_temporal

def save_data(df_train, df_test_both, df_test_catchment, df_test_temporal, df_att, q_mu, q_sigma, dist):
    if not os.path.exists('pickled'):
        os.mkdir('pickled')
    
    df_train.to_pickle('pickled/train.pkl')
    df_test_both.to_pickle('pickled/test_both.pkl')
    df_test_catchment.to_pickle('pickled/test_catchment.pkl')
    df_test_temporal.to_pickle('pickled/test_temporal.pkl')
    df_att.to_pickle('pickled/df_att.pkl')
    
    with open('pickled/q_mu.pkl', 'wb') as f:
        pickle.dump(q_mu, f)
    with open('pickled/q_sigma.pkl', 'wb') as f:
        pickle.dump(q_sigma, f)
    with open('pickled/dist.pkl', 'wb') as f:
        pickle.dump(dist, f)

def main():
    df_raw = load_data('df_raw.pkl')
    selected_basins = load_selected_basins(C.basins_file)
    df_raw = preprocess_data(df_raw, selected_basins)
    df_att = process_attributes(C.path)
    
    df = df_raw.copy()
    dist = 'gaussian'
    
    df = apply_transformations(df_raw, df, log_transformation=False, boxcox_transformation=False)
    df, q_mu, q_sigma = scale_and_shift_data(df, dist, C)
    
    # tr, te = [], []  # Example lists, replace with actual lists of training and testing basins
    tr = te = selected_basins
    
    df_train, df_test_both, df_test_catchment, df_test_temporal = split_dataframes(df, C, tr, te)
    save_data(df_train, df_test_both, df_test_catchment, df_test_temporal, df_att, q_mu, q_sigma, dist)

if __name__ == "__main__":
    main()
