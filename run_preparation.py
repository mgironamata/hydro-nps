import os
from numpy import loadtxt
import pandas as pd
from transformations import standardise, log_transform, boxcox_transform, scaling, scale_dataframe
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
    
    df_raw = df_raw[df_raw['QObs(mm/d)'] >= 0]
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
        target_fields = ["PRCP(mm/day)", "QObs(mm/d)"]
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
        Q_shift = abs(df['QObs(mm/d)'].min()) + 1
        df['QObs(mm/d)'] = df['QObs(mm/d)'] + Q_shift
        print(f'Shifted by {Q_shift}')
    else:
        print('No shift')
    
    df['QObs(mm/d)_mean'] = df.groupby('basin')['QObs(mm/d)'].transform(lambda x: x.mean())
    return df, q_mu, q_sigma

def split_dataframes(df, C, tr, te):
    train = df[(df.index >= C.s_date_tr) & (df.index <= C.e_date_tr) & (df['basin'].isin(tr))].copy()
    test_both = df[(df.index >= C.s_date_te) & (df.index <= C.e_date_te) & (df['basin'].isin(te))].copy()
    test_catchment = df[(df.index >= C.s_date_tr) & (df.index <= C.e_date_tr) & (df['basin'].isin(te))].copy()
    test_temporal = df[(df.index >= C.s_date_te) & (df.index <= C.e_date_te) & (df['basin'].isin(tr))].copy()
    
    for df in [train, test_both, test_catchment, test_temporal]:
        # df.drop(C.list_to_drop, axis=1, inplace=True)
        df.reset_index(drop=True, inplace=True)
    
    return train, test_both, test_catchment, test_temporal

def save_data(train, test_both, test_catchment, test_temporal, att, dist, basin_means, basin_stds, stats_dict):
    if not os.path.exists('pickled'):
        os.mkdir('pickled')
    
    train.to_pickle('pickled/train.pkl')
    test_both.to_pickle('pickled/test_both.pkl')
    test_catchment.to_pickle('pickled/test_catchment.pkl')
    test_temporal.to_pickle('pickled/test_temporal.pkl')
    att.to_pickle('pickled/df_att.pkl')
    
    with open('pickled/dist.pkl', 'wb') as f:
        pickle.dump(dist, f)

    with open('pickled/basin_means.pkl', 'wb') as f:
        pickle.dump(basin_means, f)
    with open('pickled/basin_stds.pkl', 'wb') as f:
        pickle.dump(basin_stds, f)
    with open('pickled/stats_dict.pkl', 'wb') as f:
        pickle.dump(stats_dict, f)

def main():
    import numpy as np

    df = load_data(r'C:\Users\Sior AMD-4\repos\hydro-nps\data\incoming\nldas_data.pkl') # load dataframe
    selected_basins = load_selected_basins(C.basins_file) # load selected basins
    selected_basins = [str(b).zfill(8) for b in selected_basins] # format basins to 8 digits
    
    df_att = process_attributes(C.path) # process attributes

    df['QObs(mm/d)_mean'] = df.groupby('basin')['QObs(mm/d)'].transform('mean')

    dist = 'gaussian' # define likelihood
    # df = apply_transformations(df, df, log_transformation=C.transform, boxcox_transformation=False) # apply transformations

    # create a dict with means and stds of each basin in df_train
    basin_means = df.groupby('basin')['QObs(mm/d)_mean'].mean()
    basin_stds = df.groupby('basin')['QObs(mm/d)'].std()
    basin_means = basin_means.to_dict()
    basin_stds = basin_stds.to_dict()

    # from selected basins, randomly take 1/12th of the basins for testing
    np.random.seed(0)
    np.random.shuffle(selected_basins)
    te = selected_basins[:len(selected_basins)//12]
    tr = selected_basins[len(selected_basins)//12:]
    
    df_train, df_test_both, df_test_catchment, df_test_temporal = split_dataframes(df, C, tr, te) # split dataframes
    
    # calculate mean of QObs by basin and add as new column
    # df_train['QObs(mm/d)_mean'] = df_train.groupby('basin')['QObs(mm/d)'].transform('mean')

    # df_train, q_mu, q_sigma = scale_and_shift_data(df, dist, C)

    df_train, stats_dict = scale_dataframe(df_train, C.fields[:-1])
    df_test_both, _ = scale_dataframe(df_test_both, C.fields[:-1], stats_dict)
    df_test_catchment, _ = scale_dataframe(df_test_catchment, C.fields[:-1], stats_dict)
    df_test_temporal, _ = scale_dataframe(df_test_temporal, C.fields[:-1], stats_dict)

    for e in [df_test_both, df_test_catchment, df_test_temporal]:
        e['DOY'] = e.apply(lambda x: pd.Timestamp(year=x['Year'], month=x['Mnth'], day=x['Day']).dayofyear, axis=1)
        print('done')
                                       
    save_data(df_train, df_test_both, df_test_catchment, df_test_temporal, df_att, dist, basin_means, basin_stds, stats_dict)

    print('Data saved')

if __name__ == "__main__":
    main()
