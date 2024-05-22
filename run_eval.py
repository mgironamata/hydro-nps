import torch
import config as C
from convcnp_architectures import ConvCNP
from architectures import UNet
import os
from datasets import HydroDataset, HydroTestDataset
from likelihoods import compute_logpdf
from run_training import unpickle_object


if __name__ == "__main__":

    q_mu = unpickle_object('pickled/q_mu.pkl')
    q_sigma = unpickle_object('pickled/q_sigma.pkl')
    # dist = unpickle_object('pickled/dist.pkl')\
    dist = "gaussian_fixed"
    
    df_train = unpickle_object('pickled/train.pkl')
    df_test_both = unpickle_object('pickled/test_both.pkl')
    df_test_catchment = unpickle_object('pickled/test_catchment.pkl')
    df_test_temporal = unpickle_object('pickled/test_temporal.pkl')
    df_att = unpickle_object('pickled/df_att.pkl')


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate ConvCNP
    model = ConvCNP(in_channels = len(C.context_channels),
                    #rho=SimpleConv(),
                    rho=UNet(),
                    #rho=DepthSepConv1d(in_channels=rho_in_channels, conv_channels=64, num_layers=7, kernel_size=15),
                    points_per_unit=64*8,
                    dynamic_embedding_dims=C.dynamic_embedding_dims,
                    static_embedding_dims=C.static_embedding_dims,
                    static_feature_embedding=C.static_feature_embedding,
                    static_embedding_in_channels=C.static_embedding_in_channels,
                    static_feature_missing_data=C.static_feature_missing_data,
                    static_embedding_location=C.static_embedding_location,
                    distribution=dist)

    # Assign model to device
    model.to(device)

    ROOT_PATH = r'_experiments\2024-05-21_16-29-19_test-noembeddings'
    model.load_state_dict(torch.load(os.path.join(ROOT_PATH, 'model_best.pth.tar')))

    print(model.num_params)

    # Instantiate data generator for testing.
    NUM_TEST_TASKS = 16 # 128
    gen_test = HydroDataset(
                                                dataframe=df_test_both,
                                                df_att = df_att,
                                                batch_size = 32,
                                                num_tasks = NUM_TEST_TASKS,
                                                channels_c = C.context_channels,
                                                channels_t = C.target_channels,
                                                channels_att = C.attributes,
                                                channels_t_val = C.target_val_channel,
                                                context_mask = C.context_mask,
                                                target_mask = C.target_mask,
                                                extrapolate = False,
                                                concat_static_features = C.concat_static_features,
                                                timeslice = C.timeslice,
                                                min_train_points = C.min_train_points,
                                                min_test_points = C.min_test_points,
                                                max_train_points = C.max_train_points,
                                                max_test_points = C.max_test_points,
                                                device = 'cpu'
                                                )
    

    # Instantiate data generator for testing.
NUM_TEST_TASKS = 10
gen_365 = data_hydro_2_extended.HydroGenerator(
                                            dataframe=df_test,
                                            df_att = df_att,
                                            batch_size = 16,
                                            num_tasks = NUM_TEST_TASKS,
                                            channels_c = C.context_channels,
                                            channels_t = C.target_channels,
                                            channels_att = C.attributes,
                                            channels_t_val = C.target_val_channel,
                                            context_mask = C.context_mask,
                                            target_mask = C.target_mask,
                                            concat_static_features = C.concat_static_features,
                                            extrapolate = True,
                                            timeslice = 90,
                                            min_train_points=89,
                                            min_test_points=1,
                                            max_train_points=89,
                                            max_test_points=1,
                                            device='cpu')

out = np.array([])
out_2 = np.array([])

# Compute average task log-likelihood.
# basins = df_test['hru08'].unique().tolist()
# print(len(basins))
# b=0
for basin in basins[:]:

    b+=1
    
    obs_basin = np.array([])
    mean_obs_basin = np.array([])
    sim_basin = np.array([])

    with torch.no_grad():    
        years = df_test['YR'][df_test['hru08']==basin].unique().tolist()
        years = sorted(years)[1:]
        
        for year in years:

            if year != 2000:
                continue

            start = time.time()
            
            task = gen_365.generate_test_task(year=year, basin=basin)

            # elapsed = time.time() - start
            # print(f'Generator: {elapsed}')
            # start = time.time()

            task = loaded_task(task=task, device='cuda')

            y_mean, y_std = model(task['x_context'], task['y_context'], 
                                      task['x_target'],task['y_att'], 
                                      task['feature'],task['m'],
                                      static_masking_rate=0,
                                      embedding=C.feature_embedding_flag)
            
            obj, y_mean, y_sigma = compute_logpdf(y_loc, y_scale, task, dist=dist, return_mu_and_sigma=True)
            
            #y_mu, y_sigma = rev_lognormal(to_numpy(y_mean), to_numpy(y_std))
            
            if dist == 'gamma':
                y_loc = y_mean
                y_scale = y_std
                g_mean = torch.distributions.gamma.Gamma(y_mean, y_std).mean
                g_var = torch.distributions.gamma.Gamma(y_mean, y_std).variance
                y_mean = g_mean
                y_std = torch.sqrt(g_var)

            obs = rev_transform(task['y_target'].flatten().cpu().numpy(), mu=q_mu, sigma=q_sigma)
            mean_obs = rev_transform(task['y_target_val'].flatten().cpu().numpy(), mu=q_mu, sigma=q_sigma)
            sim = rev_transform(y_mean.flatten().cpu().numpy(), mu=q_mu, sigma=q_sigma)

            obs_basin = np.concatenate((obs_basin,obs),axis=0)
            mean_obs_basin = np.concatenate((mean_obs_basin, mean_obs),axis=0)
            sim_basin = np.concatenate((sim_basin, sim), axis=0)

            # elapsed = time.time() - start
            # print(f'Model: {elapsed}')

            # print(year,basin)

            # plt.figure(figsize=(20,5))
            # plt.plot(obs,label='obs')
            # plt.plot(sim, label='sim')
            # plt.legend()
            # plt.show()

    try:
        nse = NSE.nse(obs=obs_basin
                ,mean_obs=mean_obs_basin
                ,sim=sim_basin)
        print(basin, nse)
    except:
        print(basin, "NSE can't be calculated")
        