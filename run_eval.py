import torch
import config as C
from convcnp_architectures import ConvCNP
from architectures import UNet
import os
from temp_file import HydroDataset
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


    ROOT_PATH = r'_experiments\2024-05-17_09-24-33_test-gaussianfixed-365'
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