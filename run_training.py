import config as C
from architectures import SimpleConv, UNet, DepthSepConv1d
from convcnp_architectures import ConvCNP
import pickle
import torch
# import data_hydro_2_extended
from experiment import WorkingDirectory, generate_root, save_checkpoint, RunningAverage
from transformations import rev_transform, rev_transform_tensor, rev_standardise
import NSE
from plot_utils import plot_training_loss, plot_model_task
from task_utils import task_to_device, loaded_task
import time
from IPython import display
import os
from torch.utils.data import DataLoader
# from data_loader_pytorch import HydroDataset
from datasets import HydroDataset

from likelihoods import compute_logpdf

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Disable interactive mode


torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_batch_size(x):
    if x.__class__ == 'torch.utils.data.dataloader.DataLoader':
        return x.dataset.gen.batch_size
    else:
        return x.batch_size
    
def unpickle_object(file_path):
    with open(file_path, 'rb') as file:
        obj = pickle.load(file)
    return obj

def train(data, model, opt, dist='gaussian'):
    
    ravg = RunningAverage()
    ravg_nse = RunningAverage() 
    model.train()
    task_obj_list = []
    
    for step, task in enumerate(data):
        
        task = loaded_task(task)
        
        y_loc, y_scale = model(x = task['x_context'],
                               y = task['y_context'], 
                               x_out = task['x_target'], 
                               y_att = task['y_att'], 
                               # f = task['feature'], 
                               # m = task['m'], 
                               static_masking_rate=static_masking_rate, 
                            )
        
        obj, y_mean, y_sigma = compute_logpdf(y_loc, y_scale, task, dist=dist, return_mu_and_sigma=True)
        
        obj.backward()
        opt.step()
        opt.zero_grad()
            
        obj_nse = NSE.nse_tensor(obs = rev_transform_tensor(task['y_target']),
                                mean_obs = rev_transform_tensor(task['y_target_val']),
                                sim = rev_transform_tensor(y_mean))
        
        task_obj_list.append(obj.item())
        ravg.update(obj.item(), data.batch_size)
        ravg_nse.update(obj_nse.item(), data.batch_size)
        
        if step % 25 == 0:
            print("step %s -- avg training loss is %.3f" % (step, ravg.avg)) 
            
    plt.plot(task_obj_list)
    plt.show
        
    return ravg.avg, ravg_nse.avg

def test(gen_test,model,dist='gaussian',fig_flag=False):
    # Compute average task log-likelihood.
    ravg = RunningAverage()
    ravg_nse = RunningAverage()
    
    model.eval()
    start = time.time()
    
    with torch.no_grad():
        for _, task in enumerate(gen_test):
            # torch.cuda.empty_cache()
            task = loaded_task(task)
            
            y_loc, y_scale = model(x = task['x_context'],
                                   y = task['y_context'], 
                                   x_out = task['x_target'], 
                                   y_att = task['y_att'], 
                                   #f = task['feature'], 
                                   #m = task['m'], 
                                   static_masking_rate = static_masking_rate,
                                   )        
            
            obj, y_mean, y_sigma = compute_logpdf(y_loc, y_scale, task, dist=dist, return_mu_and_sigma=True)
            
            batch_size = get_batch_size(gen_test)
            
            obj_nse = NSE.nse_tensor(obs=rev_transform_tensor(task['y_target'],mu=q_mu,sigma=q_sigma, scaling='STANRDARD'),
                                    mean_obs=rev_transform_tensor(task['y_target_val'],mu=q_mu,sigma=q_sigma, scaling='STANRDARD'),
                                    sim=rev_transform_tensor(y_mean,mu=q_mu,sigma=q_sigma, scaling='STANRDARD'))       
            
            ravg.update(obj.item(), batch_size)
            ravg_nse.update(obj_nse.item(), batch_size)

    if fig_flag:
        fig = plt.figure(figsize=(24, 15))
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plot_model_task(model, task, C.timeslice, idx=i, legend=i==2, dist=dist)
        plt.show()

        elapsed = time.time() - start        
        print('Test NLL: %.3f -- NSE: %.3f -- time: %.3f' % (ravg.avg, ravg_nse.avg, elapsed))
    
    return ravg.avg

if __name__ == "__main__":

    q_mu = unpickle_object('pickled/q_mu.pkl')
    q_sigma = unpickle_object('pickled/q_sigma.pkl')
    # dist = unpickle_object('pickled/dist.pkl')\
    dist = "gaussian"
    
    df_train = unpickle_object('pickled/train.pkl')
    df_test_both = unpickle_object('pickled/test_both.pkl')
    df_test_catchment = unpickle_object('pickled/test_catchment.pkl')
    df_test_temporal = unpickle_object('pickled/test_temporal.pkl')
    df_att = unpickle_object('pickled/df_att.pkl')

    # Instantiate ConvCNP
    model = ConvCNP(in_channels = len(C.context_channels)-1,
                    #rho=SimpleConv(),
                    rho=UNet(),
                    #rho=DepthSepConv1d(in_channels=rho_in_channels, conv_channels=64, num_layers=7, kernel_size=15),
                    points_per_unit=64*8,
                    dynamic_feature_embedding=False,
                    dynamic_embedding_dims=C.dynamic_embedding_dims,
                    static_embedding_dims=C.static_embedding_dims,
                    static_feature_embedding=C.static_feature_embedding,
                    static_embedding_in_channels=C.static_embedding_in_channels,
                    static_feature_missing_data=C.static_feature_missing_data,
                    static_embedding_location=C.static_embedding_location,
                    distribution=dist)

    # Assign model to device
    model.to(device)

    # Data generator
    gen = HydroDataset(
                                                dataframe = df_train,
                                                df_att = df_att,
                                                batch_size = 16,
                                                num_tasks = 16,
                                                channels_c = C.context_channels,
                                                channels_t = C.target_channels,
                                                channels_att = C.attributes,
                                                channels_t_val = C.target_val_channel,
                                                context_mask = C.context_mask,
                                                target_mask = C.target_mask,
                                                extrapolate = C.extrapolate_flag,
                                                timeslice = C.timeslice,
                                                dropout_rate = 0, #  0.3,
                                                concat_static_features = C.concat_static_features,
                                                min_train_points= C.min_train_points,
                                                min_test_points= C.min_test_points,
                                                max_train_points= C.max_train_points,
                                                max_test_points= C.max_test_points,
                                                device='cpu',
                                                )

    # Create a fixed set of outputs to predict at when plotting.
    x_test = torch.linspace(0., 1.,C.timeslice)[None, :, None].to(device)

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

    # Instantiate data generator for validation.
    NUM_TEST_TASKS = 16 # 128
    gen_val = HydroDataset(
                                                dataframe=df_test_catchment,
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


    # Experiment folder
    change_folder = True

    if change_folder:
        experiment_name = 'test_NoEmbeddings'
        wd = WorkingDirectory(generate_root(experiment_name))

    # Reset epochs
    reset_epochs = True

    if reset_epochs:
        train_obj_list, train_nse_list, test_obj_list, epoch_list = [], [], [], []

    gen.batch_size = 32
    gen_val.batch_size = 32
    gen_test.batch_size = 32

    gen.num_tasks = 128
    gen_val.num_tasks = 64
    gen_test.num_tasks = 64

    gen.dropout_rate = 0
    gen_val.dropout_rate = 0
    gen_test.dropout_rate = 0
    static_masking_rate = 0

    num_workers = 6

    train_dataloader = DataLoader(dataset=gen, batch_size=1, num_workers=num_workers)
    test_dataloader = DataLoader(dataset=gen_test, batch_size=1, num_workers=num_workers)
    val_dataloader = DataLoader(dataset=gen_val, batch_size=1, num_workers=num_workers)

    # torch.autograd.set_detect_anomaly(False)

    # Some training hyper-parameters:
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 200
    PLOT_FREQ = 1

    plot_model = False

    if len(epoch_list)>0:
        last_epoch = epoch_list[-1]
    else:
        last_epoch = 0

    # Initialize optimizer
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Run the training loop.
    for epoch in range(NUM_EPOCHS):
        # Compute training objective.
        start_time = time.time()
        train_obj, train_nse = train(train_dataloader, model, opt, dist=dist)

        epoch_list.append(epoch+last_epoch)
        train_obj_list.append(train_obj)
        train_nse_list.append(train_nse)
        
        #torch.cuda.empty_cache()
        test_obj = test(test_dataloader,model,dist=dist)
        #torch.cuda.empty_cache()
        test_obj_list.append(test_obj)
        
        elapsed = time.time() - start_time
        
        # Plot model behaviour every now and again.
        if (epoch % PLOT_FREQ == 0) and plot_model:
            
            task = task_to_device(val_dataloader.__getitem__())
            fig = plt.figure(figsize=(24, 5))
            
            for i in range(1):
                plt.subplot(1, 1, i + 1)
                plot_model_task(model, task, timeslice=C.timeslice, idx=i, legend=i==2, dist=dist)
            
            plt.title('Test set')
            display.clear_output(wait=True)


        print('Epoch %s ¦ train NLL: %.3f ¦ test NLL: %.3f ¦ train NSE: %.3f ¦ time: %.3f' % (epoch + last_epoch, train_obj, test_obj, train_nse, elapsed))

        plot_training_loss(train_obj_list, test_obj_list)
        plt.show()
        
        save_as_best = True if test_obj == min(test_obj_list) else False
        save_checkpoint(wd,model.state_dict(),is_best=save_as_best)
        
        PATH = os.path.join(wd.root,'e_%s_loss_%.3f.pth.tar' % (epoch + last_epoch, test_obj))
        torch.save(model.state_dict(), PATH)