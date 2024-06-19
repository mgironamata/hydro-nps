import matplotlib.pyplot as plt 
import numpy as np 
import torch
import scipy.stats
import scipy.special
from typing import Dict, Tuple
from utils import *
from transformations import rev_transform, rev_transform_tensor
import NSE
import config as C


def plot_task(task, idx, legend):
    x_context, y_context = to_numpy(task['x_context'][idx]), to_numpy(task['y_context'][idx][:,0])
    x_target, y_target = to_numpy(task['x_target'][idx]), to_numpy(task['y_target'][idx][:,0])
    y_target_val = to_numpy(task['y_target_val'][idx][:,0])
      
    # Plot context and target sets.
    plt.scatter(x_context, y_context, label='Context Set', color='blue', marker='.')
    plt.scatter(x_target, y_target, label = 'Target Set', color='lightblue', marker='.')
    plt.plot(x_target, y_target_val, label='Target Mean', color='green')
    
    if legend:
        plt.legend()

def plot_model_task(model, task, timeslice, idx, legend, dist='gaussian', q_mu = None, q_sigma = None, plot = True):
    
    snum_functions = task['x_context'].shape[0]
    x_test = torch.linspace(0., 1.,timeslice)[None, :, None].to(device)
    
    # Make predictions with the model.
    model.eval()
    with torch.no_grad():
        y_loc, y_scale = model(task['x_context'], 
                               task['y_context'], 
                               task['x_target'],
                               #x_test.repeat(num_functions, 1, 1), 
                               task['y_att'], 
                               #task['feature'],
                               #task['m'],
                               )
    
    
        #y_loc_NSE, y_scale_NSE = model(task['x_context'], task['y_context'], task['x_target'], task['y_att'], task['feature'], task['m'], embedding=feature_embedding_flag)

    # Plot the task and the model predictions.
    x_context, y_context = (task['x_context'][idx]), (task['y_context'][idx][:,0])
    x_target, y_target = (task['x_target'][idx]), (task['y_target'][idx][:,0])
    #y_mean_NSE, y_std_NSE = to_numpy(y_mean_NSE[idx][:,0]), to_numpy(y_std_NSE[idx][:,0])
    y_target_val = (task['y_target_val'][idx][:,0])
    
    if dist == 'gaussian':
        y_mean, y_sigma = y_loc, y_scale
        # y_mean, y_sigma = to_numpy(y_mean[idx][:,0]), to_numpy(y_sigma[idx][:,0])
        y_mean, y_sigma = (y_mean[idx][:,0]), (y_sigma[idx][:,0])
        p05, p95 = y_mean + 2 * y_sigma, y_mean - 2 * y_sigma
        #y_mean_NSE, y_std_NSE = y_loc_NSE, y_scale_NSE
    elif dist == 'gamma':
        y_mean, y_sigma = gamma_stats(y_loc, y_scale)
        # y_mean, y_sigma = to_numpy(y_mean[idx][:,0]), to_numpy(y_sigma[idx][:,0])
        y_loc, y_scale = to_numpy(y_loc[idx][:,0]), to_numpy(y_scale[idx][:,0])        
        p05 = scipy.stats.gamma.ppf(0.05, y_loc, 0, 1/y_scale)
        p95 = scipy.stats.gamma.ppf(0.95, y_loc, 0, 1/y_scale)
        #y_mean_NSE, y_std_NSE = gamma_stats(y_loc_NSE, y_scale_NSE)
    elif dist == 'gaussian_fixed':
        y_mean, y_sigma = y_loc, y_scale/y_scale
        # y_mean, y_sigma = to_numpy(y_mean[idx][:,0]), to_numpy(y_sigma[idx][:,0])
        p05, p95 = y_mean + 2 * y_sigma, y_mean - 2 * y_sigma
        #y_mean_NSE, y_std_NSE = y_loc_NSE, y_scale_NSE

    # nse = NSE.nse(rev_transform_tensor(y_target, mu=q_mu, sigma=q_sigma, scaling='STANDARD'), 
    #               rev_transform(y_target_val, mu=q_mu, sigma=q_sigma, scaling='STANDARD'),
    #               rev_transform(y_mean, mu=q_mu, sigma=q_sigma, scaling='STANDARD')
    #               )
    
    # y_mean and y_sigma to numpy arrays
    y_mean, y_sigma = y_mean.detach(), y_sigma.detach()
    
    obs = to_numpy(rev_transform_tensor(y_target, scaling='STANDARD', 
                                        transform=C.transform, mu=q_mu, sigma=q_sigma))

    context_obs = to_numpy(rev_transform_tensor(y_context[...,0:], 
                                                transform=C.transform, scaling='STANDARD', mu=q_mu, sigma=q_sigma))

    mean_obs = to_numpy(rev_transform_tensor(y_target_val, scaling='STANDARD', 
                                             transform=C.transform,
                                             mu=q_mu, sigma=q_sigma))

    sim = to_numpy(rev_transform_tensor(y_mean, scaling='STANDARD', 
                                        transform=C.transform,
                                        mu=q_mu, sigma=q_sigma, is_label=True, sigma_log=y_sigma))
    
    sim2 = to_numpy(rev_transform_tensor(y_mean, scaling='STANDARD', 
                                         transform=C.transform, 
                                         mu=q_mu, sigma=q_sigma, is_label=False, sigma_log=y_sigma))
        
    nse = NSE.nse(obs = obs,
                  mean_obs = mean_obs,
                  sim = sim2 
                )
    
    #log_nse = NSE.nse(y_mean_NSE, y_target_val, y_target)
    
    x_context = to_numpy(x_context)*timeslice
    x_target = to_numpy(x_target)*timeslice
    x_test = x_target #to_numpy(x_test[0,:,0])*timeslice

    x_test, sim, sim2 = zip(*sorted(zip(x_test, sim, sim2))) #
    x_target, obs, mean_obs = zip(*sorted(zip(x_target, obs, mean_obs))) #

    if plot:
    
        plt.figure(figsize=(15, 2.5))

        # plt.scatter(x_context, context_obs, 
        #             label='Context Set', color='orange', marker='.')

        plt.plot(x_target, obs, 
                    label='Target Set', color='lightgreen', marker='.', linewidth=0.5)
        
        plt.plot(x_target, mean_obs, 
                    label='Target Mean', color='green', linewidth=1)
        
        
        # Plot model predictions.
        # plt.plot(x_test, sim, 
        #         label='Model Output', color='blue')
        
        # Plot model predictions.
        plt.plot(x_test, sim2, 
                label='Model Output', color='blue', marker='.', linewidth=0.5)
        
        # plt.fill_between(x_test,
        #                  rev_transform(p95, mu=q_mu, sigma=q_sigma, scaling='STANDARD'),
        #                  rev_transform(p05, mu=q_mu, sigma=q_sigma, scaling='STANDARD'),
        #                  color='tab:blue', alpha=0.2)
            
        plt.title("NSE(1): %.3f " % nse)
        #plt.xlabel("Q")
        #plt.ylabel("y-label")
            
        if legend:
            plt.legend()
        plt.show()

        # plt.savefig('task.png')

    return nse

def plot_training_loss(train_loss, test_loss):
    
    fig = plt.figure(figsize=(7,5))
    
    plt.plot(train_loss,'r',label='train NLL')
    plt.plot(test_loss, 'b', label='test NLL')
    plt.ylabel('NLL')
    plt.xlabel('# epochs')

    plt.legend()

    #fig.suptitle('ConvCNP (Gaussian LL w/ new decoder)', fontsize = 16)
    plt.show()

def ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate empirical cummulative density function
    
    Parameters
    ----------
    x : np.ndarray
        Array containing the data
    
    Returns
    -------
    x : np.ndarray
        Array containing the sorted metric values
    y : np.ndarray]
        Array containing the sorted cdf values
    """
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    return xs, ys