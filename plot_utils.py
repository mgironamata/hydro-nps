import matplotlib.pyplot as plt 
import numpy as np 
import torch
import scipy.stats
import scipy.special
from typing import Dict, Tuple
from utils import *
from transformations import rev_transform
import NSE


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

def plot_model_task(model, task, timeslice, idx, legend, dist='gaussian', q_mu = None, q_sigma = None):
    num_functions = task['x_context'].shape[0]
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
    x_context, y_context = to_numpy(task['x_context'][idx]), to_numpy(task['y_context'][idx][:,0])
    x_target, y_target = to_numpy(task['x_target'][idx]), to_numpy(task['y_target'][idx][:,0])
    #y_mean_NSE, y_std_NSE = to_numpy(y_mean_NSE[idx][:,0]), to_numpy(y_std_NSE[idx][:,0])
    y_target_val = to_numpy(task['y_target_val'][idx][:,0])
    
    if dist == 'gaussian':
        y_mean, y_std = y_loc, y_scale
        y_mean, y_std = to_numpy(y_mean[idx][:,0]), to_numpy(y_std[idx][:,0])
        p05, p95 = y_mean + 2 * y_std, y_mean - 2 * y_std
        #y_mean_NSE, y_std_NSE = y_loc_NSE, y_scale_NSE
    elif dist == 'gamma':
        y_mean, y_std = gamma_stats(y_loc, y_scale)
        y_mean, y_std = to_numpy(y_mean[idx][:,0]), to_numpy(y_std[idx][:,0])
        y_loc, y_scale = to_numpy(y_loc[idx][:,0]), to_numpy(y_scale[idx][:,0])        
        p05 = scipy.stats.gamma.ppf(0.05, y_loc, 0, 1/y_scale)
        p95 = scipy.stats.gamma.ppf(0.95, y_loc, 0, 1/y_scale)
        #y_mean_NSE, y_std_NSE = gamma_stats(y_loc_NSE, y_scale_NSE)
    elif dist == 'gaussian_fixed':
        y_mean, y_std = y_loc, y_scale/y_scale
        y_mean, y_std = to_numpy(y_mean[idx][:,0]), to_numpy(y_std[idx][:,0])
        p05, p95 = y_mean + 2 * y_std, y_mean - 2 * y_std
        #y_mean_NSE, y_std_NSE = y_loc_NSE, y_scale_NSE

    nse = NSE.nse(rev_transform(y_target, mu=q_mu, sigma=q_sigma, scaling='STANDARD'), 
                  rev_transform(y_target_val, mu=q_mu, sigma=q_sigma, scaling='STANDARD'),
                  rev_transform(y_mean, mu=q_mu, sigma=q_sigma, scaling='STANDARD')
                  )
    
    #log_nse = NSE.nse(y_mean_NSE, y_target_val, y_target)
    
    x_context = x_context*timeslice
    x_target = x_target*timeslice
    x_test = x_target #to_numpy(x_test[0,:,0])*timeslice

    plt.figure(figsize=(10, 3))
    
    # plt.scatter(x_context, rev_transform(y_context), label='Context Set', color='black')
    plt.scatter(x_target, rev_transform(y_target, mu=q_mu, sigma=q_sigma, scaling='STANDARD'), 
                label='Target Set', color='red', marker='.')
    
    # plt.scatter(x_target, rev_transform(y_mean_NSE), label='Target Predictions', color='orange')
    plt.scatter(x_target, rev_transform(y_target_val, mu=q_mu, sigma=q_sigma, scaling='STANDARD'), 
                label='Target Mean', color='green', marker='.')

    # Plot model predictions.
    plt.plot(x_test, rev_transform(y_mean, mu=q_mu, sigma=q_sigma, scaling='STANDARD'), 
             label='Model Output', color='blue')
    
    plt.fill_between(x_test,
                     rev_transform(p95, mu=q_mu, sigma=q_sigma, scaling='STANDARD'),
                     rev_transform(p05, mu=q_mu, sigma=q_sigma, scaling='STANDARD'),
                     color='tab:blue', alpha=0.2)
        
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