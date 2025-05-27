import numpy as np 
import torch

def nse(obs, mean_obs, sim, epsilon=0.1):
    return (1 - np.sum((sim - obs)**2)/np.sum((obs - mean_obs)**2))

def nse_tensor(obs, sim, basin_stds,epsilon=0.1, reduction='mean'):
    # numerator = torch.sum((obs - sim)**2, dim=1)
    # denominator = torch.sum((obs - mean_obs)**2, dim=1)
    # print(numerator, denominator, 1 - numerator/denominator)
    # return 1 - numerator/denominator

    # basin_means and basin_stads are lists of means and stds for each basin: use them as the denominator of each basin's nse

    
    nses = 1 - torch.mean((obs - sim)**2,dim=1).squeeze()/(basin_stds+epsilon)**2

    # nses = 1 - torch.sum((obs - sim)**2,dim=1) / torch.sum((obs - mean_obs)**2,dim=1)
    if reduction == 'mean':
        return torch.mean(nses)
    elif reduction == 'median':
        return torch.median(nses)
    elif reduction == 'none':
        return nses
    else:
        raise RuntimeError(f'Unknown reduction "{reduction}".')
    #print(obs.shape, sim.shape, mean_obs.shape, numerator.shape, denominator.shape, nse.shape, nses.min().item(), nses.max().item(), nse.item())
    return nse

def nse_tensor_batch(obs, mean_obs, sim):
    return torch.sum(1 - torch.mean((obs - sim)**2,dim=0)/torch.sum((obs - mean_obs)**2,dim=0))

def squared_dist(x, y):
    return torch.sum((x - y)**2)

def aplha_nse(obs, mean_obs, sim):
    return np.std(sim) / np.std(obs)

def beta_nsa(obs,sim):
    return (np.mean(sim) - np.mean(obs)) / np.std(obs)

# if name is main:
if __name__ == "__main__":
    # create torch tensors of shape (16,30)
    obs = torch.rand(16,30)
    sim = torch.rand(16,30)
    basin_stds = np.random.rand(16)
    basin_stds[0] = 5
    nse = nse_tensor(obs, sim, basin_stds)
    print(nse.item())