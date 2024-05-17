import torch
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma

def _reduce(logp, reduction):

    if not reduction:
        return logp
    elif reduction == 'sum':
        return torch.sum(logp)
    elif reduction == 'mean':
        return torch.mean(logp)
    elif reduction == 'batched_mean':
        return torch.mean(torch.sum(logp, 1))
    else:
        raise RuntimeError(f'Unknown reduction "{reduction}".')

def gaussian_logpdf(inputs, mean, sigma, reduction=None):
    """Gaussian log-density.

    Args:
        inputs (tensor): Inputs.
        mean (tensor): Mean.
        sigma (tensor): Standard deviation.
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".

    Returns:
        tensor: Log-density.
    """
    dist = Normal(loc=mean, scale=sigma)
    logp = dist.log_prob(inputs)

    return _reduce(logp, reduction)

def gamma_logpdf(inputs, loc, scale, reduction=None):
    """Gamma log-density.

    Args:
        inputs (tensor): Inputs.
        mean (tensor): Mean.
        sigma (tensor): Standard deviation.
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".

    Returns:
        tensor: Log-density.
    """
    dist = Gamma(concentration=loc, rate=scale)
    logp = dist.log_prob(inputs)

    return _reduce(logp, reduction)
    
def gamma_stats(loc, scale):
    """Compute mean and standard deviation of gamma distribution
    
    Args:

    Returns:
        tensor: 
        
    """
    gamma_mean = torch.distributions.gamma.Gamma(loc, scale).mean
    gamma_std = torch.sqrt(torch.distributions.gamma.Gamma(loc, scale).variance)
    return gamma_mean, gamma_std


def compute_logpdf(y_loc, y_scale, task, dist='gaussian', return_mu_and_sigma=False):

    if dist == 'gaussian':
        y_mean, y_std = y_loc, y_scale
        obj = -gaussian_logpdf(task['y_target'], y_loc, y_scale, 'batched_mean')
    elif dist == 'gamma':
        y_mean, y_std = gamma_stats(y_loc, y_scale)
        obj = -gamma_logpdf(task['y_target'], y_loc, y_scale, 'batched_mean')
    elif dist == 'gaussian_fixed':
        y_mean, y_std = y_loc, y_scale
        obj = -gaussian_logpdf(task['y_target'], y_loc, y_scale/y_scale, 'batched_mean')

    if return_mu_and_sigma:
        return obj, y_mean, y_std
    else:
        return obj
    
