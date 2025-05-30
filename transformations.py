from scipy.special import inv_boxcox
from scipy.stats import boxcox
import numpy as np
import torch

#__all__ = [standardise,normalise,log_transform,boxcox_transform,rev_standardise,rev_normalise,rev_log_transform,rev_boxcox_transform] 

" TRANSFORMATIONS"

def scaling(df,
            fields, 
            method='std'):
    
    for field in fields:

        if field == 'QObs(mm/d)':
            if method == 'std':
                df[field], q_mu, q_sigma = standardise(df[field], stats=True)
            elif method == 'norm':
                df[field], q_min, q_max = normalise(df[field],stats=True)
    
        else:
            if method == 'std':
                df[field] = standardise(df[field], stats=False)
            elif method == 'norm':
                df[field] = normalise(df[field])

    if method == 'std':
        return df, q_mu, q_sigma
    elif method == 'norm':
        return df, q_min, q_max

def rev_transform(x,q_mu, q_sigma, a=0,b=0):
    #return x
    return rev_standardise(x,q_mu,q_sigma)
    #return rev_log_transform(x-1,e=1)
    #return rev_normalise(x-1,q_min,q_max)
    #return rev_standardise(x-Q_shift,q_mu,q_sigma)
    #return rev_log_transform(x,e=1)
    #return rev_normalise(rev_log_transform(x-1,e=1), q_min, q_max)
    #return rev_log_transform(x+Q_shift-1,e=1)
    #return rev_boxcox_transform(x,ld=lambda_OBS_RUN)

def rev_transform_tensor(x,q_mu, q_sigma, a=0,b=0):
    #return x
    return rev_standardise(x,q_mu,q_sigma)
    #return torch.exp(x-1)-1
    #return torch.exp(x)-1
    #return rev_normalise(x-1,q_min,q_max)
    #return rev_standardise(torch.exp(x)-Q_shift,q_mu,q_sigma)
    #return rev_standardise(x-1,q_mu,q_sigma)
    #return torch.exp(x+Q_shift-1)-1
    #return rev_normalise(torch.exp(x-1)-1, q_min, q_max)
    #return rev_boxcox_transform_tensor(x,ld=lambda_OBS_RUN)

def standardise(x,stats=False):
    mu = np.mean(x)
    sigma = np.std(x)
    if stats:
        return (x - mu)/sigma, mu, sigma
    else:
        return (x - mu)/sigma

def normalise(x,stats=False):
    x_max = np.max(x)
    x_min = np.min(x)
    if stats:
        return (x - x_min)/(x_max - x_min), x_min, x_max
    else:
        return (x - x_min)/(x_max - x_min)

def log_transform(x, e=0):
    return np.log(x+e)

def boxcox_transform(x, e=0):
    return boxcox(x+e)

"INVERSE TRANSFORMATIONS"

def rev_standardise(x_t, mu, sigma):
    return x_t*sigma + mu

def rev_normalise(x_t, min, max):
    return x_t*(max - min) + min   
    
def rev_log_transform(x, e=0):
    return np.exp(x)-e  

def rev_boxcox_transform(x, ld):
    return inv_boxcox(x, ld)

'REV BOXCOX TRANSFORM USING TENSORS'

def rev_boxcox_transform_tensor(x,ld):
        if ld == 0:
            return torch.exp(x)
        else:
            return torch.exp(torch.log(ld*x+1)/ld) 

def rev_transform(x, transform="NONE", scaling="NONE", mu=None, sigma=None, min=None, max=None, e=1, ld=None, s=0):

    if scaling == "STANDARD":
        x = rev_standardise(x, mu=mu, sigma=sigma)
    elif scaling == "NORM":
        x = rev_normalise(x, x_min=min, x_max=max)
    else:
        pass

    if transform == "LOG":
        x = rev_log_transform(x,e=e)
    if transform == "BOXCOX":
        x = rev_boxcox_transform(x, ld=ld)

    return x - s


def rev_transform_tensor(x, sigma_log = None, is_label = True, transform="NONE", scaling="NONE", mu=None, sigma=None, min=None, max=None, e=1, ld=None, s=0):

    if scaling == "STANDARD":
        x = rev_standardise(x, mu=mu, sigma=sigma)
    elif scaling == "NORM":
        x = rev_normalise(x, x_min=min, x_max=max)
    else:
        pass

    if transform == "LOG":
        if is_label:
            x = torch.exp(x) - e
        else:
            x, _ = rev_lognormal(mu_log = x, sigma_log = sigma_log, e = 1)
    if transform == "BOXCOX":
        x = rev_boxcox_transform_tensor(x, ld=ld)

    return x - s

def rev_lognormal(mu_log, sigma_log, e):
    E = torch.exp(mu_log + 0.5*sigma_log**2) - e
    Var = (E**2)*(torch.exp(sigma_log**2)-1)
    return E, torch.sqrt(Var)

def scale_dataframe(df, columns, stats_dict=None):
    """
    Standardize the given DataFrame by column and return the scaled DataFrame and statistics.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to be standardized.
    stats_dict (dict, optional): A dictionary containing the mean and standard deviation of each column.
                                 If provided, these statistics will be used to scale the DataFrame.
                                 If not provided, statistics will be computed from the given DataFrame.
    
    Returns:
    scaled_df (pd.DataFrame): The standardized DataFrame.
    stats_dict (dict): A dictionary containing the mean and standard deviation of each column.
    """
    if stats_dict is None:
        stats_dict = {}
        for column in columns:
            mean = df[column].mean()
            std = df[column].std()
            stats_dict[column] = {'mean': mean, 'std': std}
            df[column] = (df[column] - mean) / std
    else:
        for column in columns:
            if column in stats_dict:
                mean = stats_dict[column]['mean']
                std = stats_dict[column]['std']
                df[column] = (df[column] - mean) / std
            else:
                raise ValueError(f"Statistics for column '{column}' not found in stats_dict")

    return df, stats_dict
