# import torch
# from torch.utils.data import Dataset, DataLoader
# import multiprocessing

# # Ensure the right start method for multiprocessing is used
# multiprocessing.set_start_method('spawn', force=True)

# class RandomDataset(Dataset):
#     def __init__(self, size, length):
#         self.size = size
#         self.length = length

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         return torch.randn(self.size)

# def main():
#     dataset = RandomDataset(size=(3, 224, 224), length=1000)
#     dataloader = DataLoader(dataset, batch_size=32, num_workers=4)

#     for batch in dataloader:
#         print(batch.shape)

# if __name__ == "__main__":
#     main()

import torch
from torch.utils.data import Dataset
import numpy as np

import torch
from torch.utils.data import Dataset
import numpy as np
import random

from task_preprocessing import prep_task

def _rand(val_range, *shape):
    lower, upper = val_range
    return random.sample(range(int(lower),int(upper)),*shape)
    #return lower + np.random.rand(*shape) * (upper - lower)

def _uprank(a):
    if len(a.shape) == 1:
        return a[:, None, None]
    elif len(a.shape) == 2:
        return a[:, :, None]
    elif len(a.shape) == 3:
        return a
    else:
        return ValueError(f'Incorrect rank {len(a.shape)}.')


class HydroDataset(Dataset):
    """ Generate samples from hydrological data"""
    
    def __init__(self,
                dataframe,
                df_att,
                channels_c = ['OBS_RUN'],
                channels_t = ['OBS_RUN'],
                channels_att = ['gauge_id'],
                channels_t_val = ['OBS_RUN_log_n_mean'],
                context_mask = [1,1,1,1],
                target_mask = [0,1,1,1],
                extrapolate = True,
                timeslice = 60,
                dropout_rate = 0,
                concat_static_features = True,
                device = 'cuda',
                batch_size=16,
                num_tasks=256,
                x_range=(-2, 2),
                min_train_points = 10,
                min_test_points = 10,
                max_train_points=15,
                max_test_points=15):     

        self.dataframe = dataframe
        self.df_att = df_att
        self.channels_c = channels_c
        self.channels_t = channels_t
        self.channels_att = channels_att
        self.channels_t_val = channels_t_val
        self.context_mask = context_mask
        self.target_mask = target_mask
        self.extrapolate = extrapolate
        self.timeslice = timeslice
        self.dropout_rate = dropout_rate
        self.concat_static_features = concat_static_features
        self.device = device
        self.batch_size = batch_size
        self.num_tasks = num_tasks
        self.x_range = x_range
        self.min_train_points = min_train_points
        self.min_test_points = min_test_points
        self.max_train_points = max_train_points
        self.max_test_points = max_test_points

    def __len__(self):
        return 1000
    
    def sample(self,x,df):
        o1 = np.vstack(tuple(df[key][x] for key in self.channels_c))
        o2 = np.vstack(tuple(df[key][x] for key in self.channels_t))
        o3 = np.vstack(tuple(df[key][x] for key in self.channels_t_val)) 
        return o1, o2, o3
    
    def sample_att(self,hru08):
        return np.vstack(tuple(self.df_att[key][self.df_att['hru08']==hru08] for key in self.channels_att))

    def sample_date(self,x,df):
        return np.vstack(tuple(df[key][x] for key in ['YR','DOY']))

    def __getitem__(self,index=0):
        task = {'x': [],
                'y': [],
                'x_context': [],
                'y_context': [],
                'x_target': [],
                'y_target': [],
                'y_target_val': [],
                'y_att': [],
                }
        
        # Determine number of test and train points.
        num_train_points = np.random.randint(self.min_train_points, self.max_train_points + 1)
        num_test_points = np.random.randint(self.min_test_points, self.max_test_points + 1)
        num_points = num_train_points + num_test_points
        
        # Generate a random integer for each element in the bacth 
        randoms = np.random.randint(0,len(self.dataframe)-self.timeslice,self.batch_size)
        ids, year, hru08 = np.stack(self.dataframe[['id','YR','hru08']].iloc[randoms].values,axis=1).tolist()

        for i in range(self.batch_size):
        # Sample inputs and outputs.
            #x = _rand(self.x_range, num_points)
            s_ind, e_ind = randoms[i], randoms[i] + self.timeslice
            df = self.dataframe.iloc[s_ind:e_ind].copy()
            x_ind = _rand((s_ind, e_ind),num_points)
            
            # Sort x_ind if extrapolate is True
            if self.extrapolate == True:
                x_ind = sorted(x_ind)

            y, y_t, y_t_val = self.sample(x_ind,df)
            y_att = self.sample_att(hru08[i])

            x = np.divide(np.array(x_ind) - s_ind, e_ind - s_ind)

            # Determine indices for train and test set.
            if self.extrapolate:
                inds = np.arange(len(x))
            else:
                inds = np.random.permutation(len(x))                
            
            inds_train = sorted(inds[:num_train_points])
            inds_test = sorted(inds[num_train_points:num_points])

            # Record to task.
            task['x'].append(sorted(x))
            task['x_context'].append(x[inds_train])
            task['x_target'].append(x[inds_test])

            task['y_att'].append(y_att)

            y_aux, y_context_aux, y_target_aux = [], [], []
            
            for i in range(len(y)):
                y_aux.append(y[i][np.argsort(x)])
                y_context_aux.append(y[i][inds_train])
            
            for i in range(len(y_t)):
                y_target_aux.append(y_t[i][inds_test])
            
            task['y'].append(np.stack(y_aux,axis=1).tolist())
            task['y_context'].append(np.stack(y_context_aux,axis=1).tolist())
            task['y_target'].append(np.stack(y_target_aux,axis=1).tolist())

            #task['y'].append(y[0][np.argsort(x)])
            #task['y_context'].append(y[0][inds_train])
            #task['y_target'].append(y[0][inds_test])
            task['y_target_val'].append(y_t_val[0][inds_test])

        # Stack batch and convert to PyTorch.
        task = {k: torch.tensor(_uprank(np.stack(v, axis=0)),
                                dtype=torch.float32).to(self.device)
                for k, v in task.items()}

        task['y_att'] = task['y_att'].permute([0,2,1])
        task['y_att_context'] = task['y_att'] * torch.ones(task['x_context'].shape).to(self.device)
        task['y_att_target'] = task['y_att'] * torch.ones(task['x_target'].shape).to(self.device)
        
        if self.concat_static_features:
            task['y_context'] = torch.cat([task['y_context'],task['y_att_context']],dim=2)
            task['y_target'] = torch.cat([task['y_target'],task['y_att_target']],dim=2)

        task = prep_task(task,
                            context_mask=self.context_mask,
                            target_mask=self.target_mask,
                            dropout_rate=self.dropout_rate,
                            embedding=True,
                            concat_static_features=self.concat_static_features,
                            observe_at_target=True,
                            device=self.device)

        return task


class HydroTestDataset(Dataset):
    """ Generate samples from hydrological data"""
    
    def __init__(self,
                dataframe,
                df_att,
                channels_c = ['OBS_RUN'],
                channels_t = ['OBS_RUN'],
                channels_att = ['gauge_id'],
                channels_t_val = ['OBS_RUN_log_n_mean'],
                context_mask = [1,1,1,1],
                target_mask = [0,1,1,1],
                extrapolate = True,
                timeslice = 60,
                dropout_rate = 0,
                concat_static_features = True,
                device = 'cuda',
                batch_size=16,
                num_tasks=256,
                x_range=(-2, 2),
                min_train_points = 10,
                min_test_points = 10,
                max_train_points=15,
                max_test_points=15):     

        self.dataframe = dataframe
        self.df_att = df_att
        self.channels_c = channels_c
        self.channels_t = channels_t
        self.channels_att = channels_att
        self.channels_t_val = channels_t_val
        self.context_mask = context_mask
        self.target_mask = target_mask
        self.extrapolate = extrapolate
        self.timeslice = timeslice
        self.dropout_rate = dropout_rate
        self.concat_static_features = concat_static_features
        self.device = device
        self.batch_size = batch_size
        self.num_tasks = num_tasks
        self.x_range = x_range
        self.min_train_points = min_train_points
        self.min_test_points = min_test_points
        self.max_train_points = max_train_points
        self.max_test_points = max_test_points

    def __len__(self):
        return 1000
    
    def sample(self,x,df):
        o1 = np.vstack(tuple(df[key][x] for key in self.channels_c))
        o2 = np.vstack(tuple(df[key][x] for key in self.channels_t))
        o3 = np.vstack(tuple(df[key][x] for key in self.channels_t_val)) 
        return o1, o2, o3
    
    def sample_att(self,hru08):
        return np.vstack(tuple(self.df_att[key][self.df_att['hru08']==hru08] for key in self.channels_att))

    def sample_date(self,x,df):
        return np.vstack(tuple(df[key][x] for key in ['YR','DOY']))

    def __getitem__(self,index=0):
        task = {'x': [],
                'y': [],
                'x_context': [],
                'y_context': [],
                'x_target': [],
                'y_target': [],
                'y_target_val': [],
                'y_att': [],
                }
        
        # Determine number of test and train points.
        num_train_points = np.random.randint(self.min_train_points, self.max_train_points + 1)
        num_test_points = np.random.randint(self.min_test_points, self.max_test_points + 1)
        num_points = num_train_points + num_test_points
        
        # Generate a random integer for each element in the bacth 
        randoms = np.random.randint(0,len(self.dataframe)-self.timeslice,self.batch_size)
        ids, year, hru08 = np.stack(self.dataframe[['id','YR','hru08']].iloc[randoms].values,axis=1).tolist()

        for i in range(self.batch_size):
        # Sample inputs and outputs.
            #x = _rand(self.x_range, num_points)
            s_ind, e_ind = randoms[i], randoms[i] + self.timeslice
            df = self.dataframe.iloc[s_ind:e_ind].copy()
            x_ind = _rand((s_ind, e_ind),num_points)
            
            # Sort x_ind if extrapolate is True
            if self.extrapolate == True:
                x_ind = sorted(x_ind)

            y, y_t, y_t_val = self.sample(x_ind,df)
            y_att = self.sample_att(hru08[i])

            x = np.divide(np.array(x_ind) - s_ind, e_ind - s_ind)

            # Determine indices for train and test set.
            if self.extrapolate:
                inds = np.arange(len(x))
            else:
                inds = np.random.permutation(len(x))                
            
            inds_train = sorted(inds[:num_train_points])
            inds_test = sorted(inds[num_train_points:num_points])

            # Record to task.
            task['x'].append(sorted(x))
            task['x_context'].append(x[inds_train])
            task['x_target'].append(x[inds_test])

            task['y_att'].append(y_att)

            y_aux, y_context_aux, y_target_aux = [], [], []
            
            for i in range(len(y)):
                y_aux.append(y[i][np.argsort(x)])
                y_context_aux.append(y[i][inds_train])
            
            for i in range(len(y_t)):
                y_target_aux.append(y_t[i][inds_test])
            
            task['y'].append(np.stack(y_aux,axis=1).tolist())
            task['y_context'].append(np.stack(y_context_aux,axis=1).tolist())
            task['y_target'].append(np.stack(y_target_aux,axis=1).tolist())

            #task['y'].append(y[0][np.argsort(x)])
            #task['y_context'].append(y[0][inds_train])
            #task['y_target'].append(y[0][inds_test])
            task['y_target_val'].append(y_t_val[0][inds_test])

        # Stack batch and convert to PyTorch.
        task = {k: torch.tensor(_uprank(np.stack(v, axis=0)),
                                dtype=torch.float32).to(self.device)
                for k, v in task.items()}

        task['y_att'] = task['y_att'].permute([0,2,1])
        task['y_att_context'] = task['y_att'] * torch.ones(task['x_context'].shape).to(self.device)
        task['y_att_target'] = task['y_att'] * torch.ones(task['x_target'].shape).to(self.device)
        
        if self.concat_static_features:
            task['y_context'] = torch.cat([task['y_context'],task['y_att_context']],dim=2)
            task['y_target'] = torch.cat([task['y_target'],task['y_att_target']],dim=2)

        task = prep_task(task,
                            context_mask=self.context_mask,
                            target_mask=self.target_mask,
                            dropout_rate=self.dropout_rate,
                            embedding=True,
                            concat_static_features=self.concat_static_features,
                            observe_at_target=True,
                            device=self.device)

        return task


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import config as C
    from torch.utils.data import DataLoader
    import time

    df = pd.read_pickle('pickled/train.pkl')
    df_att = pd.read_pickle('pickled/df_att.pkl')

    dataset = HydroDataset(dataframe=df,
                           df_att=df_att,
                            channels_c = C.context_channels,
                            channels_t = C.target_channels,
                            channels_att = C.attributes,
                            channels_t_val = C.target_val_channel,
                            context_mask = C.context_mask,
                            target_mask = C.target_mask,
                            extrapolate=True,
                            timeslice=C.timeslice,
                            dropout_rate=0,
                            concat_static_features = C.concat_static_features,
                            min_train_points= C.min_train_points,
                            min_test_points= C.min_test_points,
                            max_train_points= C.max_train_points,
                            max_test_points= C.max_test_points,)
    
    dataloader = DataLoader(dataset, batch_size=1, num_workers=10)

    start = time.time()
    for idx, batch in enumerate(dataloader):
        print(batch['x'].shape)
        elapsed = time.time() - start
        if idx == 500:
            break
        
    print(f"Elapsed time: {elapsed:.2f} s")

    dataloader = DataLoader(dataset, batch_size=1, num_workers=9)

    start = time.time()
    for idx, batch in enumerate(dataloader):
        print(batch['x'].shape)
        elapsed = time.time() - start
        if idx == 500:
            break
    print(f"Elapsed time: {elapsed:.2f} s")