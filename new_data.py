import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class HydroGeneratore:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def generate_tasks(self):
        for x, y_c, y_t, attributes in self.dataloader:
            x, y_c, y_t, attributes = map(lambda t: torch.tensor(t).float().to(self.device), [x, y_c, y_t, attributes])
            yield {'x': x, 'y_context': y_c, 'y_target': y_t, 'attributes': attributes}

def create_dataloader(dataframe, df_att, channels_c, channels_t, channels_att, batch_size, shuffle=True, num_workers=0):
    dataset = HydroDataset(dataframe, df_att, channels_c, channels_t, channels_att)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

class HydroDataset(Dataset):
    def __init__(self, dataframe, df_att, channels_c, channels_t, channels_att, timeslice=60):
        self.dataframe = dataframe
        self.df_att = df_att
        self.channels_c = channels_c
        self.channels_t = channels_t
        self.channels_att = channels_att
        self.timeslice = timeslice

    def __len__(self):
        return len(self.dataframe) - self.timeslice

    def __getitem__(self, index):
        # Ensures that indexing does not go out of bounds
        start_index = index
        end_index = index + self.timeslice

        # Extract data for the slice
        df_slice = self.dataframe.iloc[start_index:end_index].copy()

        # Extract attributes
        attributes = self.sample_att(df_slice['hru08'].iloc[0])

        # Prepare inputs and outputs
        x = self._prepare_x(df_slice)
        y_c = np.vstack([df_slice[channel].values for channel in self.channels_c])
        y_t = np.vstack([df_slice[channel].values for channel in self.channels_t])

        return x, y_c, y_t, attributes

    def sample_att(self, hru08):
        return np.vstack([self.df_att[key][self.df_att['hru08'] == hru08].values for key in self.channels_att])

    def _prepare_x(self, df_slice):
        return np.linspace(0, 1, num=self.timeslice)  # Normalized time steps

