from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

class autoencoderDataset(Dataset):
    def __init__(self, dataset_file):
        data_frame   = pd.read_csv(dataset_file, sep=" ", header=None)

        self.data_np = data_frame.to_numpy()

    def __len__(self):
        return self.data_np.shape[0] # Returning nos. of rows

    def __getitem__(self, idx):
        return {
            "X": torch.tensor(self.data_np[idx]),
            "Y": torch.tensor(self.data_np[idx])
        }