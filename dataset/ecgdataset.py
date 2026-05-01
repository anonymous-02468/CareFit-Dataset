import numpy as np
import os
from matplotlib import pyplot as plt
from glob import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd

class ECGDataset(Dataset):
    def __init__(self, data_path, people_list, modality_list, label_id='vo2max', transform=None):
        self.data_path = data_path
        self.people_list = people_list
        self.modality_list = modality_list
        self.label_id = label_id
        self.transform = transform
        self.data = []
        self.labels = []
        self.load_data()

    def load_data(self):
        gt = pd.read_csv('./dataset/ground_truth.csv')
        for person in self.people_list:
            file_path = os.path.join(self.data_path, person, f"{self.modality_list}.npz")
            data = np.load(file_path, allow_pickle=True)
            self.data.append(data['data'][::, 0]) if len(data['data'].shape) == 2 else self.data.append(data['data'])
            self.labels.append(gt[gt["id"] == person][self.label_id].values[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample[::][np.newaxis, :], np.array(label).reshape(1)
