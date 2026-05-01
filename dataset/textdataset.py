import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

# print current os path

df = pd.read_csv('./dataset/ground_truth.csv')
reference = df

class TextDataset(Dataset):
    def __init__(self, labelpath, peoplelist, pft=True):
        self.labelpath = labelpath
        self.peoplelist = peoplelist
        self.pft = pft
        self.personal_info, self.vo2maxlabels, self.vo2atlabels = self.prepare_data(labelpath, peoplelist)

    def __len__(self):
        return len(self.personal_info)

    def __getitem__(self, idx):
        return self.personal_info[idx], (self.vo2maxlabels[idx] - 23) / 10, (self.vo2atlabels[idx] - 15) / 10
    
    def prepare_data(self, labelpath, peoplelist):
        personal_info = []
        vo2maxlabels = []
        vo2atlabels = []

        rawlabels = pd.read_csv(labelpath)

        for i in range(len(peoplelist)):
            person = peoplelist[i]
            age = rawlabels[rawlabels['id'] == person]['age'].values[0] / 100
            height = rawlabels[rawlabels['id'] == person]['height'].values[0] / 200
            weight = rawlabels[rawlabels['id'] == person]['weight'].values[0] / 100
            fvc0 = rawlabels[rawlabels['id'] == person]['fvc0'].values[0] / 10 if self.pft else 0
            fev10 = rawlabels[rawlabels['id'] == person]['fev10'].values[0] / 10 if self.pft else 0
            ratio0 = rawlabels[rawlabels['id'] == person]['ratio0'].values[0] / 100 if self.pft else 0
            pef0 = rawlabels[rawlabels['id'] == person]['pef0'].values[0] / 10 if self.pft else 0
            fvc = rawlabels[rawlabels['id'] == person]['fvc'].values[0] / 10 if self.pft else 0
            fev1 = rawlabels[rawlabels['id'] == person]['fev1'].values[0] / 10 if self.pft else 0
            ratio = rawlabels[rawlabels['id'] == person]['ratio'].values[0] / 100 if self.pft else 0
            pef = rawlabels[rawlabels['id'] == person]['pef'].values[0] / 10 if self.pft else 0

            vo2max = rawlabels[rawlabels['id'] == person]['vo2max'].values[0]
            vo2at = rawlabels[rawlabels['id'] == person]['vo2at'].values[0]


            personal_info.append([float(age), float(height), float(weight), float(fvc0), float(fev10), float(ratio0), float(pef0), float(fvc), float(fev1), float(ratio), float(pef)])
            vo2maxlabels.append(float(vo2max))
            vo2atlabels.append(float(vo2at))
    
        personal_info = np.array(personal_info)
        vo2maxlabels = np.array(vo2maxlabels)
        vo2atlabels = np.array(vo2atlabels)
        return personal_info, vo2maxlabels, vo2atlabels

        