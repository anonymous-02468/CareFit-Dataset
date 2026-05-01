import numpy as np
import torch
from torch.utils.data import Dataset
import sys
sys.path.append(".")
sys.path.append('..')
from dataset.ecgdataset import ECGDataset
from dataset.smdataset import SMDataset
from dataset.textdataset import TextDataset

class MultimodalityDataset(Dataset):
    def __init__(self, people_list, label_id='vo2max', transform=None, train=False):
        self.people_list = people_list
        self.label_id = label_id
        self.transform = transform
        self.path = "./data/"
        self.ecgmodality = 'ecg_clean'
        self.ecgdataset = ECGDataset(self.path,  people_list, modality_list=self.ecgmodality)

        self.smdataset = SMDataset(self.path, people_list)

        self.textdataset = TextDataset("./dataset/ground_truth.csv", peoplelist=people_list)
    
    def __len__(self):
        return len(self.ecgdataset.data)

    def __getitem__(self, idx):
        ecg_sample = self.ecgdataset[idx][0]
        sm_sample = self.smdataset[idx][0]
        personal_info = self.textdataset[idx][0]
        vo2maxlabels = self.textdataset[idx][1]
        vo2atlabels = self.textdataset[idx][2]

        return (ecg_sample, sm_sample, personal_info), vo2maxlabels, vo2atlabels

def multimodality_custom_collate_fn(data):
    signals, vo2maxlabels, vo2atlabels = zip(*data)
    ecg_max_len = max(len(s[0][0]) for s in signals)
    sm_max_len = max(len(s[1][0]) for s in signals)
    ecg_padded_signals = []
    ecg_lengths = []
    sm_padded_signals = []
    sm_lengths = []
    personal_infos = []

    vo2maxlabels = np.array(vo2maxlabels)
    vo2atlabels = np.array(vo2atlabels)


    for signal in signals:
        ecg_signal = signal[0]
        sm_signal = signal[1]
        personal_info = signal[2]

        ecg_padded_signal = torch.zeros((1,ecg_max_len))
        ecg_padded_signal[0,0:len(ecg_signal[0])] = torch.tensor(ecg_signal[0])
        ecg_padded_signals.append(ecg_padded_signal)
        ecg_lengths.append(len(ecg_signal[0]))

        sm_padded_signal = torch.zeros((1,sm_max_len))
        sm_padded_signal[0,0:len(sm_signal[0])] = torch.tensor(sm_signal[0])
        sm_padded_signals.append(sm_padded_signal)
        sm_lengths.append(len(sm_signal[0]))

        personal_info = torch.tensor(personal_info).float()
        personal_infos.append(personal_info)
    
    ecg_padded_signals = torch.stack(ecg_padded_signals, dim=0)
    sm_padded_signals = torch.stack(sm_padded_signals, dim=0)
    personal_infos = torch.stack(personal_infos, dim=0)

    padded_signals = (ecg_padded_signals, sm_padded_signals, personal_infos)
    vo2maxlabels = torch.tensor(vo2maxlabels).float()
    vo2atlabels = torch.tensor(vo2atlabels).float()

    return padded_signals, vo2maxlabels, vo2atlabels
