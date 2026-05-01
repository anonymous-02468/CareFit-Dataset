import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.set_printoptions(precision=4)
import sys
sys.path.append(".")
sys.path.append("..")
import os
from tqdm import tqdm
from dataset.multimodalitydataset import MultimodalityDataset, multimodality_custom_collate_fn
from einops import rearrange, repeat
import random
from model import EnhancedTimeSeriesModel, ECGNet, MultimodalFusionBlock
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flag", type=str, default='vo2max')
    args = parser.parse_args()

    test_list = ['s1', 's2', 's3', 's4', 's5']
    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flag = args.flag

    test_dataset = MultimodalityDataset(people_list=test_list, train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=multimodality_custom_collate_fn)

    ecg_encoder = ECGNet(use_knowledge_jumping=True)
    smo2_encoder = EnhancedTimeSeriesModel()

    fusion_block = MultimodalFusionBlock(d_model=32, 
                                        ecg_encoder=ecg_encoder, 
                                        smo2_encoder=smo2_encoder).to(device)
    fusion_block = fusion_block.to(device)

    fusion_block.load_state_dict(torch.load(f"ckpts/model_{flag}.pth"))
    fusion_block.eval()

    output_list = []
    label_list = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            (ecg, smo2, person_info), vo2maxlabel, vo2atlabel = batch
            ecg = ecg.float().to(device)
            smo2, person_info = smo2.float().to(device), person_info.float().to(device)
            vo2maxlabel, vo2atlabel = vo2maxlabel.float().to(device), vo2atlabel.float().to(device)

            output1, output2 = fusion_block(ecg, smo2, person_info)
            if flag == "vo2max":
                output_list.append(output1.cpu().numpy())
                label_list.append(vo2maxlabel.cpu().numpy())
            
            else:
                output_list.append(output2.cpu().numpy())
                label_list.append(vo2atlabel.cpu().numpy())

    # change label with de-normalization
    label_list = np.concatenate(label_list, axis=0).flatten()
    output_list = np.concatenate(output_list, axis=0).flatten()
    if flag == "vo2max":
        label_list = label_list * 10 + 23
        output_list = output_list * 10 + 23
    else:
        label_list = label_list * 10 + 15
        output_list = output_list * 10 + 15

    mae = np.mean(np.abs(label_list - output_list))
    mape = np.mean(np.abs((label_list - output_list) / (label_list)))

    print(f"MAE: {mae:.2f}, MAPE: {mape:.2f}")
