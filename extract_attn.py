import pickle
import pandas as pd
import torch
from PathVAE import PathVAE
from dataset import OmicDataset, get_loader
import os
import numpy as np
import warnings


def cal_mid(l: list):
    data = l[:]
    data.sort()
    half = len(data) // 2

    return (data[half] + data[~half]) / 2


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    cancer = 'blca'
    pathway = pd.read_csv('./data/hallmarks_signatures.csv')
    data_df = pd.read_csv(os.path.join('./data_test/', f'{cancer}_rnaseq.csv'))
    attn_list = []
    all_group_list = []
    all_risk_list = []
    for split in range(5):
        split_file = pd.read_csv(os.path.join('./split_test', f'tcga_{cancer.upper()}/splits_{split}.csv'))
        train_set = OmicDataset(data=data_df, split=split_file, train_val='train', pathway_df=pathway)
        val_set = OmicDataset(data=data_df, split=split_file, train_val='val', pathway_df=pathway)
        model_param = torch.load(f'./result_test/PathVAE_{cancer}/s_{split}_checkpoint.pt')
        model = PathVAE(input_feature=train_set.omic_dict, p_mask=0.5)
        model.load_state_dict(model_param)
        model.relocate()
        model.train()
        val_loader = get_loader(val_set, 32)

        risk_list = []
        group_list = []
        with torch.no_grad():
            for data in val_loader:
                omic_data, label, event_time, c = data
                for i in range(len(omic_data)):
                    omic_data[i] = torch.concat(omic_data[i], dim=0).to('cuda')
                attn = model(omic_data, attention_only=True)
                result = model(omic_data)
                hazards = torch.sigmoid(result)
                survival = torch.cumprod(1 - hazards, dim=1)
                risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
                attn_list.append(attn.softmax(1))
                risk_list.extend(risk.tolist())

        mid_risk = cal_mid(risk_list)
        for r in risk_list:
            if r > mid_risk:
                group_list.append('high_risk')
            else:
                group_list.append('low_risk')
        all_group_list.extend(group_list)
        all_risk_list.extend(risk_list)

    attn = torch.concat(attn_list)
    attn = attn.reshape(attn.shape[0], 1, -1)
    attn_dict = {}
    for i, k in enumerate(train_set.omic_dict.keys()):
        attn_dict.update({k: attn[:, :, i].reshape(-1).tolist()})
    df = pd.DataFrame(attn_dict)
    df['group'] = all_group_list
    df['risk'] = all_risk_list
    df.to_csv(f'./result_test/PathVAE_{cancer}/attn.csv', index=False)
