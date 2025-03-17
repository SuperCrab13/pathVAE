import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OmicDataset(Dataset):
    def __init__(self, data, split: pd.DataFrame, label_col='survival_months',
                 train_val='train', pathway_df=None, mode='pathway'):
        super(OmicDataset, self).__init__()
        self.mode = mode
        self.train_val = train_val
        self.data = data
        patients_df = self.data.copy()
        uncensored_df = patients_df[patients_df['censorship'] < 1]

        eps = 1e-7
        disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=4, retbins=True, labels=False)
        q_bins[-1] = self.data[label_col].max() + eps
        q_bins[0] = self.data[label_col].min() - eps
        self.label_col = label_col if label_col else 'event_time'

        disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False,
                                     include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))

        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)

        label_dict = {}
        key_count = 0
        for i in range(len(q_bins) - 1):
            for c in [0, 1]:
                # print('{} : {}'.format((i, c), key_count))
                label_dict.update({(i, c): key_count})
                key_count += 1

        self.label_dict = label_dict
        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = key
            censorship = slide_data.loc[i, 'censorship']
            key = (key, int(censorship))
            slide_data.at[i, 'label'] = label_dict[key]

        self.bins = q_bins
        self.num_classes = len(self.label_dict)
        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {'case_id': patients_df['case_id'].values, 'label': patients_df['label'].values}

        new_cols = list(slide_data.columns[-1:]) + list(slide_data.columns[:-1])  ### PORPOISE
        slide_data = slide_data[new_cols]
        self.slide_data = slide_data
        metadata = ['disc_label', 'case_id',
                    'survival_months', 'label','censorship']
        self.metadata = slide_data.columns[:5]
        self.omic_feature = list(slide_data.drop(self.metadata, axis=1).keys())

        assert self.metadata.equals(pd.Index(metadata))

        self.case_id = split[self.train_val]
        self.data = slide_data.loc[slide_data['case_id'].isin(self.case_id)]
        self.genomic_data = self.data.drop(self.metadata, axis=1)
        self.apply_scaler(self.get_scaler())

        if self.mode == 'pathway':
            self.omic_names = []
            for col in pathway_df.columns:
                omic = pathway_df[col].dropna().unique()
                omic = list(set(omic) & set(self.genomic_data.keys()))
                if len(omic) > 0:
                    self.omic_names.append(omic)
            self.omic_sizes = [len(omic) for omic in self.omic_names]
            self.num_pathways = len(self.omic_sizes)
            self.omic_dict = dict(zip(pathway_df.columns, self.omic_sizes))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        label = self.slide_data['disc_label'][item]
        event_time = self.slide_data[self.label_col][item]
        c = self.slide_data['censorship'][item]

        if self.mode == 'pathway':
            omic_list = []
            for i in range(self.num_pathways):
                omic_list.append(torch.tensor(self.genomic_data[self.omic_names[i]].iloc[item]).unsqueeze(0))

            return omic_list, label, event_time, c

        else:
            omic = torch.tensor([self.genomic_data.iloc[item]])
            return omic, label, event_time, c

    def get_scaler(self) -> StandardScaler:
        print(self.data.shape)
        scaler = StandardScaler()
        scaler.fit(self.genomic_data)

        return scaler

    def apply_scaler(self, scaler):
        transformed_omic = scaler.transform(self.genomic_data)
        transformed = pd.DataFrame(transformed_omic)
        transformed.columns = self.genomic_data.columns
        self.genomic_data = transformed


def collate_survival(batch):
    rna = torch.cat([item[0] for item in batch], dim=0).type(torch.FloatTensor)
    label = torch.tensor([item[1] for item in batch])
    event_time = torch.tensor([item[2] for item in batch])
    c = torch.tensor([item[3] for item in batch])

    return [rna, label, event_time, c]


def collate_pathway(batch):
    omic_data_list = []
    omic_num = len(batch[0][0])
    for omic_id in range(omic_num):
        omic_data_list.append([item[0][omic_id].to(torch.float32) for item in batch])

    label = torch.tensor([item[1] for item in batch])
    event_time = torch.tensor([item[2] for item in batch])
    c = torch.tensor([item[3] for item in batch])

    return [omic_data_list, label, event_time, c]


def get_loader(dataset, batch_size):
    kwargs = {'num_workers': 0} if device.type == "cuda" else {}
    if dataset.mode != 'pathway':
        loader = DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(dataset),
                            collate_fn=collate_survival, drop_last=False, **kwargs)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(dataset),
                            collate_fn=collate_pathway, drop_last=False, **kwargs)
    return loader
