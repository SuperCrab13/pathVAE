import pandas as pd
import os
import numpy as np

signatures = pd.read_csv(r'D:\PathVAE\data\hallmarks_signatures.csv')
gene_list = []
for k in signatures.keys():
    gene_list.extend(signatures[k].to_list())
gene_array = np.array(gene_list)
clean_gene = gene_array[~(gene_array == 'nan')]
gene_list = list(set(clean_gene))

# gene_list = []
# with open(os.path.join(r'D:\pro\tcga\gbmlgg', 'gene_list.txt'), encoding='utf-8') as f:
#     for i in f.readlines():
#         gene_list.append(i.strip('\n'))
# f.close()


from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd


def data_label(slide_data):
    patients_df = slide_data.drop_duplicates(['case_id']).copy()
    uncensored_df = patients_df[patients_df['censorship'] < 1]

    label_col = 'survival_months'
    uncensored_df = uncensored_df.drop_duplicates()
    disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=4, retbins=True, labels=False)
    q_bins[-1] = slide_data[label_col].max() + 1e-6
    q_bins[0] = slide_data[label_col].min() - 1e-6

    disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False,
                                 include_lowest=True)
    patients_df.insert(2, 'label', disc_labels.values.astype(int))

    slide_data = patients_df
    slide_data.reset_index(drop=True, inplace=True)
    slide_data = slide_data.assign(slide_id=slide_data['case_id'])

    label_dict = {}
    key_count = 0
    for i in range(len(q_bins) - 1):
        for c in [0, 1]:
            print('{} : {}'.format((i, c), key_count))
            label_dict.update({(i, c): key_count})
            key_count += 1

    for i in slide_data.index:
        key = slide_data.loc[i, 'label']
        slide_data.at[i, 'disc_label'] = key
        censorship = slide_data.loc[i, 'censorship']
        key = (key, int(censorship))
        slide_data.at[i, 'label'] = label_dict[key]

    return slide_data


data_path = r'D:\pro\tcga\rnaseq'
for file in os.listdir(data_path):
    print(file)
    if file.endswith('gz'):
        if not 'kirc' in file:
            continue
        data = pd.read_csv(os.path.join(data_path, file), sep='\t', index_col='sample')
        data = data.transpose()
        gene_select = list(set(data.keys()) & set(gene_list))
        data = data[gene_select]
        cancer = file.split('.')[1]
        survival_file = pd.read_csv(
            os.path.join(data_path, f'{cancer.lower()}_tcga_pan_can_atlas_2018_clinical_data.tsv'), sep='\t',
            index_col='Sample ID')
        # if "BRCA" in file:
        #     survival_file = survival_file.loc[survival_file['Oncotree Code'] == 'IDC']
        col = ['Overall Survival (Months)', 'Overall Survival Status']
        survival_file = survival_file[col]
        survival_file.columns = ['survival_months', 'Overall Survival Status']
        censorship = []
        for i in survival_file['Overall Survival Status'].to_list():
            if i == '0:LIVING':
                censorship.append(1)
            else:
                censorship.append(0)
        survival_file['censorship'] = censorship
        survival_file.drop('Overall Survival Status', axis=1, inplace=True)
        data = pd.concat([survival_file, data], axis=1, join='inner').dropna()

        sample_list = data.index.to_list()
        for i in range(len(sample_list)):
            sample_list[i] = sample_list[i][:-3]
        data['case_id'] = sample_list
        data.set_index('case_id', inplace=True)

        data.to_csv(os.path.join(data_path, 'data', f'{cancer}_rnaseq.csv'), index=True)

        data['case_id'] = data.index.to_list()
        data = data_label(data)
        if not os.path.exists(os.path.join(data_path, 'split', 'tcga_' + cancer)):
            os.mkdir(os.path.join(data_path, 'split', 'tcga_' + cancer))
        case = data['case_id'].to_list()
        label = data['label'].to_list()
        # print(data['label'])
        split = 0
        sfolder = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
        # sfolder = KFold(random_state=2, shuffle=True)
        for train, test in sfolder.split(case, label):
            train_list = []
            test_list = []
            for i in train:
                train_list.append(case[i])
            for j in test:
                test_list.append(case[j])
            df = pd.DataFrame()
            test_list = test_list + [0] * (len(train_list) - len(test_list))
            df['train'] = train_list
            df['val'] = test_list
            df.to_csv(os.path.join(data_path, 'split', 'tcga_'+cancer.lower(), 'splits_{}.csv'.format(split)))
            split += 1
    else:
        continue
