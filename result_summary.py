import pandas as pd
import os

df_list = []
for method in ['Cox-nnet', 'AMIL', 'PathVAE', 'SNN', 'DNN']:
    result_dict = {}
    for cancer in ['brca', 'kirp', 'lusc', 'luad', 'blca', 'lihc']:
        data = pd.read_csv(f'./result_test/{method}_{cancer}/summary_latest.csv')
        result_dict[cancer] = data['val_cindex'].tolist()
    df = pd.DataFrame(result_dict)
    df['method'] = [method]*data.shape[0]
    print(df)
    df_list.append(df)
all_result = pd.concat(df_list, axis=0)
all_result.to_csv('./all_result.csv')