import numpy as np
import pickle
import os
import pandas as pd
from lifelines.statistics import logrank_test

model = 'PathVAE'



def cal_mid(l: list):
    data = l[:]
    data.sort()
    half = len(data) // 2

    return (data[half] + data[~half]) / 2


# for cancer in ['brca', 'kirc', 'lusc', 'luad', 'blca', 'lihc']:
for cancer in ['brca', 'kirc', 'luad', 'blca']:
    high_risk = {'event_time': [], 'censorship': [], 'risk': []}
    low_risk = {'event_time': [], 'censorship': [], 'risk': []}
    for i in range(5):
        with open(os.path.join(f'./result_test/{model}_{cancer}', f'split_latest_val_{i}_results.pkl'), 'rb') as f:
            result = pickle.load(f)

        risk = result['risk'].tolist()
        censor = result['censor'].tolist()
        event_time = result['event_time'].tolist()
        patient_id = list(range(len(risk)))

        mid = cal_mid(risk)
        for n in range(len(risk)):
            if risk[n] < mid:
                low_risk['censorship'].append(censor[n])
                low_risk['event_time'].append(event_time[n])
                low_risk['risk'].append(risk[n])
            else:
                high_risk['censorship'].append(censor[n])
                high_risk['event_time'].append(event_time[n])
                high_risk['risk'].append(risk[n])

        df = pd.DataFrame({'censorship': list(low_risk['censorship']) + list(high_risk['censorship']),
                           'event_time': list(low_risk['event_time']) + list(high_risk['event_time']),
                           'risk': list(low_risk['risk'] + list(high_risk['risk']))})
        df['group'] = ['low'] * len(low_risk['censorship']) + ['high'] * len(high_risk['censorship'])
        df.to_csv(f'./result_test/{model}_{cancer}.csv')

        df['group'] = [0] * len(low_risk['event_time']) + [1] * len(high_risk['event_time'])
        results = logrank_test(df['event_time'], df['group'], df['censorship'])
        # results.print_summary()
        print(results.p_value)