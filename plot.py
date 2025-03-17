import matplotlib.pyplot as plt
import PyComplexHeatmap as pyh
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')

cancer = 'blca'
data = pd.read_csv(f'./result_test/PathVAE_{cancer}/attn.csv')
data = data.sort_values('group')
metadata = data[['group', 'risk']]

data = data.drop(['group', 'risk'], axis=1)
mean_data = data.mean(0).sort_values()
# data = data[mean_data.index[-10:]]
print(data)

col_ha = pyh.HeatmapAnnotation(label=pyh.anno_label(metadata['group'], merge=True, rotation=45),
                               group=pyh.anno_simple(metadata['group']),
                               label_side='bottom', axis=0)
plt.figure(figsize=(7, 4))
cm = pyh.ClusterMapPlotter(data=data, left_annotation=col_ha, standard_scale=1,
                           show_rownames=False, show_colnames=True, col_names_side='head',
                           row_split=metadata['group'],  # label='AUC',
                           cmap='exp1',
                           legend=True)
plt.savefig(f"clustermap_{cancer}.pdf", bbox_inches='tight')
plt.show()

