import pickle

from dataset import OmicDataset, get_loader
from timeit import default_timer as timer
import click
import pandas as pd
import os
from utils import train_val, save_pkl
from model_util import SNN, PorpoiseAMIL, DNN, Coxnnet
import warnings
import torch
import numpy as np
from PathVAE import PathVAE

warnings.filterwarnings("ignore")


def seed_torch(seed=7):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# @click.command()
# @click.option('--batch_size', default=32, type=int)
# @click.option('--lr', default=.0004)
# @click.option('--cancer', type=str, default='blca')
# @click.option('--lam_l2', type=float, default=1e-5)
# @click.option('--epoch', type=int, default=10)
# @click.option('--lam_reg', type=float, default=1e-4)
# @click.option('--alpha_surv', type=float, default=0.)
# @click.option('--result_dir', type=str, default='./result_test')
# @click.option('--model_type', type=str, default='PathVAE')
def main(batch_size=32, lr=4e-4, cancer='lihc', lam_l2=1e-5, epoch=10, lam_reg=1e-4, alpha_surv=0, result_dir='result_ablation', model_type="PathVAE"):
    seed_torch(1)
    mode = 'pathway' if model_type == "AMIL" or model_type == "PathVAE" else 'normal'
    data = pd.read_csv(os.path.join('./data_test/', f'{cancer}_rnaseq.csv'))
    with open('./random_geneset.pkl', 'rb') as f:
        pathway = pickle.load(f)
    pathway = pd.DataFrame(pathway)

    train_args = {'lr': lr, 'lam_l2': lam_l2, 'epoch': epoch, 'lam_reg': lam_reg, 'alpha_surv': alpha_surv}
    latest_val_cindex = []
    result_dir = os.path.join(result_dir, model_type + '_' + cancer)

    if not os.path.exists(result_dir):
        print("Result directory not exist, creating.....")
        os.mkdir(result_dir)
    for split in range(5):
        start = timer()
        results_pkl_path = os.path.join(result_dir, 'split_latest_val_{}_results.pkl'.format(split))
        print(f"####### Split {split} #######")
        split_file = pd.read_csv(os.path.join('./split_test', f'tcga_{cancer.upper()}/splits_{split}.csv'))
        train_set = OmicDataset(data=data, split=split_file, train_val='train', pathway_df=pathway, mode=mode)
        val_set = OmicDataset(data=data, split=split_file, train_val='val', pathway_df=pathway, mode=mode)
        print(f"Train set len {len(train_set)}, Val set len {len(val_set)}")
        train_loader = get_loader(train_set, batch_size)
        val_loader = get_loader(val_set, batch_size)

        if model_type == 'SNN':
            model = SNN(omic_input_dim=len(train_set.omic_feature))
        elif model_type == 'AMIL':
            model = PorpoiseAMIL(input_feature=train_set.omic_dict)
        elif model_type == 'DNN':
            model = DNN(omic_input_dim=len(train_set.omic_feature))
        elif model_type == 'Cox-nnet':
            model = Coxnnet(input_dim=len(train_set.omic_feature))
        elif model_type == 'PathVAE':
            model = PathVAE(input_feature=train_set.omic_dict, p_mask=0.5)
        else:
            raise NotImplementedError

        results_val_dict, val_cindex, model = train_val(train_loader, val_loader, train_args, model, split=split)
        latest_val_cindex.append(val_cindex)
        save_pkl(results_pkl_path, results_val_dict)
        torch.save(model.state_dict(), os.path.join(result_dir, "s_{}_checkpoint.pt".format(split)))
        end = timer()
        print('Fold %d Time: %f seconds' % (split, end - start))
    results_latest_df = pd.DataFrame({'folds': list(range(5)), 'val_cindex': latest_val_cindex})
    results_latest_df.to_csv(os.path.join(result_dir, 'summary_latest.csv'))


if __name__ == '__main__':
    #for method in ['DNN', 'AMIL', 'SNN']:
    for cancer in ['brca', 'kirc', 'lusc', 'luad', 'blca']:
    #         main(model_type=method, cancer=cancer)

        main(model_type='PathVAE', cancer=cancer, batch_size=64, lr=1e-3, epoch=10)
