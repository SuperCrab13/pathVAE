import torch
import os
import PathVAE
from loss_util import NLLSurvLoss, cox_ph_loss
from sksurv.metrics import concordance_index_censored
import numpy as np
import pickle
import tqdm
from model_util import Coxnnet


def save_pkl(filename, save_object):
    writer = open(filename, 'wb')
    pickle.dump(save_object, writer)
    writer.close()


def l1_reg_all(model):
    l1_reg = None

    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum()  # torch.abs(W).sum() is equivalent to W.norm(1)
    return l1_reg


class Monitor_CIndex:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.best_score = None

    def __call__(self, val_cindex, model, ckpt_name: str = 'checkpoint.pt'):

        score = val_cindex

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), ckpt_name)


def train_val(train_loader, val_loader, train_args, model, split):
    if split == 2:
        print('split 2')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device(device))
    model.train()
    # opt = torch.optim.Adam(lr=train_args['lr'], weight_decay=train_args['lam_l2'], params=model.parameters())
    opt = torch.optim.RMSprop(lr=train_args['lr'], weight_decay=train_args['lam_l2'], params=model.parameters())
    if type(model) == Coxnnet:
        loss_fn = cox_ph_loss
    else:
        loss_fn = NLLSurvLoss(alpha=train_args['alpha_surv'])
    sche = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=4, gamma=0.1)
    best_result = {}
    best_cindex = 0.
    best_model = None
    for i in range(train_args['epoch']):
        train_split(train_loader, train_args, model, i, loss_fn, opt, device)
        patient_results, c_index = val_split(val_loader, model)
        if c_index > best_cindex:
            best_result = patient_results
            best_cindex = c_index
            best_model = model
        # sche.step()
        if hasattr(model, 'path_mask'):
            model.path_mask = torch.randint(0, len(model.input_feature), (int(model.p_mask * len(model.input_feature)),))
        print(f'Best c-index: {best_cindex}')

    return best_result, best_cindex, best_model


def train_split(train_loader, train_args, model, epoch, loss_fn, opt, device):
    max_mem = 0
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    for batch_id, data in enumerate(tqdm.tqdm(train_loader)):
        opt.zero_grad()
        omic_data, label, event_time, c = data
        if type(omic_data) != torch.Tensor:
            for i in range(len(omic_data)):
                omic_data[i] = torch.concat(omic_data[i], dim=0).to(device)
        else:
            omic_data = omic_data.to(device)
        label = label.to(device)
        event_time = event_time.to(device)
        c = c.to(device)
        pred = model(omic_data)
        if type(model) == Coxnnet:
            loss_surv = loss_fn(risks=pred, time=event_time, event=c)
            risk = pred.detach().cpu().reshape(-1)
        else:
            loss_surv = loss_fn(h=pred, y=label, t=event_time, c=c)
            hazards = torch.sigmoid(pred)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        reg = l1_reg_all(model)
        loss_reg = reg * train_args['lam_reg']
        loss = loss_surv + loss_reg
        if hasattr(model, 'get_loss'):
            loss += model.get_loss(omic_data)

        # if (batch_id + 1) % 2 == 0:
        #     print('batch {}, loss: {:.4f}'.format(batch_id, loss + loss_reg))

        loss.backward()
        if torch.isnan(loss):
            print('loss NAN')
        opt.step()

        all_risk_scores.append(risk)
        all_censorships.append(c.detach().cpu().numpy())
        all_event_times.append(event_time.detach().cpu().numpy())

        if max_mem < torch.cuda.memory_allocated():
            max_mem = torch.cuda.memory_allocated()

    all_risk_scores = np.concatenate(all_risk_scores)
    all_censorships = np.concatenate(all_censorships)
    all_event_times = np.concatenate(all_event_times)
    c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times,
                                         all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {} train_c_index:, {:.4f}'.format(epoch, c_index))
    print('max memory: {:.4f}Mb'.format(max_mem / 1024 / 1024))


@torch.no_grad()
def val_split(loader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)

    patient_results = {}
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    for batch_id, data in enumerate(loader):
        omic_data, label, event_time, c = data
        if type(omic_data) != torch.Tensor:
            for i in range(len(omic_data)):
                omic_data[i] = torch.concat(omic_data[i], dim=0).to(device)
        else:
            omic_data = omic_data.to(device)
        pred = model(omic_data)

        if pred.shape[1] > 1:
            hazards = torch.sigmoid(pred)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        else:
            risk = pred.detach().cpu().numpy().reshape(-1)

        all_risk_scores.append(risk)
        all_censorships.append(c)
        all_event_times.append(event_time)

    all_risk_scores = np.concatenate(all_risk_scores)
    all_censorships = np.concatenate(all_censorships)
    all_event_times = np.concatenate(all_event_times)
    patient_results['risk'] = all_risk_scores
    patient_results['censor'] = all_censorships
    patient_results['event_time'] = all_event_times

    c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times,
                                         all_risk_scores, tied_tol=1e-08)[0]

    print('Val_c_index: {:.4f}'.format(c_index))

    return patient_results, c_index
