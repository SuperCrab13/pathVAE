#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from model_util import init_max_weights
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
from lifelines.utils import concordance_index

# In[239]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ### Model Architecture

# In[240]:


class Coxnnet(nn.Module):
    def __init__(self, input_dim, n_hidden, dropout=0.1):
        super(Coxnnet, self).__init__()
        self.fc1 = nn.Linear(input_dim, n_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(n_hidden, 1, bias=False)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.fc1(x))
        return self.fc2(x)


# ### helper functions

# In[241]:


class Coxnnet_Dataset(Dataset):
    def __init__(self, data, survival_time, event):
        self.data = torch.from_numpy(data).float()  # Convert data to torch tensor
        self.survival_time = torch.from_numpy(survival_time).float()
        self.event = torch.from_numpy(event).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        survival_time = self.survival_time[idx]
        event = self.event[idx]
        return sample, survival_time, event


def cox_ph_loss(risks, time, event):
    """
    Compute Cox proportional hazards loss per sample
    :param risks: The output from the Cox-PH layer (log hazard ratios).
    :param time: Observed times (either event or censoring time).
    :param event: Event indicator (1 if event occurred, 0 if censored).
    :return: Negative log partial likelihood.
    """
    # Sort the individuals by descending survival time
    eps = 1e-100
    time = time + eps * torch.ones(len(time))  # ensure all times are positive
    risk_order = torch.argsort(time, descending=True)
    risks = risks[risk_order]
    event = event[risk_order]

    # Compute the risk score exp(log hazard ratio) for all
    hazard_ratios = torch.exp(risks)

    # Calculate the cumulative hazard at each actual event
    log_cumulative_hazard = torch.log(torch.cumsum(hazard_ratios, dim=0))
    observed_log_hazard = risks - log_cumulative_hazard

    # Only include the cases where an event occurred
    uncensored_likelihood = observed_log_hazard * event
    return -torch.sum(uncensored_likelihood) / torch.sum(event)


def negative_log_likelihood(theta, ystatus):
    exp_theta = torch.exp(theta.squeeze(-1))
    risk = torch.cumsum(exp_theta.flip(dims=[0]), dim=0).flip(dims=[0])
    log_risk = torch.log(risk)
    log_likelihood = (theta - log_risk) * ystatus
    return -torch.mean(log_likelihood)


def Xavier_unif_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def scale(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    scaled_matrix = (matrix - min_val) / (max_val - min_val)
    return scaled_matrix


# ### Train and test

# In[242]:


def train_cox_model(model, train_dataloader, valid_dataloader, learning_rate, l2_penalty, epochs, min_delta=1e-4,
                    patience=10):
    # adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_penalty)
    # early stopping vars
    is_best_model = 0
    patient_epoch = 0
    c = 0

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_dataloader:
            X_batch, time, event = batch
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = cox_ph_loss(outputs, time, event)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)

        if epoch % 1 == 0:
            model.eval()
            val_loss = 0
            y_pred = np.array([])
            time = np.array([])
            event = np.array([])
            with torch.no_grad():
                for batch in valid_dataloader:
                    X_val, time_val, event_val = batch
                    val_outputs = model(X_val)
                    val_loss += cox_ph_loss(val_outputs, time_val, event_val).item()
                    val_outputs, time_val, event_val = torch.squeeze(val_outputs), torch.squeeze(
                        time_val), torch.squeeze(event_val)
                    y_pred = np.append(y_pred, np.array(val_outputs))
                    time = np.append(time, np.array(time_val))
                    event = np.append(event, np.array(event_val))

            avg_val_loss = val_loss / len(valid_dataloader)
            concordance = concordance_index(time, -y_pred, event_observed=event)
            if concordance > c:
                c = concordance
            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.10f}, Val Loss: {avg_val_loss:.10f}, C-index: {concordance:.10f}")

        # Early Stopping
        if epoch == 0:
            is_best_model = 1
            best_model = model
            min_loss_epoch_valid = 10000.0
            if avg_val_loss < min_loss_epoch_valid:
                min_loss_epoch_valid = avg_val_loss
        else:
            if min_loss_epoch_valid - avg_val_loss > min_delta:
                is_best_model = 1
                best_model = model
                min_loss_epoch_valid = avg_val_loss
                patient_epoch = 0
            else:
                is_best_model = 0
                patient_epoch += 1
                if patient_epoch >= patience:
                    print('Early Stopped at Epoch:', epoch)
                    break
        print("is best model:", is_best_model)
    print("max validation c-index: ", c)
    return best_model, [time, y_pred, event]


# In[243]:


def Test_Model(model, test_dataloader):
    '''
    Test the model
    '''
    output_last = True
    losses_ppl = []
    cindex_test = []
    print('start test')

    with torch.no_grad():
        for inputs, survival_time, event in test_dataloader:
            inputs = inputs.float().to(device)
            event = event.float().to(device)
            survival_time = survival_time.float().to(device)
            outputs = model(inputs)

            outputs_squeezed, survival_time_squeezed, event_squeezed = torch.squeeze(outputs), torch.squeeze(
                survival_time), torch.squeeze(event)
            loss_ppl = cox_ph_loss(outputs_squeezed, survival_time_squeezed, event_squeezed)

            cindex = concordance_index(survival_time, -outputs, event_observed=event)
            # print(torch.squeeze(event), torch.squeeze(outputs))

            losses_ppl.append(loss_ppl.data)
            cindex_test.append(cindex)

        print(cindex_test)
        print('Tested: mean_loss:{}, std_loss:{}, cindex:{}, std_cindex:{}'.format(np.mean(losses_ppl),
                                                                                   np.std(losses_ppl),
                                                                                   np.mean(cindex_test),
                                                                                   np.std(cindex_test)))

    return [losses_ppl, cindex_test]


# In[244]:


home = "/home/path/"
folder = "/folder/path/"
i = "3"
X_train_val = np.loadtxt(home + folder + "x_train" + i + ".csv", skiprows=1, delimiter=",")
y_train_val = np.loadtxt(home + folder + "ytime_train" + i + ".csv", skiprows=1, delimiter=",")
event_train_val = np.loadtxt(home + folder + "ystatus_train" + i + ".csv", skiprows=1, delimiter=",")
X_test = np.loadtxt(home + folder + "x_test" + i + ".csv", skiprows=1, delimiter=",")
y_test = np.loadtxt(home + folder + "ytime_test" + i + ".csv", skiprows=1, delimiter=",")
event_test = np.loadtxt(home + folder + "ystatus_test" + i + ".csv", skiprows=1, delimiter=",")

# In[245]:


from sklearn.model_selection import train_test_split

random_state = 20
X_train, X_val, y_train, y_val, event_train, event_val = train_test_split(X_train_val, y_train_val, event_train_val,
                                                                          test_size=0.20, random_state=random_state)

# In[ ]:


if __name__ == "__main__":
    BATCH_SIZE = 32
    num_epochs = 10
    patience = 10
    min_delta = 0.0001
    learning_rate = 0.0004
    dropout = 0.2
    l2_penalty = 1e-4

    # device config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = Coxnnet_Dataset(X_train, y_train, event_train)
    valid_dataset = Coxnnet_Dataset(X_val, y_val, event_val)
    test_dataset = Coxnnet_Dataset(X_test, y_test, event_test)

    # Create data loaders for our datasets; shuffle for training, not for validation
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    inputs, survival, event = next(iter(train_dataloader))
    [batch_size, fea_size] = inputs.size()
    hidden_size = np.ceil(np.sqrt(fea_size)).astype(int)
    print(hidden_size)

    model = Coxnnet(input_dim=fea_size, n_hidden=hidden_size)
    # model.apply(Xavier_unif_init)
    best_model, val_pred = train_cox_model(model, train_dataloader, valid_dataloader, learning_rate, l2_penalty,
                                           epochs=num_epochs, min_delta=min_delta, patience=patience)

# In[ ]:


[losses_ppl, cindex_test] = Test_Model(best_model, test_loader)

# In[122]:


np.savetxt(home + folder + "pytorch_coxnnet_cv_cindex" + i + ".csv", cindex_test)
