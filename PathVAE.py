from model_util import Attn_Net_Gated, init_max_weights
import torch.nn as nn
import torch
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_feature, out_feature, hidden_dim=512, dropout=0.25):
        super(Encoder, self).__init__()
        self.layer = nn.Sequential(*[nn.Linear(input_feature, hidden_dim),
                                     nn.BatchNorm1d(num_features=hidden_dim),
                                     nn.LeakyReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.BatchNorm1d(num_features=hidden_dim),
                                     nn.LeakyReLU(),
                                     nn.Dropout(dropout)
                                     ])
        self.mu = nn.Linear(hidden_dim, out_feature)
        self.logstd = nn.Linear(hidden_dim, out_feature)

    def reparameterize(self, mu, logstd):
        if self.training:
            std = torch.exp(logstd)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        feat = self.layer(x)
        mu = self.mu(feat)
        logstd = self.logstd(feat)

        return mu, logstd


class Decoder(nn.Module):
    def __init__(self, input_feature, out_feature, hidden_dim, dropout):
        super(Decoder, self).__init__()
        self.layer = nn.Sequential(*[nn.Linear(input_feature, hidden_dim),
                                     nn.BatchNorm1d(num_features=hidden_dim),
                                     nn.LeakyReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.BatchNorm1d(num_features=hidden_dim),
                                     nn.LeakyReLU(),
                                     nn.Dropout(dropout)
                                     ])
        self.outlayer = nn.Sequential(*[nn.Linear(hidden_dim, out_feature),
                                        nn.Dropout(dropout)
                                        ])

    def forward(self, x):
        feat = self.layer(x)
        return self.outlayer(feat)


class PathVAE(nn.Module):
    def __init__(self, input_feature, cls_num=4, hidden_dim=128, p_mask=0.5):
        super(PathVAE, self).__init__()
        self.input_layer = []
        self.output_layer = []
        for i, k in enumerate(input_feature.keys()):
            self.input_layer.append(Encoder(input_feature[k], out_feature=hidden_dim, hidden_dim=256, dropout=0.2))
            self.output_layer.append(Decoder(hidden_dim, input_feature[k], hidden_dim=256, dropout=0.2))
        self.input_layer = nn.ModuleList(self.input_layer)
        self.output_layer = nn.ModuleList(self.output_layer)
        fc = []
        attention_net = Attn_Net_Gated(L=hidden_dim, D=hidden_dim, dropout=0.2, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        self.classifier = nn.Linear(hidden_dim, cls_num)
        self.mu = None
        self.logstd = None
        self.fusion_feature = None
        self.input_feature = input_feature

        self.p_mask = p_mask
        self.path_mask = torch.randint(0, len(self.input_feature), (int(self.p_mask * len(self.input_feature)),))

    @property
    def path_mask(self):
        return self._mask

    @path_mask.setter
    def path_mask(self, mask):
        self._mask = mask

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.input_layer = nn.DataParallel(self.input_layer, device_ids=device_ids).to('cuda: 0')
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')
            self.output_layer = nn.DataParallel(self.output_layer, device_ids=device_ids).to('cuda: 0')
        else:
            self.attention_net = self.attention_net.to(device)
            self.input_layer = self.input_layer.to(device)
            self.output_layer = self.output_layer.to(device)

        self.classifier = self.classifier.to(device)


    def forward(self, data, attention_only=False):
        feat = []
        mu_list = []
        logstd_list = []
        for i in range(len(data)):
            if i in self.path_mask and self.training:
                data[i] = torch.zeros_like(data[i], device=data[i].device, dtype=data[i].dtype)
            mu, logstd = self.input_layer[i](data[i])
            mu_list.append(mu)
            logstd_list.append(logstd)
            feat.append(self.input_layer[i].reparameterize(mu, logstd).unsqueeze(1))

        self.mu = mu_list
        self.logstd = logstd_list
        h = torch.concat(feat, dim=1)  # b*n*h

        A, h = self.attention_net(h)

        if attention_only:
            return A

        A_raw = A
        A = F.softmax(A, dim=1)
        A = torch.transpose(A, 2, 1)
        M = torch.bmm(A, h).reshape([h.shape[0], -1])
        self.fusion_feature = M
        h = self.classifier(M)

        return h

    def get_loss(self, x):
        kl = []
        rec = []
        for i in range(len(x)):
            mu = self.mu[i]
            logstd = self.logstd[i]
            kl.append(torch.mean(-0.5 * torch.sum(1 + logstd - mu**2 - torch.exp(logstd), 1), 0))
            rec.append(F.mse_loss(self.output_layer[i](self.fusion_feature), x[i]))
            # rec.append(F.mse_loss(self.output_layer[i](self.input_layer[i].reparameterize(mu, logstd)), x[i]))

        return (sum(kl) + sum(rec)) / len(x)

class PathVAE_ablation(nn.Module):
    def __init__(self, input_feature, cls_num=4, hidden_dim=128, p_mask=0.5):
        super(PathVAE_ablation, self).__init__()
        self.input_layer = []
        self.output_layer = []
        for i, k in enumerate(input_feature.keys()):
            self.input_layer.append(Encoder(input_feature[k], out_feature=hidden_dim, hidden_dim=256, dropout=0.2))
            self.output_layer.append(Decoder(hidden_dim, input_feature[k], hidden_dim=256, dropout=0.2))
        self.input_layer = nn.ModuleList(self.input_layer)
        self.output_layer = nn.ModuleList(self.output_layer)

        self.fc = nn.Sequential(*[nn.Linear(hidden_dim*len(input_feature), hidden_dim), nn.LeakyReLU()])

        self.classifier = nn.Linear(hidden_dim, cls_num)
        self.mu = None
        self.logstd = None
        self.fusion_feature = None
        self.input_feature = input_feature

        self.p_mask = p_mask
        self.path_mask = torch.randint(0, len(self.input_feature), (int(self.p_mask * len(self.input_feature)),))

    @property
    def path_mask(self):
        return self._mask

    @path_mask.setter
    def path_mask(self, mask):
        self._mask = mask

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.input_layer = nn.DataParallel(self.input_layer, device_ids=device_ids).to('cuda: 0')
            self.fc = nn.DataParallel(self.fc, device_ids=device_ids).to('cuda:0')
            self.output_layer = nn.DataParallel(self.output_layer, device_ids=device_ids).to('cuda: 0')
        else:
            self.fc = self.fc.to(device)
            self.input_layer = self.input_layer.to(device)
            self.output_layer = self.output_layer.to(device)

        self.classifier = self.classifier.to(device)


    def forward(self, data):
        feat = []
        mu_list = []
        logstd_list = []
        for i in range(len(data)):
            if i in self.path_mask and self.training:
                data[i] = torch.zeros_like(data[i], device=data[i].device, dtype=data[i].dtype)
            mu, logstd = self.input_layer[i](data[i])
            mu_list.append(mu)
            logstd_list.append(logstd)
            feat.append(self.input_layer[i].reparameterize(mu, logstd))

        self.mu = mu_list
        self.logstd = logstd_list
        h = torch.concat(feat, dim=1)  # b*n*h
        M = self.fc(h)
        # A, h = self.attention_net(h)

        # if attention_only:
        #     return A
        #
        # A_raw = A
        # A = F.softmax(A, dim=1)
        # A = torch.transpose(A, 2, 1)
        # M = torch.bmm(A, h).reshape([h.shape[0], -1])
        self.fusion_feature = M
        h = self.classifier(M)

        return h

    def get_loss(self, x):
        kl = []
        rec = []
        for i in range(len(x)):
            mu = self.mu[i]
            logstd = self.logstd[i]
            kl.append(torch.mean(-0.5 * torch.sum(1 + logstd - mu**2 - torch.exp(logstd), 1), 0))
            rec.append(F.mse_loss(self.output_layer[i](self.fusion_feature), x[i]))
            # rec.append(F.mse_loss(self.output_layer[i](self.input_layer[i].reparameterize(mu, logstd)), x[i]))

        return (sum(kl) + sum(rec)) / len(x)

