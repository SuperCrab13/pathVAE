import torch
import torch.nn as nn
import torch.nn.functional as F


class full_block(nn.Module):
    def __init__(self, in_features, out_features, p_drop=0.2):
        super(full_block, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=p_drop)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        feat = self.fc(x)
        feat = self.bn(feat)
        feat = self.act(feat)
        feat = self.dropout(feat)

        return feat


class SNN_Block(nn.Module):
    def __init__(self, dim1, dim2, dropout=0.25):
        super(SNN_Block, self).__init__()
        activation = nn.SELU()
        alpha_dropout = nn.AlphaDropout(dropout)
        self.net = nn.Sequential(nn.Linear(dim1, dim2), activation, alpha_dropout)
        for param in self.net.parameters():
            if len(param.shape) == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="linear")

    def forward(self, data):
        return self.net(data)


def init_max_weights(module):
    r"""
    Initialize Weights function.

    args:
        modules (torch.nn.Module): Initalize weight using normal distribution
    """
    import math
    import torch.nn as nn

    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


##########################
#### Genomic FC Model ####
##########################
class SNN(nn.Module):
    def __init__(self, omic_input_dim: int, model_size_omic: str = 'small', n_classes: int = 4):
        super(SNN, self).__init__()
        self.n_classes = n_classes
        self.size_dict_omic = {'small': [256, 256, 256, 256], 'big': [2048, 1024, 512, 256]}

        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        fc_omic = [SNN_Block(dim1=omic_input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
        self.fc_omic = nn.Sequential(*fc_omic)
        self.classifier = nn.Linear(hidden[-1], n_classes)
        init_max_weights(self)

    def forward(self, data):
        # x = torch.concat(data, dim=1)
        x = data
        h_omic = self.fc_omic(x)
        h = self.classifier(h_omic)  # logits needs to be a [B x 4] vector
        assert len(h.shape) == 2 and h.shape[1] == self.n_classes

        return h

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.fc_omic = nn.DataParallel(self.fc_omic, device_ids=device_ids).to('cuda:0')
        else:
            self.fc_omic = self.fc_omic.to(device)

        self.classifier = self.classifier.to(device)


class DNN(nn.Module):
    def __init__(self, omic_input_dim: int, model_size_omic: str = 'small', n_classes: int = 4):
        super(DNN, self).__init__()
        self.n_classes = n_classes
        self.size_dict_omic = {'small': [256, 256, 256, 256], 'big': [2048, 1024, 512, 256]}

        ### Constructing Genomic DNN
        hidden = self.size_dict_omic[model_size_omic]
        fc_omic = [full_block(in_features=omic_input_dim, out_features=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(full_block(in_features=hidden[i], out_features=hidden[i + 1], p_drop=0.25))
        self.fc_omic = nn.Sequential(*fc_omic)
        self.classifier = nn.Linear(hidden[-1], n_classes)
        init_max_weights(self)

    def forward(self, data):
        # x = torch.concat(data, dim=1)
        x = data
        h_omic = self.fc_omic(x)
        h = self.classifier(h_omic)  # logits needs to be a [B x 4] vector
        assert len(h.shape) == 2 and h.shape[1] == self.n_classes

        return h

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.fc_omic = nn.DataParallel(self.fc_omic, device_ids=device_ids).to('cuda:0')
        else:
            self.fc_omic = self.fc_omic.to(device)

        self.classifier = self.classifier.to(device)


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes (experimental usage for multiclass MIL)
"""


class Attn_Net_Gated(nn.Module):

    def __init__(self, L=1024, D=256, dropout=0.2, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class PorpoiseAMIL(nn.Module):
    def __init__(self, input_feature, size_arg="small", n_classes=4):
        super(PorpoiseAMIL, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        self.size = size
        self.input_layer = []
        for i, k in enumerate(input_feature.keys()):
            self.input_layer.append(nn.Sequential(*[nn.Linear(input_feature[k], size[1]), nn.ReLU(), nn.Dropout(0.25)]))
        self.input_layer = nn.ModuleList(self.input_layer)

        fc = [nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(0.25)]
        attention_net = Attn_Net_Gated(L=size[2], D=size[2], dropout=0.25, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        self.classifier = nn.Linear(size[2], n_classes)
        init_max_weights(self)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.input_layer = nn.DataParallel(self.input_layer, device_ids=device_ids).to('cuda: 0')
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')
        else:
            self.attention_net = self.attention_net.to(device)
            self.input_layer = self.input_layer.to(device)

        self.classifier = self.classifier.to(device)

    def forward(self, data, attention_only=False):
        feat = []
        for i in range(len(data)):
            feat.append(self.input_layer[i](data[i]).unsqueeze(1))
        h = torch.concat(feat, dim=1)  # b*n*h

        A, h = self.attention_net(h)
        # A = torch.transpose(A, 1, 0)

        if attention_only:
            return A

        A_raw = A
        A = F.softmax(A, dim=1)
        A = torch.transpose(A, 2, 1)
        M = torch.bmm(A, h).reshape([h.shape[0], -1])
        h = self.classifier(M)

        return h


class Coxnnet(nn.Module):
    def __init__(self, input_dim, n_hidden=256):
        super(Coxnnet, self).__init__()
        self.fc1 = nn.Linear(input_dim, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1, bias=False)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.fc1(x))
        return self.fc2(x)



