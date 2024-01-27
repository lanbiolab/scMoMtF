import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import random
from torch.autograd import Variable
import pandas as pd
import scanpy as sc
import os
import torch.nn.functional as F

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def real_label(index, classify_dim):
    output_v = []
    label_num = pd.Categorical(index).codes
    for i in range(classify_dim):
        temp = index[np.array(np.where(label_num == i)[0][0])]
        output_v.append(temp)
    return output_v


def Totensor(X_train, X_test, Y_train, Y_test):
    X_train = torch.from_numpy(X_train)
    X_train = Variable(X_train.type(FloatTensor))
    X_test = torch.from_numpy(X_test)
    X_test = Variable(X_test.type(FloatTensor))
    Y_train = torch.from_numpy(Y_train)
    Y_train = Y_train.type(LongTensor)
    Y_test = torch.from_numpy(Y_test)
    Y_test = Y_test.type(LongTensor)
    return X_train, X_test, Y_train, Y_test


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):  # 返回的是tensor
        img, target = self.data[index, :], self.label[index]
        sample = {'data': img, 'label': target}
        return sample

    def __len__(self):
        return len(self.data)


def save_checkpoint(state, save):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'model_best.pth.tar')
    torch.save(state, filename)


def get_encodings(model, dl):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    encodings = []
    ori_data = []
    label = []
    with torch.no_grad():
        for i, batch_sample in enumerate(dl):
            x = batch_sample['data']
            x = Variable(x.type(FloatTensor))
            x = torch.reshape(x, (x.size(0), -1))
            valid_label = batch_sample['label']
            valid_label = Variable(valid_label.type(LongTensor))

            x_prime = model.encoder(x.to(device))
            encodings.append(x_prime)
            ori_data.append(x)
            label.append(valid_label)

    return torch.cat(encodings, dim=0), torch.cat(ori_data, dim=0), torch.cat(label, dim=0)


def get_decodings(model, dl):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    decodings = []
    ori_data = []
    with torch.no_grad():
        for i, batch_sample in enumerate(dl):
            x = batch_sample['data']
            x = Variable(x.type(FloatTensor))
            x = torch.reshape(x, (x.size(0), -1))
            x_prime, x_cty, mu, var = model(x.to(device))
            decodings.append(x_prime)
            ori_data.append(x)
    return torch.cat(decodings, dim=0), torch.cat(ori_data, dim=0)


def get_simulation(model, dl):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    decodings = []
    ori_data = []
    with torch.no_grad():
        for i, batch_sample in enumerate(dl):
            x = batch_sample['data']
            x = Variable(x.type(FloatTensor))
            x = torch.reshape(x, (x.size(0), -1))
            x_prime, h, x_r, x_d, x_g, x_cluster, mu, var = model(x.to(device))
            decodings.append(x_prime)
            ori_data.append(x)
    return torch.cat(decodings, dim=0), torch.cat(ori_data, dim=0)


def normalize(adata):
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)

    return adata


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes=17, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def KL_loss(mu, logvar):
    KLD = -0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())
    return KLD


class BaseLoss:
    eps = 1e-9

    def __init__(self, n_clusters):
        self.n_output = n_clusters
        self.weight = 1

    @staticmethod
    def compute_distance(output, target):

        return F.mse_loss(output, target)


class ReconstructionLoss(BaseLoss):
    def __init__(self, config):
        super().__init__(config['n_clusters'])
        loss_weight = config['train_loss_weight']
        if loss_weight is not None:
            self.weight = loss_weight
        else:
            self.weight = 1

    def __call__(self, reconstruction, raw_data):

        loss = BaseLoss.compute_distance(
            reconstruction,
            raw_data,
        )

        loss *= self.weight
        return loss


class DiscriminatorLoss(BaseLoss):

    def __init__(self, config):
        super().__init__(config['n_clusters'])
        loss_weight = config['train_loss_weight']
        if loss_weight is not None:
            self.weight = loss_weight
        else:
            self.weight = 0.1

    def __call__(self, reconstruction, raw_data):
        loss = 0
        loss += F.mse_loss(raw_data, torch.ones_like(raw_data, device='cuda:0'))
        loss += F.mse_loss(reconstruction, torch.zeros_like(reconstruction, device='cuda:0'))
        loss *= self.weight
        return loss


class GeneratorLoss(BaseLoss):

    def __init__(self, config):
        super().__init__(config['n_clusters'])
        loss_weight = config['train_loss_weight']
        if loss_weight is not None:
            self.weight = loss_weight
        else:
            self.weight = 0.1

    def __call__(self, generator_output):
        loss = F.mse_loss(generator_output, torch.ones_like(generator_output, device='cuda:0'))
        loss *= self.weight
        return loss




