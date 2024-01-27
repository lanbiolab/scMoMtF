import torch
import torch.nn as nn
from torch.autograd import Variable


def classification_module(model, dl, real_label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval()
    classified_label = []
    groundtruth_label = []
    prob = []
    with torch.no_grad():
        for i, batch_sample in enumerate(dl):
            # load data
            x = batch_sample['data']
            x = Variable(x)
            x = torch.reshape(x, (x.size(0), -1))
            test_label = batch_sample['label']
            test_label = Variable(test_label)
            # classification
            x_prime, h, x_r, x_d, x_g, x_cluster, mu, var = model(x.to(device))
            a = torch.max(nn.Softmax()(x_cluster), 1)
            for j in range(x_prime.size(0)):
                classified_label.append(real_label[a.indices[j]])
                groundtruth_label.append(real_label[test_label[j]])
                prob.append(a.values[j])

    return classified_label, groundtruth_label, prob
