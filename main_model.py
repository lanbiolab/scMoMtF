import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from model.model import scMoMtF
from model.train import train_model
from model.util import setup_seed, MyDataset, normalize, real_label, Totensor
from main_task import classification, dim_reduce, simulation
from sklearn.model_selection import StratifiedKFold
from config import *


def model_name_RC(annData_1, annData_2, dataset):
    args = args_rc
    if dataset == 'CITE':
        args.hidden_2 = 50
    # set random to reproduce the result
    setup_seed(args.seed)
    cuda = True if torch.cuda.is_available() else False
    args.device = "cuda:0" if cuda else "cpu"
    args.dataset = dataset
    print(args)
    index = np.array(annData_1.obs['label'])
    train_label = pd.Categorical(index).codes
    # normalize
    adata1 = normalize(annData_1)
    adata2 = normalize(annData_2)

    classify_dim = max(train_label) + 1
    args.n_clusters = classify_dim
    nfeatures_1 = adata1.X.shape[1]
    nfeatures_2 = adata2.X.shape[1]
    train_data = np.concatenate((adata1.X, adata2.X), axis=1)
    k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    temp_1 = []
    temp_2 = []
    # five-fold cross-validation
    for train_index, test_index in k_fold.split(train_data, train_label):
        X_train, X_test = train_data[train_index], train_data[test_index]
        Y_train, Y_test = train_label[train_index], train_label[test_index]
        X_train, X_test, Y_train, Y_test = Totensor(X_train, X_test, Y_train, Y_test)
        train_dataset = MyDataset(X_train, Y_train)
        test_dataset = MyDataset(X_test, Y_test)
        train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              drop_last=False)
        test_dl = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                             drop_last=False)

        print("The dataset is", args.dataset)
        model_save_path = "trained_model/{}/".format(args.dataset)
        transform_real_label = real_label(index, classify_dim)
        # define model
        model = scMoMtF(args=args, nfeatures_modality1=nfeatures_1, nfeatures_modality2=nfeatures_2,
                        hidden_modality1=args.hidden_1, hidden_modality2=args.hidden_2, z_dim=args.z_dim,
                        classify_dim=classify_dim)

        # use gpu
        model = model.cuda()
        # train model
        model = train_model(args, model, train_dl, lr=args.lr, epochs=args.epochs,
                            classify_dim=classify_dim, save_path=model_save_path)
        # task testing
        if args.classification:
            accuracy = classification(args, model, test_dl, transform_real_label)

        if args.dim_reduce:
            metrics = dim_reduce(args, model, test_dl, transform_real_label)

        if args.simulation:
            simulation(args, model, test_dl, nfeatures_1, nfeatures_2)
        temp_1.append(accuracy)
        temp_2.append(metrics)
    average_accuracy = np.mean(temp_1)
    average_metrics = np.mean(np.array(temp_2), axis=0)
    print(average_accuracy, average_metrics)
