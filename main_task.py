import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from model.predict import classification_module
from model.util import setup_seed, get_encodings, get_simulation
from sklearn import metrics
from sklearn.cluster import KMeans
import scanpy as sc
import torch


def classification(args, model, dl, real_label):
    # set random to reproduce the result
    setup_seed(args.seed)
    if args.query:
        path = "query"
    else:
        path = "reference"
    if not os.path.exists('output/classification/{}/{}'.format(args.dataset, path)):
        os.makedirs('output/classification/{}/{}'.format(args.dataset, path))
    save_path = open('output/classification/{}/{}/accuracy_each_cell.txt'.format(args.dataset, path), "w")
    classified_label, groundtruth_label, prob = classification_module(model, dl, real_label)
    t = 0
    for j in range(len(groundtruth_label)):
        if classified_label[j] == groundtruth_label[j]:
            t = t + 1
        print('cell ID: ', j, '\t', '\t', 'real cell type:', groundtruth_label[j], '\t', '\t', 'predicted cell type:',
              classified_label[j], '\t', '\t', 'probability:', round(float(prob[j]), 2), file=save_path)

    accuracy = t / len(groundtruth_label)
    print(accuracy)
    print("finish classification")
    return accuracy


def dim_reduce(args, model, dl, real_label):
    if args.query:
        path = "query"
    else:
        path = "reference"
    encoding_data, ori_data, label = get_encodings(model, dl)
    # eliminate outliers
    if not os.path.exists('output/dim_reduce/{}/{}/'.format(args.dataset, path)):
        os.makedirs('output/dim_reduce/{}/{}/'.format(args.dataset, path))

    b_list = range(0, encoding_data.size(1))
    feature_index = ['feature_{}'.format(b) for b in b_list]
    b_list = range(0, ori_data.size(0))
    cell_name_real = ['cell_{}'.format(b) for b in b_list]

    real_label_new = []
    for j in range(ori_data.size(0)):
        real_label_new.append(real_label[label[j]])
    # visualization
    adata_all = sc.AnnData(encoding_data.cpu().numpy())
    adata_all.obs["label"] = real_label_new
    sc.pp.neighbors(adata_all)
    sc.tl.umap(adata_all, min_dist=0.2)
    fig, ax = plt.subplots(figsize=(9, 6), tight_layout=True)
    sc.pl.umap(adata_all, color=['label'], frameon=True, palette='tab20',
               ax=ax)
    # calculate metrics
    kmeans = KMeans(args.n_clusters, n_init=30)
    y_pred = kmeans.fit_predict(encoding_data.data.cpu().numpy())
    label = label.cpu().numpy()
    ami = np.round(metrics.adjusted_mutual_info_score(label, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(label, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(label, y_pred), 5)
    print('Final: AMI= %.4f, NMI= %.4f, ARI= %.4f' % (ami, nmi, ari))

    pd.DataFrame(encoding_data.cpu().numpy(), index=cell_name_real, columns=feature_index).to_csv(
        'output/dim_reduce/{}/{}/latent_space.csv'.format(args.dataset, path))
    pd.DataFrame(real_label_new, index=cell_name_real, columns=["label"]).to_csv(
        'output/dim_reduce/{}/{}/latent_space_label.csv'.format(args.dataset, path))
    print("finish dimension reduction")
    return [ami, nmi, ari]


def simulation(args, model, dl, nfeatures_1, nfeatures_2):
    if args.query:
        path = "query"
    else:
        path = "reference"
    sim_data, real_data = get_simulation(model, dl)
    # eliminate outliers
    sim_data[sim_data > torch.max(real_data)] = torch.max(real_data)
    sim_data[sim_data < torch.min(real_data)] = torch.min(real_data)
    sim_data[torch.isnan(sim_data)] = torch.max(real_data)
    sim_modality1_data = sim_data[:, 0:nfeatures_1]
    sim_modality2_data = sim_data[:, nfeatures_1:nfeatures_1+nfeatures_2]
    b_list = range(0, sim_data.size(0))
    cell_name_real = ['cell_{}'.format(b) for b in b_list]
    if not os.path.exists('output/simulation/{}/{}/'.format(args.dataset, path)):
        os.makedirs('output/simulation/{}/{}/'.format(args.dataset, path))
    pd.DataFrame(sim_modality1_data.cpu().numpy(), index=cell_name_real).to_csv(
        'output/simulation/{}/{}/sim_modality1_data.csv'.format(args.dataset, path))
    pd.DataFrame(sim_modality2_data.cpu().numpy(), index=cell_name_real).to_csv(
        'output/simulation/{}/{}/sim_modality2_data.csv'.format(args.dataset, path))
    print("finish simulation")
