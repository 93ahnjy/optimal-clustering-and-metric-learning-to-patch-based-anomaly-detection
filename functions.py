import numpy as np
import torch
import torch.nn as nn
import cv2
import torchvision

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import umap  # pip install umap-learn
from mpl_toolkits.mplot3d import Axes3D



def plot_embedding(X, labels, n_cluster, savedir, mode='UMAP', add_anomaly=False):

    from matplotlib import pyplot as plt

    plt.figure(figsize=(7, 7))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple', 'gray', 'brown', 'gold', 'navy', 'magenta', 'silver', 'b'
    markers = ['o'] * n_cluster

    if add_anomaly:
        markers[-1] = 'x'


    # if   mode == 'TSNE': X_embedded = TSNE(n_components=3, perplexity=10).fit_transform(X)
    # elif mode == 'UMAP': X_embedded = umap.UMAP().fit_transform(X)
    #
    # fig = plt.figure()
    # ax1 = fig.add_subplot(221, projection='3d')
    # ax2 = fig.add_subplot(222, projection='3d')


    # ax2.view_init(30, 90)
    #
    # for i, c, in zip(range(n_cluster), colors):
    #     ax1.scatter(X_embedded[labels == i, 0], X_embedded[labels == i, 1], X_embedded[labels == i, 2], label=i, c=colors[i], marker=markers[i])
    #     ax2.scatter(X_embedded[labels == i, 0], X_embedded[labels == i, 1], X_embedded[labels == i, 2], label=i, c=colors[i], marker=markers[i])


    if   mode == 'TSNE': X_embedded = TSNE(n_components=2, perplexity=10).fit_transform(X)
    elif mode == 'UMAP': X_embedded = umap.UMAP().fit_transform(X)


    for i, c, in zip(range(n_cluster), colors):
        plt.scatter(X_embedded[labels == i, 0], X_embedded[labels == i, 1], label=i, c=colors[i], marker=markers[i])






    plt.legend(bbox_to_anchor=(0.5, 0), ncol=5)
    plt.savefig(savedir, dpi=300)
    plt.close()
    plt.clf()
    # plt.show()
















def init_patch_dict(mode='bg32'):

    new_dict = {}

    # Save images & feature to dict
    new_dict[mode + '_p']     = []
    new_dict[mode + '_p_pos'] = []


    new_dict[mode + '_h']     = []
    new_dict[mode + '_h_pos'] = []
    new_dict[mode + '_h_n'] = []
    new_dict[mode + '_h_a'] = []

    return new_dict









def save_muliple_patches_and_features(patches, features, num, save_dict, mode='bg32'):

    # convert patch images to numpy
    p      = patches[0][:num].cpu().numpy()
    p_pos  = patches[1][:num].cpu().numpy()

    h      = features[0][:num].detach().squeeze(2).squeeze(2)
    h_pos  = features[1][:num].detach().squeeze(2).squeeze(2)
    h_n    = features[2][:num].detach()
    h_a    = features[3][:num].detach()


    # Normalize images
    p = (p - np.min(p)) / np.ptp(p)
    p_pos = (p_pos - np.min(p_pos)) / np.ptp(p_pos)


    # Save images to dict
    save_dict[mode + '_p'].append(p)
    save_dict[mode + '_p_pos'].append(p_pos)

    # Save features to dict
    save_dict[mode + '_h'].append(h)
    save_dict[mode + '_h_pos'].append(h_pos)
    save_dict[mode + '_h_n'].append(h_n)
    save_dict[mode + '_h_a'].append(h_a)









































######################################################################################################







def Split_feature_by_kmeans(data_dict, batch_idx, class_idx, grid_col, K=32, min_cluster=2, max_cluster=16):

    p_concat  = np.concatenate(data_dict[f'obj{K}_p'], axis=0)                     # (n, 3, 32, 32)
    h_concat  = torch.cat(data_dict[f'obj{K}_h'], dim=0)             # (2n, 64)


    """ 2. n_cluster 개수를 바꾸어 가며 kmeans 를 수행하고 각 경우에 따라 silhouette score 가 언제 최소인지 확인 """
    X = h_concat.detach().cpu().numpy()
    max_score, n_clusters_opt = get_num_of_cluster(X, min_cluster, max_cluster, class_idx, K)


    cluster = MiniBatchKMeans(n_clusters=n_clusters_opt, batch_size=10).fit(X)
    labels  = cluster.labels_
    centers = cluster.cluster_centers_

    p_grid_list = []
    h_list = []

    for i in range(n_clusters_opt):
        p_i = torch.as_tensor(p_concat[labels == i])
        try: p_i_grid = torchvision.utils.make_grid(p_i, nrow=grid_col, padding=6).permute(1, 2, 0).numpy()
        except: continue
        p_grid_list.append(p_i_grid)

        h_i = h_concat[labels==i]
        h_list.append(h_i.mean(0))



    try:
        p_grid = np.concatenate(p_grid_list, axis=0)
        cv2.imwrite(f"result_patch/Result{class_idx}/labeled{K}_img_{class_idx}_{batch_idx}.jpg", p_grid * 255)
    except:
        print("Skip concatenate due to input array dimensions error. ")
        pass


    return max_score, centers, (X, labels)














































def Split_embs_by_kmeans(patches, embs, epoch_idx, class_idx, grid_col, K=32, max_cluster=16, grid_max=30):

    max_score, n_clusters_opt = get_num_of_cluster(embs, max_cluster, class_idx, K)
    p_grid_list, labels, centers = split_patches_and_features(embs, n_clusters_opt, patches=patches, features=embs, grid_col=grid_col, grid_max=grid_max)

    try:
        p_grid = np.concatenate(p_grid_list, axis=0)
        cv2.imwrite(f"result_patch/Result{class_idx}/labeled{K}_img_{class_idx}_{epoch_idx}.jpg", p_grid * 255)
    except:
        print("Skip concatenate due to input array dimensions error. ")
        pass

    return centers, labels



















def get_num_of_cluster(X, min_cluster, max_cluster, class_idx, K):

    """ Clustering patch,feature with Kmeans """
    import matplotlib.pyplot as plt


    """ Get optimal number of cluster """
    """ k-means clustering 평균 시간 복잡도는 O(NKDI)가 된다.      N : 데이터벡터수, K : cluster 수, D : 벡터 차원크기. I : iter   """
    inertia = []
    num_of_iters = []
    sil_scores = []



    """ n_cluster 개수를 바꾸어 가며 kmeans 를 수행하고 각 경우에 따라 silhouette score 가 언제 최소인지 확인 """
    iter_N = range(min_cluster, max_cluster+1)
    for n in iter_N:
        cluster = MiniBatchKMeans(n_clusters=n, batch_size=10).fit(X)
        inertia.append(cluster.inertia_)
        num_of_iters.append(cluster.n_iter_)
        sil_scores.append(silhouette_score(X, cluster.labels_, metric='euclidean'))


    inertia_grad = [inertia[i] - inertia[i+2] for i in range(len(inertia) - 2)]









    fig, ax = plt.subplots()
    for i, txt in enumerate(num_of_iters):
        ax.annotate(txt, (list(iter_N)[i], inertia[i]))


    ax.plot(iter_N, sil_scores, 'bx-')
    ax.set_xlabel('num_of_cluster')
    ax.set_ylabel('Silhouette score')
    ax.set_title('Silhouette score For Optimal k')
    plt.savefig(f"result_patch/Silhouette score_{class_idx}_{K}")
    plt.cla()
    plt.close(fig)
    max_score = max(sil_scores)
    n_clusters_opt = sil_scores.index(max(sil_scores)) + min_cluster
    print(f"**** K={K}  Max score : {max_score: .3f},  n_cluster : {n_clusters_opt}") # 시작이 n=2 부터.


    return max_score, n_clusters_opt








def split_patches_and_features(X, n_cluster, patches, features, grid_col, grid_max=None):
    # cluster = KMeans(n_clusters=n_clusters_opt)
    kmeans = MiniBatchKMeans(n_clusters=n_cluster, batch_size=100).fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    p_grid_list = []
    h_list = []




    for i in range(n_cluster):
        p_i = patches[labels == i]

        p_i = (p_i - np.min(p_i)) / np.ptp(p_i)

        p_i = torch.as_tensor(p_i)
        index = np.random.choice(len(p_i), min(grid_max, len(p_i)), replace=False)
        p_i = p_i[index]

        p_i_grid = torchvision.utils.make_grid(p_i, nrow=grid_col, padding=6).permute(1, 2, 0).numpy()
        p_grid_list.append(p_i_grid)

        h_i = features[labels == i]
        h_list.append(h_i)



    return p_grid_list, labels, centers
















