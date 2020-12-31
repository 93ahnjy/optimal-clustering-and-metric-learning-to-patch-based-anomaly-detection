
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, MiniBatchKMeans





def Deep_SVDD_loss(h, center, R, nu=0.1):

    h_dist = (h - center).norm(dim=1)
    h_R = np.quantile((h_dist.clone().detach().cpu().numpy()), 1 - nu)
    h_R = max(h_R, R)

    h_R = torch.tensor(h_R).cuda()

    scores32 = h_dist ** 2 - h_R ** 2
    loss_R_32 = h_R ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores32), scores32))

    return loss_R_32, h_R





''' ######################## Texture class 에 대해서 사용 ########################################'''

def quantile(inputs, ratio, descending=False):
    return torch.sort(inputs, dim=0, descending=descending)[0][int(len(inputs)*ratio)-1]





def Distill_feature(h, n_cluster, ro_n=0.666, ro_h=0.333):

    num, D = h.shape
    h_hard = []

    if n_cluster > 1:
        X = h.data.cpu().numpy().squeeze()

        kmeans = KMeans(n_clusters=n_cluster).fit(X)
        labels  = torch.as_tensor(kmeans.labels_)
        centers = torch.as_tensor(kmeans.cluster_centers_).cuda()


        for i in range(len(centers)):
            h_i = h[labels == i].view(-1, D)
            h_dist = ((h_i - centers[i]) ** 2).sum(dim=1)
            h_dist_srt, idx_srt = torch.sort(h_dist, dim=0)

            idx_n = int(len(h_dist) * ro_n) - 1
            idx_hard_n = int(len(h_dist) * ro_n * (1 - ro_h)) - 1

            h_i_hard = h[idx_srt][idx_hard_n:idx_n,:]
            h_hard.append(h_i_hard)

    else:
        h_dist = ((h - h.mean(0)) ** 2).sum(dim=1)
        h_dist_srt, idx_srt = torch.sort(h_dist, dim=0)

        idx_n      = int(len(h_dist) * ro_n) - 1
        idx_hard_n = int(len(h_dist) * ro_n * (1 - ro_h)) - 1

        h_hard = h[idx_srt][idx_hard_n:idx_n,:]


    return h_hard









def DeepCluster_loss(h, n_cluster, is_obj, p):

    X = h.clone().detach().cpu().numpy().squeeze()

    if n_cluster <= 3 : n_cluster = 4
    kmeans = KMeans(n_clusters=n_cluster).fit(X)
    labels = kmeans.labels_

    h_nn_loss  = 0
    sign = -1 if is_obj else 1

    centers = []
    cen = 0
    for i in range(n_cluster):
        h_i     = h[labels == i]
        n       = int(len(h_i)*p)

        h_i_cen = h_i.mean(0)
        centers.append(h_i_cen)

        h_i_dist = torch.norm(h_i - h_i_cen.detach(), dim=1)
        h_i_dist = torch.sort(h_i_dist, descending=True)[0]
        h_i_dist[n:] = 0

        h_nn_loss += h_i_dist.mean(0)
        cen += (h_i_cen / n_cluster)


    centers = torch.stack(centers)
    h_cen_loss = torch.cdist(centers, centers).mean() * sign


    return h_nn_loss + h_cen_loss





def DeepCluster_loss_na(h_n, h_a, p):

    n_elem, D = h_n.shape

    h_na_dist = torch.cdist(h_n, h_a)


    h_na_dist = h_na_dist.view(-1, n_elem)
    h_na_dist = torch.sort(h_na_dist, descending=False)[0]

    n = int(len(h_na_dist) * p)
    h_na_dist[n:] = 0
    return h_na_dist.mean()























def silhouette_score_loss(h, center, is_obj):

    X = h.clone().detach().cpu().numpy().squeeze()

    #print(X.shape, center.shape)

    kmeans = KMeans(n_clusters=len(center), init=center, n_init=1).fit(X)
    labels = kmeans.labels_

    loss = 0
    sign = 1 if is_obj else -1

    for i in range(len(center)):

        cluster_i = h[labels == i]
        cluster_j = h[labels != i]
        num_i = len(cluster_i)

        intra_dist = torch.cdist(cluster_i, cluster_i.mean(0)).sum(dim=1)
        inter_dist = torch.cdist(cluster_i, cluster_j).min(dim=1)[0]

        sil_score = (inter_dist - intra_dist) / (torch.max(intra_dist, inter_dist) + 1)
        #sil_score = torch.where(sil_score < 0.5, sil_score, torch.zeros_like(sil_score).cuda())

        loss -= sil_score.mean() * sign

    return loss

















# def DeepCluster_loss3(h, n_cluster, ro_n=0.666, ro_h=0.333):
#
#     h_distill, sil_score = Distill_feature(h, n_cluster, ro_n=ro_n)
#
#     loss = 0
#     centers = []
#     for h_i in h_distill:
#
#         num, D = h_i.shape
#         dists = h_i - h_i.mean(0)
#
#         covar = torch.t(dists).mm(dists) / num
#         mahalanobis_distances = torch.diagonal(torch.sqrt(dists.view(-1, D) @ torch.inverse(covar) @ dists.view(D,-1))).mean()
#         #mahalanobis_distances = F.relu(magnitude + torch.t(magnitude) - 2 * squared_matrix).sqrt().mean()
#
#         print(mahalanobis_distances.mean())
#
#         loss += mahalanobis_distances
#
#
#
#     return loss / len(h_distill)
























''' #############################################################################################'''








def Deep_dist_loss(h, n_cluster, centers=None):
    ''' Deep SVDD - soft radius 는 다음과 같다. '''


    X = h.clone().detach().cpu().numpy().squeeze()

    if centers is None:
        kmeans = KMeans(n_clusters=n_cluster).fit(X)
        labels = torch.as_tensor(kmeans.labels_)
        centers = torch.as_tensor(kmeans.cluster_centers_).cuda()


    else:
        kmeans = KMeans(n_clusters=len(centers), init=centers, n_init=1).fit(X)
        labels = torch.as_tensor(kmeans.labels_)
        centers = torch.as_tensor(centers).cuda()


    intra_loss = 0
    inter_loss = 0
    for i in range(n_cluster):

        h_i = F.softmax(h[labels == i], dim=1)

        ''' Decrease intra-class distance'''
        intra_dist = (h_i - h_i.mean(0)).norm(dim=1)


        try:
            idx_max = torch.argmax(intra_dist)
            intra_loss += intra_dist[idx_max] / n_cluster
        except:
            continue



        ''' Increase inter-class distance'''
        neg_cen = torch.cat([centers[0:i], centers[i + 1:]])
        for j in range(len(neg_cen)):
            h_j= F.softmax(h[labels == j], dim=1)
            h_dist = torch.cdist(h_i, h_j)
            inter_dist_min = torch.min(h_dist)
            inter_loss += inter_dist_min / len(h_j) / n_cluster


    #intra_loss = 0
    triplet_loss = max(intra_loss - inter_loss + 0.1 , 0)

    return triplet_loss, intra_loss, inter_loss

# triplet_loss32, intra32_loss, inter32_loss = Deep_dist_loss(h32, self.n_cluster32, self.center32)
# triplet_loss64, intra64_loss, inter64_loss = Deep_dist_loss(h64, self.n_cluster64, self.center64)
# loss += triplet_loss32 #+ triplet_loss64 # + loss_R_32*0.2 #+ loss_R_64*0.2