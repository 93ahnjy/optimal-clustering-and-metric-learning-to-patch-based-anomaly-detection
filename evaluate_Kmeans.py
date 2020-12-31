import numpy as np
import os, glob
import torch

import shutil
import cv2
import time


from Patch_SVDD import *
from evaluate import *
from Dataset import MVtec_Stride_Patch_Dataset

from sklearn.metrics import silhouette_score
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans, MiniBatchKMeans







""" ################################## eval 주요 함수.################################################# """




def eval_encoder_Kmeans2(enc, dataset32, dataset64, class_idx, n_clusters=None, cluster_centers=None, sel_center='kmeans'):

    dataset32_tr, dataset32_te = dataset32
    dataset64_tr, dataset64_te = dataset64

    embs64_tr  = infer_(dataset64_tr, enc)      # K=64, S=16
    embs64_te  = infer_(dataset64_te, enc)      # K=64, S=16

    embs32_tr  = infer_(dataset32_tr, enc.enc)      # K=32, S=4
    embs32_te  = infer_(dataset32_te, enc.enc)      # K=32, S=4


    ''' n_clusters, cluster_centers 둘 중 하나는 반드시 입력이 존재해야 함.'''
    if cluster_centers:
        center32, center64 = cluster_centers
        cluster_elem64, cluster_cen64 = cluster_emb_tr(embs64_tr, cluster_centers=center64, sel_center=sel_center)
        cluster_elem32, cluster_cen32 = cluster_emb_tr(embs32_tr, cluster_centers=center32, sel_center=sel_center)
        n_clusters32, n_clusters64 = len(cluster_cen32), len(cluster_cen64)

    else:
        n_clusters32, n_clusters64 = n_clusters
        cluster_elem64, cluster_cen64 = cluster_emb_tr(embs64_tr, n_clusters=n_clusters64, sel_center=sel_center)
        cluster_elem32, cluster_cen32 = cluster_emb_tr(embs32_tr, n_clusters=n_clusters32, sel_center=sel_center)


    cluster32 = (cluster_elem32, cluster_cen32, embs32_te)
    cluster64 = (cluster_elem64, cluster_cen64, embs64_te)


    return eval_embeddings_Kmeans(class_idx, cluster32, cluster64, n_clusters32, n_clusters64)










def eval_embeddings_Kmeans(class_idx, cluster32, cluster64, n_clusters32, n_clusters64):

    cluster_elem64, cluster_cen64, embs64_te = cluster64
    cluster_elem32, cluster_cen32, embs32_te = cluster32

    start1 = time.time()
    print(f"\n ** Searching NN in K=64. Cluster : {len(cluster_cen64)}")
    maps_64 = search_NN_ngt(embs64_te, cluster_elem64, cluster_cen64, NN=1)
    maps_64 = distribute_scores(maps_64, (256, 256), K=64, S=16)         # N 개의 anomaly map 생성.
    print(f" Time elapsed : {time.time() - start1: .3f} sec\n\n")


    start2 = time.time()
    print(f"\n ** Search NN in K=32. Cluster : {len(cluster_cen32)}")
    maps_32 = search_NN_ngt(embs32_te, cluster_elem32, cluster_cen32, NN=1)
    maps_32 = distribute_scores(maps_32, (256, 256), K=32, S=4)         # N 개의 anomaly map 생성.
    print(f" Time elapsed : {time.time() - start2: .3f} sec\n\n")



    print(f" Total Time elapsed : {time.time() - start1: .3f} sec")
    print(f" Number of image -  test : {embs32_te.shape[0]}")
    print(f" Speed : {(time.time() - start1)/embs32_te.shape[0]: .3f} sec / img\n\n")


    det_32,   seg_32   = assess_anomaly_maps(class_idx, maps_32)
    det_mult, seg_mult = assess_anomaly_maps(class_idx, maps_64 * maps_32)

    if det_32 + seg_32 >= det_mult + seg_mult:
        maps_sel = maps_32
    else:
        maps_sel = maps_64 * maps_32


    return {
        'anomaly_maps' : maps_sel,

        'det_32': det_32,
        'seg_32': seg_32,

        'det_mult': det_mult,
        'seg_mult': seg_mult,
    }













""" ################################## Infer 관련 함수들.################################################# """



def infer_(dataset, enc):

    N, H, W, D = dataset.total_img, dataset.emb_row, dataset.emb_col, enc.D

    enc  = enc.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=320, shuffle=False, pin_memory=True)

    hs_list = []
    with torch.no_grad():
        for xs, ns, iis, js in dataloader:
            xs = xs.cuda()
            hs = enc(xs)
            hs_list.append(hs.squeeze())

    embs = torch.cat(hs_list).view(N, H, W, D)
    embs = embs.cpu().numpy()


    #print("EMBS m, std : ", embs.mean(), embs.std())
    return embs






def infer_with_images(dataset, enc, K, num_of_images=1):

    N, H, W, D = dataset.total_img, dataset.emb_row, dataset.emb_col, enc.D
    enc  = enc.eval()

    xs_list = []
    hs_list = []
    index = np.random.choice(range(N), num_of_images, replace=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=H*W, shuffle=False, pin_memory=True)

    with torch.no_grad():
        for i, (xs, ns, _, _) in enumerate(dataloader):

            if int(ns[0]) in index:
                odd_idx = range(1, len(xs), 2)
                xs = xs[odd_idx]

                hs = enc(xs.cuda())
                hs_list.append(hs.squeeze())
                xs_list.append(xs)

    xs   = torch.cat(xs_list).view(len(odd_idx) * num_of_images, 3, K, K)
    embs = torch.cat(hs_list).view(len(odd_idx) * num_of_images, D)

    xs = xs.cpu().numpy()
    embs = embs.cpu().numpy()

    return xs, embs















""" ##################### Embedding 된 train feature 들로 Kmeans clustering 후 return.################################### """




def cluster_emb_tr(emb_tr, n_clusters=None, cluster_centers=None,sel_center='kmeans'):              # 둘 다 N, H, W, D

    D = emb_tr.shape[-1]
    emb_tr_all = emb_tr.reshape(-1, D)          # N*H*W, D


    if cluster_centers is not None:
        kmeans = MiniBatchKMeans(n_clusters=len(cluster_centers), batch_size=20, compute_labels=True, init=cluster_centers, n_init=1).fit(emb_tr_all)
        n_clusters = len(cluster_centers)

    elif n_clusters is not None:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=20, compute_labels=True).fit(emb_tr_all)


    labels = kmeans.labels_
    cluster_elem = []
    cluster_cen = []


    if sel_center == 'kmeans':
        for i in range(n_clusters):
            cluster_i_embs = emb_tr_all[labels==i]
            cluster_elem.append(cluster_i_embs)

        cluster_cen = torch.as_tensor(kmeans.cluster_centers_)




    elif sel_center == 'mean':
        for i in range(n_clusters):
            cluster_i_embs   = emb_tr_all[labels==i]
            cluster_i_center = cluster_i_embs.mean(0)

            cluster_elem.append(cluster_i_embs)
            cluster_cen.append(cluster_i_center)

        cluster_cen = torch.as_tensor(np.array(cluster_cen))

    return cluster_elem, cluster_cen





# def cluster_emb_tr(emb_tr, n_clusters):              # 둘 다 N, H, W, D
#
#     D = emb_tr.shape[-1]
#     emb_tr_all = emb_tr.reshape(-1, D)          # N*H*W, D
#
#     kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=20, compute_labels=True)
#     labels = kmeans.fit_predict(emb_tr_all)
#
#     cluster_elem = []
#     cluster_cen = []
#
#     for i in range(n_clusters):
#         cluster_i_embs   = emb_tr_all[labels==i]
#         cluster_i_center = cluster_i_embs.mean(0)
#
#         cluster_elem.append(cluster_i_embs)
#         cluster_cen.append(cluster_i_center)
#
#     cluster_cen = torch.as_tensor(np.array(cluster_cen))
#
#     return cluster_elem, cluster_cen











""" ##################### NN 알고리즘.################################### """

def search_NN_ngt(emb_te, cluster_elem, cluster_cen, NN=1):

    import ngtpy

    Ntest, H, W, D = emb_te.shape
    n_clusters = len(cluster_cen)


    dpath = f'/tmp/{os.getpid()}'
    ngtpy.create(dpath, D)

    index_list = []

    for i in range(n_clusters):
        index = ngtpy.Index(dpath)
        index.batch_insert(cluster_elem[i])
        index_list.append(index)

                       # cluster_cen --> (n_cluster, D)
    #cluster_cen  = torch.stack(cluster_cen, dim=0)                   # cluster_cen --> (n_cluster, D)
    l2_maps      = np.empty((Ntest, H, W), dtype=np.float32)





    """ 1. Find nearest cluster for each queries in emb_te. """
    cluster_cen = cluster_cen.cuda()                                # (n, D)
    emb_te_cuda = torch.tensor(emb_te).cuda()                   # (N, H, W, D)

    emb_dist_list = []
    for n in range(n_clusters):
        emb_dist = (emb_te_cuda - cluster_cen[n]).pow(2).sum(dim=3)
        emb_dist_list.append(emb_dist)

    emb_dist = torch.stack(emb_dist_list)                           # (n, N, H, W)
    emb_dist_argmin = emb_dist.argmin(dim=0).cpu().numpy()          # (N, H, W)





    for n in range(Ntest):

        #start = time.time()

        for i in range(H):
            for j in range(W):

                query = emb_te[n,i,j,:]
                clus_idx = int(emb_dist_argmin[n,i,j])


                # """ 1. For each query, calculate distances between each cluster's center. """                                            #
                # clus_idx2 = (cluster_cen - torch.tensor(query)).norm(dim=1).argmin()      # query --> (1, D),  query_cluster_dists --> (n_cluster, 1)
                # print(clus_idx == clus_idx2)



                """ 2. Load elements of Nearest cluster and get Nearest element of the cluster. """

                index = index_list[clus_idx]
                results = index.search(query, 1)
                inds = [result[0] for result in results]
                try    : vecs = np.asarray([index.get_object(inds[nn]) for nn in range(NN)])
                except : vecs = query



                """ 3. Get distance between query and the element and use it as an anomaly map"""
                dists = np.linalg.norm(query - vecs, axis=-1)
                l2_maps[n, i, j] = dists



        # print(f"image - {time.time() - start:.5f}")


    shutil.rmtree(dpath)
    return l2_maps








""" ######################## 출력 가지 고 score 및 AUROC 계산.########################################## """


def distribute_scores(score_masks, output_shape, K: int, S: int) -> np.ndarray:

    N    = score_masks.shape[0]
    I, J = score_masks[0].shape[:2]
    H, W = output_shape

    score_masks = torch.tensor(score_masks).cuda()
    mask = torch.zeros([N, H, W], dtype=torch.float32).cuda()
    cnt = torch.zeros([N, H, W], dtype=torch.int32).cuda()

    for i in range(I):
        for j in range(J):
            h, w = i * S, j * S

            mask[:, h: h + K, w: w + K] += score_masks[:, i, j].view(N, 1, 1)
            cnt[:,  h: h + K, w: w + K] += 1

    cnt[cnt == 0] = 1

    return (mask / cnt).cpu().numpy()







def assess_anomaly_maps(class_idx, anomaly_maps):

    class_names = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
                       'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

    DATASET_PATH = './dataset/mvtec_anomaly_detection'
    obj = class_names[class_idx]

    def detection_auroc(obj, anomaly_scores):
        def get_label(obj):
            fpattern = os.path.join(DATASET_PATH, f'{obj}/test/*/*.png')
            fpaths  = sorted(glob.glob(fpattern))
            fpaths1 = list(filter(lambda fpath: os.path.basename(os.path.dirname(fpath)) != 'good', fpaths))
            fpaths2 = list(filter(lambda fpath: os.path.basename(os.path.dirname(fpath)) == 'good', fpaths))

            Nanomaly = len(fpaths1)
            Nnormal = len(fpaths2)
            labels = np.zeros(Nanomaly + Nnormal, dtype=np.int32)
            labels[:Nanomaly] = 1
            return labels



        label = get_label(obj)  # 1: anomaly 0: normal

        print("Anomaly score mean, std : ", anomaly_scores.mean(), anomaly_scores.std())
        auroc = roc_auc_score(label, anomaly_scores)
        return auroc

    def segmentation_auroc(obj, anomaly_maps):
        def get_mask(obj):
            fpattern = os.path.join(DATASET_PATH, f'{obj}/ground_truth/*/*.png')
            fpaths = sorted(glob.glob(fpattern))
            masks = np.asarray(list(map(lambda fpath: cv2.resize(cv2.imread(fpath, cv2.IMREAD_GRAYSCALE), (256, 256)), fpaths)))
            Nanomaly = masks.shape[0]
            Nnormal = len(glob.glob(os.path.join(DATASET_PATH, f'{obj}/test/good/*.png')))

            masks[masks <= 128] = 0
            masks[masks > 128] = 255
            results = np.zeros((Nanomaly + Nnormal,) + masks.shape[1:], dtype=masks.dtype)



            results[:Nanomaly] = masks

            return results


        #print(label)
        #print(anomaly_scores)

        def bilinears(images, shape) -> np.ndarray:
            import cv2
            N = images.shape[0]
            new_shape = (N,) + shape
            ret = np.zeros(new_shape, dtype=images.dtype)
            for i in range(N):
                ret[i] = cv2.resize(images[i], dsize=shape[::-1], interpolation=cv2.INTER_LINEAR)
            return ret


        gt = get_mask(obj)
        gt = gt.astype(np.int32)
        gt[gt == 255] = 1  # 1: anomaly

        anomaly_maps = bilinears(anomaly_maps, (256, 256))                   # cv2.resize 그대로 사용. 단지 Batch가 있다보니 따로 함수화함.
        auroc = roc_auc_score(gt.flatten(), anomaly_maps.flatten())
        return auroc



    auroc_seg = segmentation_auroc(obj, anomaly_maps)

    anomaly_scores = anomaly_maps.max(axis=-1).max(axis=-1)
    auroc_det = detection_auroc(obj, anomaly_scores)
    return auroc_det, auroc_seg
