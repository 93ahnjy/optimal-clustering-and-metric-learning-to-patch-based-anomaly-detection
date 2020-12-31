
import numpy as np
import os, glob
import torch

import shutil
import cv2


from Patch_SVDD import *
from Dataset import MVtec_Stride_Patch_Dataset

from sklearn.metrics import roc_auc_score
import mkl
mkl.set_num_threads(12)

__all__ = ['eval_encoder_NN_multiK', 'MVtec_Stride_Patch_Dataset' ,'infer', 'log_results', 'make_grid', 'distribute_scores', 'assess_anomaly_maps']









def eval_encoder_NN_multiK(enc, dataset32, dataset64, class_idx, eval_setting):

    save_train_emb = eval_setting['save_train_emb']
    load_train_emb = eval_setting['load_train_emb']
    save_name      = eval_setting['filename']

    dataset32_tr, dataset32_te = dataset32
    dataset64_tr, dataset64_te = dataset64


    if load_train_emb and os.path.exists(f'{save_name}_{class_idx}_{64}.npy'):
        print(f"Load file : {save_name}_{class_idx}_{64}.npy")                    # N, H, W, D
        embs64 = np.load(f'{save_name}_{class_idx}_{64}.npy', allow_pickle=True)
        print(embs64.mean(), embs64.std())

    else:
        embs64_tr  = infer(dataset64_tr, enc, K=64, S=16)
        embs64_te  = infer(dataset64_te, enc, K=64, S=16)
        embs64     = (embs64_tr, embs64_te)

        if save_train_emb: np.save(f'{save_name}_{class_idx}_{64}.npy', embs64)





    if load_train_emb and os.path.exists(f'{save_name}_{class_idx}_{32}.npy'):
        print(f"Load file : {save_name}_{class_idx}_{32}.npy")
        embs32 = np.load(f'{save_name}_{class_idx}_{32}.npy', allow_pickle=True)

    else:
        embs32_tr  = infer(dataset32_tr, enc.enc, K=32, S=4)
        embs32_te  = infer(dataset32_te, enc.enc, K=32, S=4)
        embs32     = (embs32_tr, embs32_te)

        if save_train_emb: np.save(f'{save_name}_{class_idx}_{32}.npy', embs32)


    return eval_embeddings_NN_multiK(class_idx, embs64, embs32)










def infer(dataset, enc, K, S):

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


    print("EMBS m, std : ", embs.mean(), embs.std())
    return embs






###########################################################################################



def eval_embeddings_NN_multiK(class_idx, embs64, embs32, NN=1):

    start = time.time()

    print("\n ** K = 64")
    emb_tr, emb_te = embs64
    maps_64 = measure_emb_NN(emb_te, emb_tr, method='kdt', NN=NN)
    maps_64 = distribute_scores(maps_64, (256, 256), K=64, S=16)

    print("\n ** K = 32")
    emb_tr, emb_te = embs32
    maps_32 = measure_emb_NN(emb_te, emb_tr, method='ngt', NN=NN)
    maps_32 = distribute_scores(maps_32, (256, 256), K=32, S=4)         # N 개의 anomaly map 생성.

    print(f" Total Time elapsed : {time.time() - start: .3f} sec")
    print(f" Number of image -  train : {emb_tr.shape[0]},  test : {emb_te.shape[0]}")
    print(f" Speed : {(time.time() - start)/emb_te.shape[0]: .3f} sec / img\n\n")





    # Map을 합치거나, 곱해보면서 어떤 map 구성이 더 성능이 좋은 지 확인.
    det_64, seg_64 = assess_anomaly_maps(class_idx, maps_64)
    det_32, seg_32 = assess_anomaly_maps(class_idx, maps_32)

    maps_sum = maps_64 + maps_32
    det_sum, seg_sum = assess_anomaly_maps(class_idx, maps_sum)

    maps_mult = maps_64 * maps_32
    det_mult, seg_mult = assess_anomaly_maps(class_idx, maps_mult)


    results = [det_32, seg_32, det_64, seg_64, det_sum, seg_sum, det_mult, seg_mult]
    perf    = [det_32 + seg_32, det_64 + seg_64, det_sum + seg_sum, det_mult + seg_mult]
    maps    = [maps_32, maps_64, maps_sum, maps_mult]
    types   = ['K32', 'K64', 'sum', 'mult']


    idx      = perf.index(max(perf))
    maps_best = maps[idx]
    type_best  = types[idx]
    det_best = results[idx*2]
    seg_best = results[idx*2 + 1]


    return {
        'anomaly_maps': maps_best,

        'det_64': det_64,
        'seg_64': seg_64,

        'det_32': det_32,
        'seg_32': seg_32,

        'det_sum': det_sum,
        'seg_sum': seg_sum,

        'det_mult': det_mult,
        'seg_mult': seg_mult,

        'det_best': det_best,
        'seg_best': seg_best,
        'type_best': type_best
    }





def log_results(results):
    det_64 = results['det_64']
    seg_64 = results['seg_64']

    det_32 = results['det_32']
    seg_32 = results['seg_32']

    det_sum = results['det_sum']
    seg_sum = results['seg_sum']

    det_mult = results['det_mult']
    seg_mult = results['seg_mult']
    print(
        f'| K64 | Det: {det_64:.3f} Seg:{seg_64:.3f} | K32 | Det: {det_32:.3f} Seg:{seg_32:.3f} | sum | Det: {det_sum:.3f} Seg:{seg_sum:.3f} | mult | Det: {det_mult:.3f} Seg:{seg_mult:.3f} ')

    return




###########################################################################################
















def measure_emb_NN(emb_te, emb_tr, method='kdt', NN=1):
    D = emb_tr.shape[-1]
    train_emb_all = emb_tr.reshape(-1, D)  # flatten


    print("\n Find nearest element in train_emb_all")
    start = time.time()

    if method == 'ngt': l2_maps, _ = search_NN_ngt(emb_te, train_emb_all, NN=NN)
    elif method == 'kdt': l2_maps, _ = search_NN(emb_te, train_emb_all, NN=NN)

    print(f" Time elapsed : {time.time() - start: .3f} sec\n\n")


    anomaly_maps = np.mean(l2_maps, axis=-1)

    return anomaly_maps




def search_NN(test_emb, train_emb_flat, NN=1):

    print("Use KDT")

    from sklearn.neighbors import KDTree
    kdt = KDTree(train_emb_flat)

    Ntest, I, J, D = test_emb.shape
    closest_inds = np.empty((Ntest, I, J, NN), dtype=np.int32)
    l2_maps = np.empty((Ntest, I, J, NN), dtype=np.float32)

    for n in range(Ntest):
        for i in range(I):
            dists, inds = kdt.query(test_emb[n, i, :, :], return_distance=True, k=NN)
            closest_inds[n, i, :, :] = inds[:, :]
            l2_maps[n, i, :, :] = dists[:, :]

    return l2_maps, closest_inds




def search_NN_ngt(test_emb, train_emb_flat, NN=1):

    print("Use NGT")

    import ngtpy

    Ntest, I, J, D = test_emb.shape
    closest_inds   = np.empty((Ntest, I, J, NN), dtype=np.int32)
    l2_maps        = np.empty((Ntest, I, J, NN), dtype=np.float32)

    # os.makedirs('tmp', exist_ok=True)
    dpath = f'/tmp/{os.getpid()}'
    ngtpy.create(dpath, D)

    index = ngtpy.Index(dpath)
    index.batch_insert(train_emb_flat)

    for n in range(Ntest):
        for i in range(I):
            for j in range(J):
                query = test_emb[n, i, j, :]
                results = index.search(query, NN)
                inds = [result[0] for result in results]

                closest_inds[n, i, j, :] = inds
                vecs = np.asarray([index.get_object(inds[nn]) for nn in range(NN)])
                dists = np.linalg.norm(query - vecs, axis=-1)
                l2_maps[n, i, j, :] = dists
    shutil.rmtree(dpath)

    return l2_maps, closest_inds




#########################################################################



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




#########################################################################




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





















def make_grid(img_list, num_col):

    col_imgs = []

    for i in range(len(img_list)//num_col):
        col_img = np.concatenate(img_list[i*num_col:(i+1)*num_col], axis=1)
        col_imgs.append(col_img)

    print(col_imgs[0].shape)
    total_img = np.concatenate(col_imgs, axis=0)

    return total_img

















mvtec_setting = {'name': 'mvtec',
                 'class_idx' : 5,

                 'image_shape': (256, 256, 3),
                 'batch_size' : 64,

                 'train_dir': './dataset/mvtec_anomaly_detection',
                 'test_dir': './dataset/mvtec_anomaly_detection',

                 'eval_setting':
                     {'save_train_emb': True,
                      'load_train_emb': False,  # model 새로 train 시 한번은 여기를 False 로 둬야함.
                      'filename': "./models/train_dataset_embeddings",
                      }
                 }



if __name__ == '__main__':


    for i in [3]:
        mvtec_setting["class_idx"] = i
        SVDD = Model(mvtec_setting)
        SVDD.load()

        #  이 imgs는 eval 에 쓰이는 게 아님. 그냥 결과 를 표시하 는 용도.
        imgs, _ = Read_MVtec_image_files('./dataset/mvtec_anomaly_detection', (256, 256), mode='test', class_idx=mvtec_setting['class_idx'], K=32, standardization=False)
        print("class index : ", i, "\n")


        results = eval_encoder_NN_multiK(SVDD.EncoderHier, class_idx=mvtec_setting['class_idx'], eval_setting=mvtec_setting['eval_setting'])
        log_results(results)


        print("\n\n\n\n\n")