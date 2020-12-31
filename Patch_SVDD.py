# prerequisites
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, glob, cv2
import math
import torchvision
import random
import pickle
import umap

from functions import*
from evaluate import*
from evaluate_Kmeans_old import*
from layers import*
from Dataset import*
from loss import*
from functools import reduce
from architecture import*




def load_logs(class_idx):
    if os.path.exists(f'models/result_log_{class_idx}.pkl'):
        with open(f'models/result_log_{class_idx}.pkl', 'rb') as f:
            result_log = pickle.load(f)

        try:
            del result_log["center32"]
            del result_log["center64"]
        except:
            pass
        print(" **** Loads previous params.", result_log)



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']




cls_name = ["bottle", "cable",  "capsule",  "carpet",  "grid",  "hazelnut",  "leather",  "metal_nut",  "pill",  "screw",  "tile",  "toothbrush",  "transistor",  "wood",  "zipper"]
lambdas  = [ 0.001,    0.001,    0.001,      0.01,      0.01,    0.001,       0.01,       0.001,        0.001,   0.001,    0.01,    0.001,         0.001,         0.01,    0.001]



class Model():

    def __init__(self, settings, training=False):
        super(Model, self).__init__()

        # build model
        self.batch_size = settings['batch_size']
        self.image_shape= settings['image_shape']
        self.class_idx = settings['class_idx']
        self.object = False if self.class_idx in [3,6,10,13] else True
        self.lr = settings['lr']


        self.D = 64
        self.EncoderHier = EncoderHier(K=64, D=self.D, class_idx=self.class_idx).cuda()
        self.PositionClassifier_32  = PositionClassifier(K=32, D=self.D, class_idx=self.class_idx).cuda()
        self.PositionClassifier_64  = PositionClassifier(K=64, D=self.D, class_idx=self.class_idx).cuda()




        self.normal_dataset = settings['name']
        self.train_dir = settings['train_dir']
        self.test_dir = settings['test_dir']


        self.eval_setting = settings['eval_setting']
        self.n_cluster32 = self.eval_setting['n_cluster32'] if self.object else 2
        self.n_cluster64 = self.eval_setting['n_cluster64'] if self.object else 2

        self.center32 = None
        self.center64 = None
        self.sil_score32 = 0
        self.sil_score64 = 0

        self.cnt64 = 0
        self.cnt32 = 0


        self.last_epoch = 0

        self.max_det = 0
        self.max_seg = 0
        self.result_log = {"class_idx" : self.class_idx,
                           "n_cluster32": self.n_cluster32,
                           "n_cluster64": self.n_cluster64,
                           "anomaly_map_type" : None,
                           "max_det" : 0,
                           "max_seg" : 0,
                           "last_epoch" : 1}




        """ Datset for training """

        root_dir = './dataset/mvtec_anomaly_detection'
        train_imgs, _ = Read_MVtec_image_files(root_dir, self.image_shape[:2], mode='train', class_idx=self.class_idx, standardization=True)
        test_imgs,  _ = Read_MVtec_image_files(root_dir, self.image_shape[:2], mode='test', class_idx=self.class_idx, standardization=True)


        if training:
            print("\n **** Loading dataset for training.")


            d_svdd32 = MVtec_SVDD_Dataset(K=32, imgs=train_imgs)
            d_svdd64 = MVtec_SVDD_Dataset(K=64, imgs=train_imgs)
            d_pos32  = MVtec_Pos_Dataset(K=32, imgs=train_imgs)
            d_pos64  = MVtec_Pos_Dataset(K=64, imgs=train_imgs)
            d_obj32  = MVtec_Ano_Dataset(K=32, imgs=train_imgs)
            d_obj64  = MVtec_Ano_Dataset(K=64, imgs=train_imgs)


            self.d_svdd32 = torch.utils.data.DataLoader(d_svdd32, self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
            self.d_svdd64 = torch.utils.data.DataLoader(d_svdd64, self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
            self.d_pos32  = torch.utils.data.DataLoader(d_pos32,  self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
            self.d_pos64  = torch.utils.data.DataLoader(d_pos64,  self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
            self.d_na32   = torch.utils.data.DataLoader(d_obj32, self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
            self.d_na64   = torch.utils.data.DataLoader(d_obj64,  self.batch_size, shuffle=True, num_workers=0, pin_memory=True)


            # d_jig32 = MVtec_Jigsaw_Dataset(K=32, imgs=train_imgs)
            # d_jig64 = MVtec_Jigsaw_Dataset(K=64, imgs=train_imgs)
            # self.d_jig32 = torch.utils.data.DataLoader(d_jig32, batch_size=64, shuffle=True, num_workers=2,pin_memory=True)
            # self.d_jig64 = torch.utils.data.DataLoader(d_jig64, batch_size=64, shuffle=True, num_workers=2,pin_memory=True)




        """ Dataset for evaluating """
        print("\n **** Loading dataset for testing.")

        self.d_stride32_tr = MVtec_Stride_Patch_Dataset(K=32, S=4, imgs=train_imgs)
        self.d_stride32_te = MVtec_Stride_Patch_Dataset(K=32, S=4, imgs=test_imgs)
        self.d_stride64_tr = MVtec_Stride_Patch_Dataset(K=64, S=16, imgs=train_imgs)
        self.d_stride64_te = MVtec_Stride_Patch_Dataset(K=64, S=16, imgs=test_imgs)

        print("\n **** Loading dataset for result of anomaly map.")
        self.test_imgs, _ = Read_MVtec_image_files(root_dir, self.image_shape[:2], mode='test', class_idx=self.class_idx, standardization=False)

        print("\n\n\n")



    def load(self, eval_only=False):

        self.EncoderHier.load()
        if not eval_only:
            self.PositionClassifier_64.load()
            self.PositionClassifier_32.load()


        if os.path.exists(f'models/result_log_{self.class_idx}.pkl'):
            with open(f'models/result_log_{self.class_idx}.pkl', 'rb') as f:
                self.result_log = pickle.load(f)
                self.max_det = self.result_log["max_det"]
                self.max_seg = self.result_log["max_seg"]
                self.n_cluster32 = self.result_log["n_cluster32"]
                self.n_cluster64 = self.result_log["n_cluster64"]
                self.center32 = self.result_log["center32"]
                self.center64 = self.result_log["center64"]
                self.last_epoch = self.result_log["last_epoch"]

            del self.result_log["center32"]
            del self.result_log["center64"]



            print(" **** Loads previous params.\n", self.result_log)
            print("\n\n\n")






    def save(self):
        self.EncoderHier.save()
        self.PositionClassifier_32.save()
        self.PositionClassifier_64.save()

        with open(f'models/result_log_{self.class_idx}.pkl', 'wb') as f:
            pickle.dump(self.result_log, f, pickle.HIGHEST_PROTOCOL)







    def eval_and_save(self, mode='kmeans', save=True):

        d_stride32 = (self.d_stride32_tr, self.d_stride32_te)
        d_stride64 = (self.d_stride64_tr, self.d_stride64_te)


        if mode == 'kmeans':
            print("\n **** Evaluating model with Kmeans function.")
            results = eval_encoder_Kmeans(self.EncoderHier, d_stride32, d_stride64, self.class_idx, self.n_cluster32,self.n_cluster64)
        else:
            print("\n **** Evaluating model with original code")
            results = eval_encoder_NN_multiK(self.EncoderHier, d_stride32, d_stride64, self.class_idx, self.eval_setting)



        print(
            f"| K64 | Det: {results['det_64']:.3f} Seg:{results['seg_64']:.3f} "
            f"| K32 | Det: {results['det_32']:.3f} Seg:{results['seg_32']:.3f} "
            f"| sum | Det: {results['det_sum']:.3f} Seg:{results['seg_sum']:.3f} "
            f"| mult | Det: {results['det_mult']:.3f} Seg:{results['seg_mult']:.3f}\n")




        if (self.max_det + self.max_seg) < (results['det_best'] + results['seg_best']):
            print(" **** Best performance detected. Save model. \n\n")

            self.max_det, self.max_seg = round(results['det_best'], 3), round(results['seg_best'], 3)
            self.result_log["anomaly_map_type"] = results['type_best']
            self.result_log["max_det"] = self.max_det
            self.result_log["max_seg"] = self.max_seg
            self.result_log["n_cluster32"] = self.n_cluster32
            self.result_log["n_cluster64"] = self.n_cluster64
            self.result_log["center32"] = self.center32
            self.result_log["center64"] = self.center64
            self.result_log["last_epoch"] = self.last_epoch
            self.result_log["lr"] = self.lr

            if save:
                self.save()


            import matplotlib.pyplot as plt
            for i in range(len(results['anomaly_maps'])):
                plt.imshow(self.test_imgs[i])
                plt.imshow(results['anomaly_maps'][i], interpolation='bilinear', cmap='jet', alpha=0.6)
                plt.savefig(f"./result/result{self.class_idx}/map_result_{self.class_idx}_{i}", dpi=300)
                plt.clf()






    def Cluster_updating(self, X, centers, sil_scores):

        cluster32_centers, cluster64_centers = centers
        X32, X64 = X
        s32, s64 = sil_scores
        print()


        if self.n_cluster32  < len(cluster32_centers):
            self.cnt32 += 1
            if self.cnt32 > 2:
                print(f"**** Increase n_cluster32 due to counter: {min(16, self.n_cluster32 + 2)} \n")
                self.center32 = MiniBatchKMeans(n_clusters=min(16, self.n_cluster32 + 2), batch_size=10).fit(X32).cluster_centers_
                self.n_cluster32 = len(self.center32)
                self.sil_score32 = s32
                self.cnt32 = 0

        elif self.n_cluster32  == len(cluster32_centers):
            self.center32 = cluster32_centers
            self.n_cluster32 = len(self.center32)
            self.sil_score32 = s32
            self.cnt32 = 0


        elif self.n_cluster32  > len(cluster32_centers):
            self.cnt32 -= 1
            if self.cnt32 < - 2:
                print(f"**** Decrease n_cluster32 due to counter : {max(2, self.n_cluster32 - 2)} \n")
                self.center32 = MiniBatchKMeans(n_clusters=max(2, self.n_cluster32 - 2), batch_size=10).fit(X32).cluster_centers_
                self.n_cluster32 = len(self.center32)
                self.sil_score32 = s32
                self.cnt32 = 0




        if self.n_cluster64 < len(cluster64_centers):
            self.cnt64 += 1
            if self.cnt64 > 2:
                print(f"**** Increase n_cluster64 due to counter : {min(16, self.n_cluster64 + 2)} \n")
                self.center64 = MiniBatchKMeans(n_clusters=min(16, self.n_cluster64 + 2), batch_size=10).fit(X64).cluster_centers_
                self.n_cluster64 = len(self.center64)
                self.sil_score64 = s64
                self.cnt64 = 0

        elif self.n_cluster64  == len(cluster64_centers):
            self.center64 = cluster64_centers
            self.n_cluster64 = len(self.center64)
            self.sil_score64 = s64
            self.cnt64 = 0


        elif self.n_cluster64 > len(cluster64_centers):
            self.cnt64 -= 1
            if self.cnt64 < -2:
                print(f"**** Decrease n_cluster64 due to counter : {max(2, self.n_cluster64 - 2)} \n")
                self.center64 = MiniBatchKMeans(n_clusters=max(2, self.n_cluster64 - 2), batch_size=10).fit(X64).cluster_centers_
                self.n_cluster64 = len(self.center64)
                self.sil_score64 = s64
                self.cnt64 = 0


        print(f"**** n_cluster32 : {self.n_cluster32},    cnt32 : {self.cnt32},    score32 : {self.sil_score32:.4f}")
        print(f"**** n_cluster64 : {self.n_cluster64},    cnt64 : {self.cnt64},    score64 : {self.sil_score64:.4f}")

























    def train(self, epoch):

        modules = [self.EncoderHier, self.PositionClassifier_64, self.PositionClassifier_32]
        params = [list(module.parameters()) for module in modules]
        params = reduce(lambda x, y: x + y, params)

        opt  = torch.optim.Adam(params=params, lr=self.lr)
        xent = nn.CrossEntropyLoss(reduction='none')



        for epoch_idx in range(self.last_epoch + 1, self.last_epoch + epoch):

            obj32_dict = init_patch_dict(mode='obj32')
            obj64_dict = init_patch_dict(mode='obj64')


            for module in modules:
                module.train()



            ''' ******************* Train Patch_SVDD ******************* '''
            for batch_idx, (d32_svdd, d64_svdd, d32_pos, d64_pos, d32_na, d64_na) in enumerate(zip(self.d_svdd32, self.d_svdd64, self.d_pos32, self.d_pos64, self.d_na32, self.d_na64)):

                opt.zero_grad()


                p32_1, p32_jit, _ = d32_svdd
                p64_1, p64_jit, _ = d64_svdd

                p32_2, p32_pos, cls32, _ = d32_pos
                p64_2, p64_pos, cls64, _ = d64_pos

                p32_n, p32_a = d32_na
                p64_n, p64_a = d64_na



                p32_1, p32_jit        = p32_1.to('cuda:0', non_blocking=True), p32_jit.to('cuda:0',non_blocking=True)  # 셋 다 [B, 3, 32, 32]
                p32_2, p32_pos, cls32 = p32_2.to('cuda:0', non_blocking=True), p32_pos.to('cuda:0', non_blocking=True), cls32.to('cuda:0', non_blocking=True)
                p32_n, p32_a          = p32_n.to('cuda:0', non_blocking=True), p32_a.to('cuda:0',  non_blocking=True)  # 셋 다 [B, 3, 32, 32]

                p64_1, p64_jit        = p64_1.to('cuda:0', non_blocking=True), p64_jit.to('cuda:0', non_blocking=True)  # 셋 다 [B, 3, 32, 32]
                p64_2, p64_pos, cls64 = p64_2.to('cuda:0', non_blocking=True), p64_pos.to('cuda:0', non_blocking=True), cls64.to('cuda:0', non_blocking=True)
                p64_n, p64_a          = p64_n.to('cuda:0', non_blocking=True), p64_a.to('cuda:0', non_blocking=True)  # 셋 다 [B, 3, 32, 32]




                h64_2       = self.EncoderHier(p64_2)
                h64_pos     = self.EncoderHier(p64_pos)

                cls64_pred  = self.PositionClassifier_64(h64_2, h64_pos)  # [B, 8]
                loss_pos_64 = xent(cls64_pred, cls64).mean()*self.object

                h32_2       = self.EncoderHier.enc(p32_2)
                h32_pos     = self.EncoderHier.enc(p32_pos)
                cls32_pred  = self.PositionClassifier_32(h32_2, h32_pos)  # [B, 8]
                loss_pos_32 = xent(cls32_pred, cls32).mean()*self.object

                weight = min(1.0, max(0.0, -1 + epoch_idx / 15))
                # weight = 0
                h64_1        = self.EncoderHier(p64_1)
                h64_jit      = self.EncoderHier(p64_jit)  # 셋 다 [B, D, 1, 1]
                loss_svdd_64 = (h64_1 - h64_jit).norm(dim=1).mean() *(1-weight)#* (not self.object)

                h32_1        = self.EncoderHier.enc(p32_1)
                h32_jit      = self.EncoderHier.enc(p32_jit)  # 셋 다 [B, D, 1, 1]
                loss_svdd_32 = (h32_1 - h32_jit).norm(dim=1).mean()  *(1-weight)#* (not self.object)


                loss = (loss_svdd_64 + loss_svdd_32) * lambdas[self.class_idx] + (loss_pos_64 + loss_pos_32)






                """  Self - supervised learning (No training yet) """

                h32_n      = self.EncoderHier.enc(p32_n).detach()
                h32_a      = self.EncoderHier.enc(p32_a).detach()

                h64_n      = self.EncoderHier(p64_n).detach()
                h64_a      = self.EncoderHier(p64_a).detach()



                h32 = torch.cat([h32_1, h32_2, h32_pos, h32_n]).squeeze()
                h64 = torch.cat([h64_1, h64_2, h64_pos, h64_n]).squeeze()
                h32_n, h32_a = h32_n.squeeze(), h32_a.squeeze()
                h64_n, h64_a = h64_n.squeeze(), h64_a.squeeze()




                ''' texture  '''
                h32_nn_loss, h64_nn_loss = 0, 0
                h32_na_loss, h64_na_loss = 0, 0




                h32_nn_loss = DeepCluster_loss(h32, n_cluster=self.n_cluster32, is_obj=self.object, p=0.6) * lambdas[self.class_idx]
                h64_nn_loss = DeepCluster_loss(h64, n_cluster=self.n_cluster64, is_obj=self.object, p=0.6) * lambdas[self.class_idx]

                #h32_na_loss = torch.cdist(h32_n, h32_a).mean()* lambdas[self.class_idx]
                #h64_na_loss = torch.cdist(h64_n, h64_a).mean()* lambdas[self.class_idx]
                h32_na_loss = DeepCluster_loss_na(h32_n, h32_a, p=0.2)
                h64_na_loss = DeepCluster_loss_na(h64_n, h64_a, p=0.2)


                total_cluster_loss32 = (h32_nn_loss - h32_na_loss) * weight
                total_cluster_loss64 = (h64_nn_loss - h64_na_loss) * weight

                loss += (total_cluster_loss64 + total_cluster_loss32)

                loss.backward()
                opt.step()




                save_muliple_patches_and_features([p32_2, p32_pos], [h32_2, h32_pos, h32_n, h32_a], num=15, save_dict=obj32_dict, mode='obj32')
                save_muliple_patches_and_features([p64_2, p64_pos], [h64_2, h64_pos, h64_n, h64_a], num=15, save_dict=obj64_dict, mode='obj64')



                if (batch_idx + 1) % 30 == 0:

                    print(  f'\nTrain Epoch: {epoch_idx} [{batch_idx * len(p64_1)}/{len(self.d_svdd32.dataset)} ({batch_idx * len(p64_1)/len(self.d_svdd32.dataset)*100:.0f}%)] '
                            f'\nloss_pos_64: {loss_pos_64:.4f} '
                            f'\tloss_pos_32: {loss_pos_32:.4f} '
                            f'\tloss_svdd_64: {loss_svdd_64:.4f} '
                            f'\tloss_svdd_32: {loss_svdd_32:.4f} '
                            f'\tlr: {get_lr(opt):.6f}'
                            f'\nh32_nn_loss32: {h32_nn_loss: .4f}'
                            f'\th64_nn_loss64: {h64_nn_loss: .4f}'
                            f'\nh32_na_loss32: {h32_na_loss: .4f}'
                            f'\th64_na_loss64: {h64_na_loss: .4f}'
                         )




                    #if not self.object:

                    max_score32, cluster32_centers, (X32, labels32) = Split_feature_by_kmeans(obj32_dict, batch_idx, self.class_idx, K=32, min_cluster=2, max_cluster=16, grid_col=10)
                    max_score64, cluster64_centers, (X64, labels64) = Split_feature_by_kmeans(obj64_dict, batch_idx, self.class_idx, K=64, min_cluster=2, max_cluster=16, grid_col=10)

                    #if epoch_idx < 50:
                    self.n_cluster32 = len(cluster32_centers)
                    self.n_cluster64 = len(cluster64_centers)
                    self.center32 = cluster32_centers
                    self.center64 = cluster64_centers


                        # else:
                        #     X = (X32, X64)
                        #     centers = (cluster32_centers, cluster64_centers)
                        #     scores  = (max_score32, max_score64)
                        #     self.Cluster_updating(X, centers, scores)





                    if not self.object:
                        h32_n_list, h32_a_list = obj32_dict['obj32_h'], obj32_dict['obj32_h_a']
                        h64_n_list, h64_a_list = obj64_dict['obj64_h'], obj64_dict['obj64_h_a']

                        X32 = torch.cat(h32_n_list + h32_a_list, dim=0).detach().cpu().numpy()
                        X64 = torch.cat(h64_n_list + h64_a_list, dim=0).detach().cpu().numpy()

                        labels32 = np.array([0]*len(h32_n_list)*len(h32_n_list[0]) + [1]*len(h32_a_list)*len(h32_a_list[0]))
                        labels64 = np.array([0]*len(h64_n_list)*len(h64_n_list[0]) + [1]*len(h64_a_list)*len(h64_a_list[0]))


                    obj32_dict = init_patch_dict(mode='obj32')
                    obj64_dict = init_patch_dict(mode='obj64')




            plot_embedding(X32, labels32, self.n_cluster32, savedir=f"./result_patch/Result{self.class_idx}/TSNE32_{self.class_idx}_{epoch_idx}",mode="TSNE", add_anomaly=not self.object)
            plot_embedding(X64, labels64, self.n_cluster64, savedir=f"./result_patch/Result{self.class_idx}/UMAP64_{self.class_idx}_{epoch_idx}",mode="TSNE", add_anomaly=not self.object)

            if (epoch_idx % 1 == 0)  and not     self.object: self.eval_and_save()
            if (epoch_idx % 1 == 0)  and   self.object: self.eval_and_save(mode='original')

            # # Evaluate first too save initial performance.
            self.last_epoch = epoch_idx


            print("\n\n\n\n")






















mvtec_setting = {'name': 'mvtec',
                 'class_idx' : 0,

                 'image_shape': (256, 256, 3),
                 'batch_size' : 64,
                 'lr': 1e-4,

                 'train_dir': './dataset/mvtec_anomaly_detection',
                 'test_dir': './dataset/mvtec_anomaly_detection',

                 'eval_setting' :
                     { 'save_train_emb' : True,
                       'load_train_emb' : False,         # model 새로 train 시 한번은 여기를 False 로 둬야함.
                       'filename' : "./models/train_dataset_embeddings",

                       'n_cluster32': 3,
                       'n_cluster64': 3,
                     }
                 }


if __name__ == '__main__':


    for i in range(0, 15):
        load_logs(i)
    print("\n\n\n")




    # for i in [3, 9, 12, 14]:
    #     # if i in [3, 6, 10, 13]:
    #     #     continue
    #
    #     mvtec_setting["class_idx"] = 1
    #     SVDD = Model(mvtec_setting, training=True)
    #
    #     SVDD.load()
    #     SVDD.train(200)








    # mvtec_setting["class_idx"] = 1
    # SVDD = Model(mvtec_setting, training=False)
    # SVDD.load()
    # #SVDD.eval_and_save(mode='original', save=False)
    # print("\n\n\n\n\n\n")
    # SVDD.eval_and_save(save=False)
    # SVDD.eval_and_save(save=False)
    # SVDD.eval_and_save(save=False)

    # mvtec_setting["class_idx"] = 8
    # mvtec_setting["lr"] = 1e-5
    # SVDD = Model(mvtec_setting, training=True)
    # #SVDD.eval_and_save(save=False)
    # SVDD.load()
    # SVDD.train(300)
    #
    #
    # mvtec_setting["class_idx"] = 9
    # mvtec_setting["lr"] = 3e-7
    # SVDD = Model(mvtec_setting, training=True)
    # #SVDD.load()
    # SVDD.train(300)


    # mvtec_setting["class_idx"] = 10
    # mvtec_setting["lr"] = 1e-4
    # SVDD = Model(mvtec_setting, training=True)
    # #SVDD.load()
    # SVDD.train(300)


    # mvtec_setting["class_idx"] = 11
    # mvtec_setting["lr"] = 3e-5
    # SVDD = Model(mvtec_setting, training=True)
    # #SVDD.load()
    # SVDD.train(300)


    # mvtec_setting["class_idx"] = 12
    # mvtec_setting["lr"] = 1e-6
    # SVDD = Model(mvtec_setting, training=True)
    # SVDD.eval_and_save(save=False)
    # SVDD.load()
    # SVDD.train(300)

    mvtec_setting["class_idx"] = 13
    mvtec_setting["lr"] = 1e-4
    SVDD = Model(mvtec_setting, training=True)
    SVDD.load()
    SVDD.train(300)

    mvtec_setting["class_idx"] = 14
    mvtec_setting["lr"] = 1e-4
    SVDD = Model(mvtec_setting, training=True)
    #SVDD.load()
    SVDD.train(300)



    # for i in range(8, 15):
    #     mvtec_setting["class_idx"] = i
    #     SVDD = Model(mvtec_setting, training=True)
    #
    #     if i in [3, 6, 10, 13, 9]:
    #         continue
    #     # SVDD.load()
    #     SVDD.train(200)



    for i in range(1, 15):
        if i == 10: continue

        try:
            mvtec_setting["class_idx"] = i
            SVDD = Model(mvtec_setting, training=False)
            SVDD.load(eval_only=True)
            SVDD.eval_and_save(mode='original', save=False)
            SVDD.eval_and_save(save=False)
        except:
            continue


    print("\n\n\n\n\n")