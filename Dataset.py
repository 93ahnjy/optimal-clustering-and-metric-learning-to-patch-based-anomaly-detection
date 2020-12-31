
import torch
import numpy as np
import os, glob, cv2
import random
from Dataset_augmentation import remove_background, RandomPolygon



direction = {
    0: (-1, -1),
    1: (-1, 0),
    2: (-1, 1),
    3: (0, -1),
    4: (0, 1),
    5: (1, -1),
    6: (1, 0),
    7: (1, 1)
}




def RandomJitter_patch(img, p, K):

    H, W, C = img.shape

    p_y, p_x     = p
    h_jit, w_jit = 0, 0

    while h_jit == 0 and w_jit == 0:
        h_jit = np.random.randint(-K//32, K//32 + 1)
        w_jit = np.random.randint(-K//32, K//32 + 1)

    p1_y = p_y + h_jit
    p1_x = p_x + w_jit

    p1_y = np.clip(p1_y, 0, H - K)
    p1_x = np.clip(p1_x, 0, W - K)

    patch  = img[p_y :p_y  + K, p_x :p_x  + K, :].copy()
    patch1 = img[p1_y:p1_y + K, p1_x:p1_x + K, :].copy()

    patch    = torch.from_numpy(patch.transpose(2, 0, 1))
    patch1   = torch.from_numpy(patch1.transpose(2, 0, 1))

    p1     = (p1_y, p1_x)

    return patch, (patch1, p1)




def RandomNeighborhood_patch(img, p, K):

    H, W, C = img.shape

    p_y, p_x   = p


    while True:
        cls = np.random.randint(8)
        h_dir, w_dir = direction[cls]
        h_sca, w_sca = np.random.randint(K * 0.5, K, size=2)

        p2_y = p_y + (h_dir * h_sca)
        p2_x = p_x + (w_dir * w_sca)

        if 0<p2_y <H-K and 0<p2_x<W-K:
            break



    patch  = img[p_y :p_y  + K, p_x :p_x  + K, :].copy()
    patch2 = img[p2_y:p2_y + K, p2_x:p2_x + K, :].copy()
    p2     = (p2_y, p2_x)


    # perturb RGB
    rgbshift    = np.random.normal(scale=0.02, size=(1, 1, 3))
    rgbshift2   = np.random.normal(scale=0.02, size=(1, 1, 3))

    patch    += rgbshift
    patch2   += rgbshift2

    # additive noise
    noise2_o = np.random.normal(scale=0.02, size=(K, K, 3))
    noise2   = np.random.normal(scale=0.02, size=(K, K, 3))

    patch    += noise2_o
    patch2   += noise2

    patch    = torch.from_numpy(patch.transpose(2, 0, 1))
    patch2   = torch.from_numpy(patch2.transpose(2, 0, 1))

    return patch, (patch2, p2), cls     # .copy() 안하면 해당 patch에  noise, rgbshift 적용 시 원본이미지도 영향받음.













# def RandomJigsaw_patch(img, p, K):
#
#     H, W, C = img.shape
#
#     p_y, p_x   = p
#
#
#     coord = []
#     for cls in range(8):
#
#         while True:
#             h_dir, w_dir = direction[cls]
#             h_sca, w_sca = np.random.randint(K*0.5, K, size=2)
#
#             p2_y = p_y + (h_dir * h_sca)
#             p2_x = p_x + (w_dir * w_sca)
#
#             if 0<p2_y <H-K and 0<p2_x<W-K:
#                 coord.append((p2_y, p2_x))
#                 break
#
#
#     patches_0_3 = []
#     patches_5_8 = []
#     patch = img[p_y:p_y + K, p_x:p_x + K, :].copy()
#     patch += np.random.normal(scale=0.02, size=(1, 1, 3))       # rgbshift
#     patch += np.random.normal(scale=0.02, size=(K, K, 3))       # noise
#     patch = torch.from_numpy(patch.transpose(2, 0, 1))
#
#     for cls, (p2_y, p2_x) in enumerate(coord):
#         patch2 = img[p2_y:p2_y + K, p2_x:p2_x + K, :].copy()
#
#         patch2   += np.random.normal(scale=0.02, size=(1, 1, 3))    # perturb RGB
#         patch2   += np.random.normal(scale=0.02, size=(K, K, 3))    # additive noise
#         patch2   = torch.from_numpy(patch2.transpose(2, 0, 1))
#
#         if cls <=3:
#             patches_0_3.append(patch2)
#         else:
#             patches_5_8.append(patch2)
#
#
#     patches = torch.stack(patches_0_3 + [patch] + patches_5_8, dim=0)
#     classes     = torch.randperm(9)
#     patches = patches[cls]
#
#     return patches, classes





def permutation_opt():
    able = [0,1,2,3,4,5,6,7,8]
    output = []
    for i in range(9):
        sel = random.choice(able[:i] + able[i+1:])
        output.append(sel)
        able.remove(sel)

    return output






def RandomJigsaw_patch(img, p, K):

    H, W, C = img.shape
    p_y, p_x   = p

    direction = {
        0: (-1, -1),
        1: (-1, 0),
        2: (-1, 1),
        3: (0, -1),
        4: (0, 0),
        5: (0, 1),
        6: (1, -1),
        7: (1, 0),
        8: (1, 1)
    }


    coord = []
    for cls in range(9):

        while True:
            h_dir, w_dir = direction[cls]
            h_sca, w_sca = np.random.randint(K, K*1.2, size=2)

            p2_y = p_y + (h_dir * h_sca)
            p2_x = p_x + (w_dir * w_sca)

            if 0<=p2_y <H-K and 0<=p2_x<W-K:
                coord.append((p2_y, p2_x))
                break


    patches = []

    for cls, (p2_y, p2_x) in enumerate(coord):
        patch2 = img[p2_y:p2_y + K, p2_x:p2_x + K, :].copy()

        patch2   += np.random.normal(scale=0.02, size=(1, 1, 3))    # perturb RGB
        patch2   = torch.from_numpy(patch2.transpose(2, 0, 1))

        patches.append(patch2)


    patches = torch.stack(patches)
    #classes = torch.tensor([0,1,2,3,4,5,6,7,8])
    classes = torch.tensor(permutation_opt())
    patches = patches[classes]

    return patches, classes

























def Read_MVtec_image_files(root_dir, input_size, mode, class_idx, standardization=False):

    class_names = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
                   'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    data_dir = os.path.join(root_dir, class_names[class_idx], mode)

    print(f"Gathering data : {class_names[class_idx]},   type : {mode}")

    if   mode == 'train':
        files = sorted(glob.glob(data_dir + "/*.png"))
    elif mode == 'test' :
        files = sorted(glob.glob(data_dir + "/*/*.png"))        # './dataset/mvtec_anomaly_detection/leather/test/[!good]*/*.png'

        anomal = [fn for fn in files if not "good" in fn]
        normal = [fn for fn in files if "good" in fn]            # './dataset/mvtec_anomaly_detection/leather/test/good/*.png'
        files = anomal + normal


    data    = np.asarray(list(map(lambda fpath: cv2.resize(cv2.cvtColor(cv2.imread(fpath), cv2.COLOR_BGR2RGB), input_size), files)))
    targets = list(map(lambda fpath: 1 if "good" in fpath else 0, files))



    data = np.float32(data)

    # Standardization
    if standardization: data = (data - data.mean(axis=0)) / 255
    # if standardization:
    #     data = (data - np.mean(data, (0,1,2))) / np.std(data, (0,1,2))
    else: data /= 255

    return data, targets
































class MVtec_SVDD_Dataset(torch.utils.data.Dataset):  # ./dataset/mvtec_anomaly_detection

    def __init__(self, K, imgs):
        self.K = K
        self.data = imgs

    def __len__(self):
        return len(self.data) * 100

    def __getitem__(self, index):

        index = index % len(self.data)

        img = self.data[index]
        H, W, C = img.shape
        h = np.random.randint(0, H - self.K + 1)
        w = np.random.randint(0, W - self.K + 1)

        patch, (patch1, p1) = RandomJitter_patch(img, (h, w), self.K)

        return patch, patch1, p1







class MVtec_Pos_Dataset(torch.utils.data.Dataset):  # ./dataset/mvtec_anomaly_detection

    def __init__(self, K, imgs):
        self.K = K
        self.data = imgs

    def __len__(self):
        return len(self.data) * 100

    def __getitem__(self, index):

        index = index % len(self.data)

        img = self.data[index]
        H, W, C = img.shape
        h = np.random.randint(0, H - self.K + 1)
        w = np.random.randint(0, W - self.K + 1)

        patch, (patch2, p2), cls = RandomNeighborhood_patch(img, (h, w), self.K)

        return patch, patch2, cls, p2














class MVtec_Ano_Dataset(torch.utils.data.Dataset):  # ./dataset/mvtec_anomaly_detection

    def __init__(self, K, imgs):
        self.K = K
        self.data = imgs

    def __len__(self):
        return len(self.data) * 100

    def __getitem__(self, index):

        index = index % len(self.data)
        img  = self.data[index]
        H, W, C = img.shape

        ''' 2. Select locations of two patches'''
        p1_y, p1_x = np.random.randint(0, H - self.K + 1), np.random.randint(0, W - self.K + 1)
        p1_y = np.clip(p1_y, 0, H - self.K)
        p1_x = np.clip(p1_x, 0, W - self.K)

        p2_y, p2_x = np.random.randint(0, H - self.K + 1), np.random.randint(0, W - self.K + 1)
        p2_y = np.clip(p2_y, 0, H - self.K)
        p2_x = np.clip(p2_x, 0, W - self.K)


        ''' 3. Extract two patches from image'''
        patch1 = img[p1_y:p1_y + self.K, p1_x:p1_x + self.K, :].copy()
        patch2 = img[p1_y:p1_y + self.K, p1_x:p1_x + self.K, :].copy()

        patch2 = RandomPolygon(patch2)

        patch1 = torch.from_numpy(patch1.transpose(2, 0, 1))
        patch2 = torch.from_numpy(patch2.transpose(2, 0, 1))

        return patch1, patch2










class MVtec_Jigsaw_Dataset(torch.utils.data.Dataset):  # ./dataset/mvtec_anomaly_detection

    def __init__(self, K, imgs):
        self.K = K
        self.data = imgs

    def __len__(self):
        return len(self.data) * 100

    def __getitem__(self, index):

        index = index % len(self.data)

        img = self.data[index]
        H, W, C = img.shape
        h = np.random.randint(self.K,  H - self.K*1.2*2 + 1)
        w = np.random.randint(self.K,  W - self.K*1.2*2 + 1)

        patches, classes = RandomJigsaw_patch(img, (h, w), self.K)

        return patches, classes










class MVtec_Stride_Patch_Dataset(torch.utils.data.Dataset):

    def __init__(self, K, S, imgs, usage_ratio=1):

        self.data = imgs
        self.data = self.data[:int(len(self.data) * usage_ratio)]

        self.H = imgs.shape[1]
        self.W = imgs.shape[2]

        self.K = K
        self.S = S

        self.emb_row = 1 + (self.H - K) // S
        self.emb_col = 1 + (self.W - K) // S

        self.total_img = len(self.data)


    def __len__(self):
        return len(self.data) * self.emb_row * self.emb_col       # (img 당 self.emb_row * self.emb_col 개의 patch) * (img 는 총 len(self.data) 장)


    def __getitem__(self, index):

        n, i, j = np.unravel_index(index, (len(self.data), self.emb_row, self.emb_col))
        img = self.data[n]

        h = self.S * i
        w = self.S * j
        patch = img[h: h + self.K, w: w + self.K, :]

        patch = torch.from_numpy(patch.transpose(2,0,1))

        return patch, n, i, j


















