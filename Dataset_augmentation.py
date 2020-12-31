import numpy as np
import cv2
import os, glob
import Augmentor
import time
import random
import torch
import torchvision

from math import sin, cos, radians
from Dataset import *


def apply_motion_blur(image, size, angle):
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[ (size-1)// 2 , :] = np.ones(size, dtype=np.float32)

    M = cv2.getRotationMatrix2D( (size / 2 -0.5 , size / 2 -0.5 ) , angle, 1.0)

    kernel = cv2.warpAffine(kernel, M, (size, size) )
    kernel = kernel * ( 1.0 / np.sum(kernel) )
    return cv2.filter2D(image, -1, kernel)







def RandomPolygon(img, num_defect=2, standardization=True):  # image is float

    H,W,C = img.shape



    for _ in range(num_defect):
        ''' Generate random rectangle '''
        w = random.randint(W // 6, W // 2)
        h = random.randint(H // 6, H // 2)
        angle = random.randint(0, 90)
        color = (random.uniform(0.1, 0.5), random.uniform(0.1, 0.5), random.uniform(0.1, 0.5))


        aug_temp = np.zeros((h, w, 3), np.float32)
        aug_temp[:] = color



        ''' Rotate rectangle '''
        w_rot = int(w*cos(radians(angle)) + h*sin(radians(angle)))
        h_rot = int(h*cos(radians(angle)) + w*sin(radians(angle)))

        matrix = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        aug = cv2.warpAffine(aug_temp, matrix, (w_rot, h_rot), cv2.BORDER_ISOLATED, borderValue=0)
        aug = cv2.GaussianBlur(aug, (3,3), sigmaX=1)


        ''' Select location to put augmentation'''
        x = random.randint(0, W-w_rot)
        y = random.randint(0, H-h_rot)
        roi = img[y:y+h_rot, x:x+w_rot]


        ''' Fill aug's black area with background'''
        aug = np.where(aug !=0, aug, roi)


        ''' Add with roi to make semi-transparent image'''
        p = random.uniform(0.3, 0.8)
        dst = cv2.addWeighted(aug, p, img[y:y+h_rot, x:x+w_rot], 1-p, 0)

        img[y:y+h_rot, x:x+w_rot] = dst



    if random.uniform(0, 1) < 0.5:
        img = apply_motion_blur(img, size=H//8, angle=random.randint(10, 180))



    if random.uniform(0, 1) < 0.3:
        x1 = random.randint(3, W-1)
        x2 = random.randint(3, W-1)
        th = random.randint(2, 3)
        color = (0, 0, 0)
        img = cv2.line(img, (x1,0), (x2,H-1), color, th)


    if random.uniform(0, 1) < 0.3:
        y1 = random.randint(5, H-1)
        y2 = random.randint(5, H-1)
        th = random.randint(2, 3)
        color = (0, 0, 0)
        img = cv2.line(img, (0,y1), (W-1,y2), color, th)


    return img




















def remove_background(img):

    img  = cv2.resize(img, (64, 64), cv2.INTER_AREA)
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (1, 0, img.shape[0]-2, img.shape[1]-1)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount=20, mode=cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    mask3  = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask3















if __name__ == '__main__':


    class_idx = 14     # 0,1,2,5,7,8,9,11,12,14

    class_names = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
                   'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']


    data_dir = os.path.join('./dataset/mvtec_anomaly_detection', class_names[class_idx], 'train')
    files = sorted(glob.glob(data_dir + "/*.png"))




    H = 256
    W = 256
    K = 32
    colors = [(0,0,255), (0,255,0), (255,0,0)]

    # start = time.time()
    # for i, file in enumerate(files):
    #     img = cv2.resize(cv2.imread(file), (256, 256), cv2.INTER_AREA)
    #     mask = remove_background(img)
    #     mask_coor = np.argwhere(mask >= 128)
    #     mask = cv2.resize(mask, img.shape[:2], cv2.INTER_NEAREST)
    #
    #     for j, (p1_y, p1_x) in enumerate(mask_coor):
    #         p1_y = np.clip(4*p1_y- K//2 , 0, H-K)
    #         p1_x = np.clip(4*p1_x- K//2, 0, W-K)
    #         cv2.rectangle(img, (p1_x, p1_y), (p1_x+K, p1_y+K), colors[j%2], -1)
    #
    #     cv2.imshow("Object area", img)
    #     cv2.waitKey(0)
    #
    #
    #
    # print(f"total - {time.time() - start:.5f}")
    # print(f"{(time.time() - start) / len(files):.3f} second per image")


    # for i, file in enumerate(files):
    #     img = cv2.resize(cv2.imread(file), (64, 64), cv2.INTER_AREA)
    #     img = apply_motion_blur(img, size=10, angle=130)
    #
    #     cv2.imshow("motionBlur", img)
    #     cv2.waitKey(300)

    # class_idx = 1
    # for i, file in enumerate(files):
    #     img = cv2.resize(cv2.imread(file).astype(np.float32)/255, (256, 256), cv2.INTER_AREA)
    #     img = RandomPolygon(img)
    #
    #     cv2.imshow("randompoly", img)
    #     cv2.waitKey(0)




    image_shape = (256,256,3)
    class_idx = 5
    root_dir = './dataset/mvtec_anomaly_detection'
    train_imgs, _ = Read_MVtec_image_files(root_dir, (256,256), mode='train', class_idx=class_idx, standardization=False)




    ''' ###################### Testing N-A dataset ############################ '''

    d_na32 = MVtec_Ano_Dataset(K=32, imgs=train_imgs)
    d_na64 = MVtec_Ano_Dataset(K=64, imgs=train_imgs)

    d_na32_ = torch.utils.data.DataLoader(d_na32, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    d_na64_ = torch.utils.data.DataLoader(d_na64, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)




    for batch_idx, (d32_na, d64_na) in enumerate(zip(d_na32_, d_na64_)):
        p32_n, p32_a = d32_na
        p64_n, p64_a = d64_na

        test_b2 = torchvision.utils.make_grid(p32_a, nrow=8, padding=6).permute(1, 2, 0).numpy()
        test_b4 = torchvision.utils.make_grid(p64_a, nrow=8, padding=6).permute(1, 2, 0).numpy()
        test_b3 = torchvision.utils.make_grid(p64_n, nrow=8, padding=6).permute(1, 2, 0).numpy()

        cv2.imshow("test_batch1",test_b2)
        cv2.imshow("test_batch3",test_b4)
        cv2.waitKey(0)








    # ''' ###################### Testing jigsaw dataset ############################ '''
    #
    # d_jig32 = MVtec_Jigsaw_Dataset(K=32, imgs=train_imgs)
    # d_jig64 = MVtec_Jigsaw_Dataset(K=64, imgs=train_imgs)
    #
    # d_jig32_ = torch.utils.data.DataLoader(d_jig32, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    # d_jig64_ = torch.utils.data.DataLoader(d_jig64, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    #
    #
    # for batch_idx, (d32_jig, d64_jig) in enumerate(zip(d_jig32_, d_jig64_)):
    #     p32_jig, cls32 = d32_jig
    #     p64_jig, cls64 = d64_jig
    #
    #     print(p32_jig.shape, cls32.shape)
    #     print(p64_jig.shape, cls64.shape)
    #
    #
    #
    #     for jig32, jig64 in zip(p32_jig, p64_jig):
    #         test_b2 = torchvision.utils.make_grid(jig32, nrow=3, padding=6).permute(1, 2, 0).numpy()
    #         test_b4 = torchvision.utils.make_grid(jig64, nrow=3, padding=6).permute(1, 2, 0).numpy()
    #
    #         cv2.imshow("test_batch1",test_b2)
    #         cv2.imshow("test_batch3",test_b4)
    #         cv2.waitKey(0)
