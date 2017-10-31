# - coding: utf-8 -*-
# python 3.5

import numpy as np
import cv2
import os
import random

BKroot = "/home/shinpoi/dataset/backgrounds"
ITEMroot = "/home/shinpoi/dataset/humans/"

def histogram_matching(srcArr, dstArr, srcPNG=True):
    src_HSV = cv2.cvtColor(srcArr, cv2.COLOR_RGB2HSV)
    srcHist = cv2.calcHist((src_HSV,), (2,), None, (256,), (0, 256)).reshape((-1,))
    if srcPNG:
        srcHist[0] = 0
    srcHist /= sum(srcHist)
    srcHistMap = np.zeros(256, dtype=np.float32)
    for i in range(len(srcHist)):
        srcHistMap[i] = sum(srcHist[:i])

    dst_HSV = cv2.cvtColor(dstArr, cv2.COLOR_RGB2HSV)
    dstHist = cv2.calcHist((dst_HSV,), (2,), None, (256,), (0, 256)).reshape((-1,))
    dstHist /= sum(dstHist)
    dstHistMap = np.zeros(256, dtype=np.float32)
    for i in range(len(dstHist)):
        dstHistMap[i] = sum(dstHist[:i])

    HistMap = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        minMap = 1
        minTag = None
        for j in range(256):
            if minMap > abs(srcHistMap[i] - dstHistMap[j]):
                minMap = abs(srcHistMap[i] - dstHistMap[j])
                minTag = j
        HistMap[i] = minTag
        # flatten??? may be...
        if i > 100000:
            if HistMap[i] < HistMap[i-1]:
                HistMap[i] = HistMap[i-1]
            if HistMap[i] == HistMap[i-1] == HistMap[i-2] == HistMap[i-3]:
                HistMap[i] += 1

    for i in range(src_HSV.shape[0]):
        for j in range(src_HSV.shape[1]):
            if src_HSV[i, j, 2] == 0:
                continue
            else:
                src_HSV[i, j, 2] = HistMap[src_HSV[i, j, 2]]
    return cv2.cvtColor(src_HSV, cv2.COLOR_HSV2RGB)
    
def reshape_item(imgArr, maxsize=448, min_t=0.4, filip_rate=0.5):
    if imgArr.shape[0] > imgArr.shape[1]:
        max_t = (maxsize/imgArr.shape[1])*0.8 - min_t
    else:
        max_t = (maxsize/imgArr.shape[0])*0.8 - min_t

    times = min_t + random.random()*max_t
    imgArr = cv2.resize(imgArr, (int(imgArr.shape[1]*times), int(imgArr.shape[0]*times)))

    # flip
    if random.random() < filip_rate:
        imgArr = cv2.flip(imgArr, 1)
    return imgArr
    
    
class ImageGenerator(object):
    def __init__(self, bk_root, item_root, max_img_size=448, batch_size=16):
        self.bk_root = bk_root
        self.item_root = item_root
        self.batch_size = batch_size
        self.max_img_size = max_img_size
        self.bk_list = []
        self.images_pool = None
        self.items_pool_rgba = []
        self.n_bk = 0
        self.BKindex = None

    def init_all(self, backup=None):
        if not backup:
            self.init_backgrounds_pool()
        self.init_BKimages_pool(backup=backup)
        self.init_items_pool()
        self.init_BKindex()
        
    def init_backgrounds_pool(self): 
        for i in os.walk(self.bk_root):
            root, folder, files = i
            for f in files:
                if f.endswith(".jpg"):
                    dir_ = root + "/" + f
                    self.bk_list.append(dir_)
        self.n_bk = len(self.bk_list)
        print("init_backgrounds_pool() end")
    
    # sepcial
    def init_BKimages_pool(self, backup=None):
        self.bk_list = np.random.permutation(self.bk_list)
        if backup:
            bk = np.load(backup)
            self.n_bk = len(bk)
            return bk

        self.n_bk = 32000  # test mode
        max_img_size = self.max_img_size
        self.images_pool = np.zeros((self.n_bk, max_img_size, max_img_size, 3), dtype=np.uint8)
        n = 0
        for dir_ in self.bk_list[:self.n_bk]:
            img = cv2.imread(dir_)
            if img.shape[0] == 270:
                st = random.randint(0, 210) # 480-270
                img = cv2.resize(img[:, st: st+270], (max_img_size, max_img_size))
            elif img.shape[0] == 720:
                st = random.randint(0, 560) # 1280-720
                img = cv2.resize(img[:, st: st+720], (max_img_size, max_img_size))
            else:
                print("ignore %s. shape:(%d, %d)" % (dir_, img.shape[0], img.shape[1]))
                continue
            self.images_pool[n] = img
            n += 1
            if n%500 == 0:
                print("%d/%d" % (n, self.n_bk))
        if n != self.n_bk:
            print("has igroned images!!")
            self.n_bk = n
            self.images_pool = self.images_pool[:n]
        print("init_BKimages_pool() end, get %d images" % n)

    # special
    def init_items_pool(self):
        item_list = []
        for i in os.walk(self.item_root):
            root, folder, files = i
            for f in files:
                if f.endswith(".png"):
                    dir_ = root + "/" + f
                    item_list.append(dir_)
        self.items_pool_rgba = [[] for i in range(10)]

        for dir_ in item_list:
            if "blue_1" in dir_:
                self.items_pool_rgba[0].append(cv2.imread(dir_, -1))
            elif "blue_2" in dir_:
                self.items_pool_rgba[1].append(cv2.imread(dir_, -1))
            elif "blue_3" in dir_ or "blue_4" in dir_:
                self.items_pool_rgba[2].append(cv2.imread(dir_, -1))
            elif "blue_5" in dir_:
                self.items_pool_rgba[3].append(cv2.imread(dir_, -1))
            elif "blue_6" in dir_:
                self.items_pool_rgba[4].append(cv2.imread(dir_, -1))
            elif "orange_1" in dir_:
                self.items_pool_rgba[5].append(cv2.imread(dir_, -1))
            elif "orange_2" in dir_:
                self.items_pool_rgba[6].append(cv2.imread(dir_, -1))
            elif "orange_3" in dir_ or "orange_4" in dir_:
                self.items_pool_rgba[7].append(cv2.imread(dir_, -1))
            elif "orange_5" in dir_:
                self.items_pool_rgba[8].append(cv2.imread(dir_, -1))
            elif "orange_6" in dir_:
                self.items_pool_rgba[9].append(cv2.imread(dir_, -1))
        print("init_items_pool() end")
        
    
    def compose_img(self, item, background, hm_rate=0.8):
        rgba = item.copy()
        bk = background.copy()
        bk_rows, bk_cols, ch = background.shape
        if random.random() <= hm_rate:
            rgba[:, :, :3] = histogram_matching(rgba[:, :, :3], background)
        # insert coordinate
        hum_rows, hum_cols, ch = rgba.shape
        lim_rows = int((bk_rows - hum_rows)/2)
        lim_cols = bk_cols - hum_cols
        row_start = int(lim_rows*random.random()) + lim_rows
        col_start = int(lim_cols*random.random())

        # create mask
        mask = cv2.GaussianBlur(rgba[:, :, 3], (1, 1), 1)
        mask_inv = cv2.bitwise_not(mask)
        mask = np.array(mask, dtype=np.float32)/255
        mask_inv = np.array(mask_inv, dtype=np.float32)/255
        mask.resize((hum_rows, hum_cols, 1))
        mask_inv.resize((hum_rows, hum_cols, 1))
        mask = np.concatenate((mask, mask, mask), axis=2)
        mask_inv = np.concatenate((mask_inv, mask_inv, mask_inv), axis=2)

        # insert
        # print(row_start, col_start, hum_rows, hum_cols)
        bk_part = bk[row_start:row_start+hum_rows, col_start:col_start+hum_cols]
        bk[row_start:row_start + hum_rows, col_start:col_start + hum_cols] = \
np.array(bk_part * mask_inv + rgba[:, :, :3] * mask, dtype=np.uint8)

        # t
        x = (2*row_start + hum_rows)/bk_rows/2
        y = (2*col_start + hum_cols)/bk_cols/2
        h = hum_rows/bk_rows
        w = hum_cols/bk_cols
        t = [{'x':x, 'y':y, 'w':w, 'h':h, 'label':None, 'one_hot_label':None }]
        
        return bk, t

    def init_BKindex(self):
        self.BKindex = np.arange(0, self.n_bk, 1, dtype=np.int32)
        self.BKindex = np.random.permutation(self.BKindex)

    def create_batch(self, batch_size=None, index=0, reuse=0, reuserate=2, img_size=None):
        if not img_size:
            img_size = self.max_img_size
        if not batch_size:
            batch_size = self.batch_size

        if 2*batch_size > self.n_bk:
            self.init_BKindex()
            if (reuse != 0) and (reuse % reuserate == 0):
                self.init_BKimages_pool()
            index = 0
            reuse += 1

        for i in range(batch_size, img_size):
            batch = np.zeros((batch_size, 3, img_size, img_size), dtype=np.float32)
            tt = []
            BKimg_list = self.BKindex[index: index+batch_size]
            index += batch_size
            n = 0
            for i in BKimg_list:
                cla = random.randint(0, 9)
                item = random.sample(self.items_pool_rgba[cla], 1)
                x, t = self.compose_img(item[0], self.images_pool[i])
                if not x.shape[0] == img_size:
                    x = cv2.resize(x, (img_size, img_size))
                batch[n] = x.transpose((2, 0, 1))
                t[0]['label'] = cla
                oh = np.zeros(10, dtype=np.float32)
                oh[cla] = 1
                t[0]['one_hot_label'] = oh
                tt.append(t)
                n += 1
            return batch/255, tt
    """
    def save_imgspool_backup(self):
        print("start backup")
        np.save("img_pools.npy", self.images_pool)
        print("backup: img_pools.npy")
    """



"""
import image_generator_2 as ig
gen = ig.ImageGenerator(ig.BKroot, ig.ITEMroot)
gen.init_all()
x, t = gen.create_batch()
"""



