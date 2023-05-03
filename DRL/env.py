import sys
import json
import torch
import numpy as np
import argparse
import torchvision.transforms as transforms
import cv2
from DRL.ddpg import decode
from utils.util import *
from PIL import Image
from DRL.content_loss import *
from torchvision import transforms, utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

aug = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             ])

width = 128
convas_area = width * width

img_train = []
img_test = []
train_num = 0
test_num = 0

class Paint:
    def __init__(self, batch_size, max_step, loss_mode):
        self.batch_size = batch_size
        self.max_step = max_step
        self.action_space = (13)
        self.observation_space = (self.batch_size, width, width, 7)
        self.test = False
        self.loss_mode = loss_mode
        self.mask_train = []
        self.mask_test = []


    def load_monet_data(self):
        global train_num, test_num
        for i in range(7001):
            img_id = '%d' %(i+1)
            try:
                img = cv2.imread('./data/monet_style/' + img_id + '.jpg', cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img, (width, width))
                if i > 2000:
                    train_num += 1
                    img_train.append(img)
                    if self.loss_mode == 'cml1':
                            mask = get_l2_mask(torch.unsqueeze(torch.tensor(np.transpose(img.astype('float32'), (2, 0, 1))), 0) / 255).cpu()[:,0,:,:]
                            mask = mask.numpy() * 255
                            mask = mask.astype(np.uint8)
                            #print(mask.shape, img.shape, len(img_train), len(mask_train))
                            self.mask_train.append(mask)
                else:
                    test_num += 1
                    img_test.append(img)
                    if self.loss_mode == 'cml1':
                            mask = get_l2_mask(torch.unsqueeze(torch.tensor(np.transpose(img.astype('float32'), (2, 0, 1))), 0) / 255).cpu()[:,0,:,:]
                            mask = mask.numpy() * 255
                            mask = mask.astype(np.uint8)
                            #print(mask.shape, img.shape, len(img_test), len(mask_test), type(img[0,0,0]), type(mask[0,0,0]))
                            self.mask_test.append(mask)
            finally:
                if (i + 1) % 10000 == 0:
                    print('loaded {} images'.format(i + 1))
        print('finish loading data, {} training images, {} testing images'.format(str(train_num), str(test_num)))

    def load_data(self):
        # CelebA
        global train_num, test_num
        for i in range(20000):
            img_id = '%06d' % (i + 1)
            try:
                img = cv2.imread('../data/img_align_celeba/' + img_id + '.jpg', cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img, (width, width))
                if i > 2000:
                    train_num += 1
                    img_train.append(img)
                    if self.loss_mode == 'cml1':
                            mask = get_l2_mask(torch.unsqueeze(torch.tensor(np.transpose(img.astype('float32'), (2, 0, 1))), 0) / 255).cpu()[:,0,:,:]
                            mask = mask.numpy() * 255
                            mask = mask.astype(np.uint8)
                            #print(mask.shape, img.shape, len(img_train), len(mask_train))
                            self.mask_train.append(mask)
                else:
                    test_num += 1
                    img_test.append(img)
                    if self.loss_mode == 'cml1':
                            mask = get_l2_mask(torch.unsqueeze(torch.tensor(np.transpose(img.astype('float32'), (2, 0, 1))), 0) / 255).cpu()[:,0,:,:]
                            mask = mask.numpy() * 255
                            mask = mask.astype(np.uint8)
                            #print(mask.shape, img.shape, len(img_test), len(mask_test), type(img[0,0,0]), type(mask[0,0,0]))
                            self.mask_test.append(mask)
            finally:
                if (i + 1) % 10000 == 0:
                    print('loaded {} images'.format(i + 1))
        print('finish loading data, {} training images, {} testing images'.format(str(train_num), str(test_num)))

    def pre_data(self, id, test):
        if test:
            img = img_test[id]
        else:
            img = img_train[id]
        if not test:
            img = aug(img)
        img = np.asarray(img)
        return np.transpose(img, (2, 0, 1))

    def get_mask(self, id, test):
        if test:
            img = self.mask_test[id]
        else:
            img = self.mask_train[id]
        # if not test:
        #     img = aug(img)
        img = torch.tensor(img)
        return img.to(device)

    def reset(self, test=False, begin_num=False):
        self.test = test
        self.mask = None
        self.imgid = [0] * self.batch_size
        self.gt = torch.zeros([self.batch_size, 3, width, width], dtype=torch.uint8).to(device)
        if self.loss_mode == 'cml1':
            self.mask = torch.zeros([self.batch_size, 1, width, width], dtype=torch.uint8).to(device)
        for i in range(self.batch_size):
            if test:
                id = (i + begin_num)  % test_num
            else:
                id = np.random.randint(train_num)
            self.imgid[i] = id
            self.gt[i] = torch.tensor(self.pre_data(id, test))
            if self.loss_mode == 'cml1':
                self.mask[i] = self.get_mask(id, test)
        self.tot_reward = ((self.gt.float() / 255) ** 2).mean(1).mean(1).mean(1)
        self.stepnum = 0
        self.canvas = torch.zeros([self.batch_size, 3, width, width], dtype=torch.uint8).to(device)
        self.lastdis = self.ini_dis = self.cal_dis()
        return self.observation()

    def observation(self):
        # canvas B * 3 * width * width
        # gt B * 3 * width * width
        # T B * 1 * width * width
        ob = []
        T = torch.ones([self.batch_size, 1, width, width], dtype=torch.uint8) * self.stepnum
        if self.loss_mode == 'cml1':
            return torch.cat((self.canvas, self.gt, self.mask, T.to(device)), 1), self.mask # canvas, img, mask, T

        return torch.cat((self.canvas, self.gt, T.to(device)), 1), None # canvas, img, T

    def cal_trans(self, s, t):
        return (s.transpose(0, 3) * t).transpose(0, 3)

    def step(self, action):
        self.canvas = (decode(action, self.canvas.float() / 255) * 255).byte()
        self.stepnum += 1
        ob, mask = self.observation()
        done = (self.stepnum == self.max_step)
        reward = self.cal_reward() # np.array([0.] * self.batch_size)
        return ob.detach(), reward, np.array([done] * self.batch_size), None, mask

    def cal_dis(self):
        return (((self.canvas.float() - self.gt.float()) / 255) ** 2).mean(1).mean(1).mean(1)

    def cal_reward(self):
        dis = self.cal_dis()
        reward = (self.lastdis - dis) / (self.ini_dis + 1e-8)
        self.lastdis = dis
        return to_numpy(reward)
