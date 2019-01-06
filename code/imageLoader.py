#!/usr/bin/python
# -*- coding: utf-8 -*-

from PIL import Image, ImageOps
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os

# Représente un couple vache+masque
class Data():
    def __init__(self,dir_img,dir_mask,file,extension):
        self.dir_img=dir_img
        self.dir_mask=dir_mask
        self.file=file    # i
        self.extension=extension
        # self.cwd = os.getcwd()
        # self.dic = {'image':self.img, 'mask':self.mask}

    def ExtractAsPIL(self):
        img = Image.open(self.dir_img+self.file+self.extension)
        mask = Image.open(self.dir_mask + self.file + self.extension)
        return {'image' : img, 'mask' : mask}

    def ExtractAsNP(self):
        doc = self.ExtractAsPIL()
        img = doc['image'] # [RGB]
        mask = doc['mask'] # [True ou False]
        return {'image' : np.asarray(img), 'mask' : np.asarray(mask)}

    def Plot(self, it): # it = 'image' ou 'mask'
        img = self.ExtractAsPIL()[it]
        img.show()
        img.close()

    def Resize(self, size):
        img = self.ExtractAsPIL()['image']
        mask = self.ExtractAsPIL()['mask']
        img = img.resize(size, Image.LANCZOS)
        mask = mask.resize(size, Image.LANCZOS)
        return {'image' : img, 'mask' : mask}

    def Pil2Np(self):
        return np.array(self)

    def Torch2Pil(self):
        return transforms.ToPilImage()(self)

# Représente l'ensemble du dataset
class DataLoader(Dataset):
    def __init__(self,dir_img,dir_mask,file,extension):
        self.dir_img=dir_img
        self.dir_mask=dir_mask
        self.file=file
        self.extension=extension

    def __len__(self):
        return len(os.listdir(self.dir_img))

    def __getitem__(self,i):
        return Data(self.dir_img, self.dir_mask, self.file+str(i), self.extension)

    def Plot(self,i): # Visualiser une image et son mask
        sample=self[i]
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        ax1.imshow(sample['image'])
        ax2 = fig.add_subplot(1,2,2)
        ax2.imshow(sample['mask'])
        plt.suptitle('Image '+str(i))
        plt.show()



def plplot(img1, img2="", title=""):
    if (img2!=""):
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        ax1.imshow(img1)
        ax2 = fig.add_subplot(1,2,2)
        ax2.imshow(img2)
        plt.suptitle(title)
        plt.show()
    else:
        plt.imshow(img1)
        plt.suptitle(title)
        plt.show()





#
# cows = DataLoader(
#     "./data/cow_img/",
#     "./data/cow_mask/",
#     "cow_",
#     ".png")
#
# cow = cows[10].ExtractAsPIL()
# print(cow["image"])
#
# diCow = cows[10].Resize((256, 256)) # Steven : à changer
# print(diCow['image'])
# #
# imageOriginal = ImageOps.grayscale(diCow['image'])
# maskOriginal = ImageOps.grayscale(diCow['mask'])
# print(imageOriginal)
# print(maskOriginal)
# # # imageOriginal = diCow['image']
# # # maskOriginal = diCow['mask']
# #
# X = transforms.ToTensor()(imageOriginal)
# y = transforms.ToTensor()(maskOriginal)
# print(X.shape)
# print(y.shape)
#
#
#
# print("  ")
#
#
# min = 0
# max = 0
# X=y
# for i in range(X.shape[1]):
#     for j in range (X.shape[2]):
#         x = X[0,i,j]
#         if x<min: min=x
#         if x>max: max=x
#
# print(min, max)
#
#
#
#
#





#
