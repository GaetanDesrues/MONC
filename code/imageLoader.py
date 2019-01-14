#!/usr/bin/python
# -*- coding: utf-8 -*-

from PIL import Image, ImageOps
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os
from random import uniform

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


    def Rotation(self, degree):
        img = self.ExtractAsPIL()['image']
        mask = self.ExtractAsPIL()['mask']
        img = img.rotate(degree)
        mask = mask.rotate(degree)
        return {'image' : img, 'mask' : mask}

    def SymmetryLeftRight(self):
        img = self.ExtractAsPIL()['image']
        mask = self.ExtractAsPIL()['mask']
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return {'image' : img, 'mask' : mask}

    def RandomCrop(self):
        img = self.ExtractAsPIL()['image']
        mask = self.ExtractAsPIL()['mask']
        img_size = img.img_size
        mask_size = mask.img_size
        #Def d'une box
        C = 1.5
        D = 0.5
        # x = im_size[0]/3
        # y = im_size[1]/3
        # Position aléatoire de la fenêtre de crop comprise dans une bande centrale de l'image
        x = uniform(im_size[0]/10,im_size[0]/C)
        y = uniform(im_size[1]/3,2/3*im_size[1])
        width = C*x
        heigt = D*y
        box = (x,y,width,heigt)
        img = img.crop(box)
        mask = mask.crop(box)
        img = img.resize(im_size, Image.LANCZOS)
        mask = mask.resize(mask_size, Image.LANCZOS)
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
        # if (i == len(os.listdir(self.dir_img))-1): print("Hehehe")
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



def listData(path):
    # Crée un fichier lD.txt qui regroupe les chemins de tous les couples image+mask
    lec_data = os.system("ls -R "+path+" > listeData.txt")
    if (lec_data != 0):
        print("Mauvaise lecture des données.")
    else:
        ld = open("listeData.txt", "r")
        ld2 = open("lD.txt", "w")

        liste = ld.read().split("\n")
        dossier = "None"
        nb = 0

        for i, valeur in enumerate(liste):
            dossierTrouve = valeur.find(path)
            if (dossierTrouve == 0):
                dossier = valeur[len(path)+1 : -1]
            elif (dossier != "None"):
                if (valeur.find("image") == 0):
                    nb = nb + 1
                    ld2.write(path+"/"+dossier+"/"+valeur+"  "+path+"/"+dossier+"/"+valeur.replace("image","mask")+"\n")
        print(str(nb)+" fichiers")
        ld.close()
        ld2.close()




class DataSample():
    def __init__(self, path):
        # Crée un fichier listeData.txt qui regroupe les chemins de tous les couples image+mask
        lec_data = os.system("ls -R "+path+" > lD.txt")
        if (lec_data != 0):
            print("Mauvaise lecture des données.")
        else:
            ld = open("lD.txt", "r")
            liste = ld.read().split("\n")
            dossier = "None"
            self.nb = 0
            self.files = []

            for valeur in liste:
                dossierTrouve = valeur.find(path)
                if (dossierTrouve == 0):
                    dossier = valeur[len(path)+1 : -1]
                elif (dossier != "None"):
                    if (valeur.find("image") == 0):
                        self.nb = self.nb + 1
                        self.files.append(path+" "+dossier+" "+valeur+"\n")
            # print(str(self.nb)+" fichiers")

            ld.close()
            os.system("rm lD.txt")

    def __len__(self):
        return self.nb


    def __getitem__(self, lig):
        path, dossier, id = self.files[lig].split(" ")

        img = Image.open(path+"/"+dossier+"/"+id.replace("\n",""))
        mas = Image.open(path+"/"+dossier+"/"+id.replace("image_", "mask_").replace("\n",""))

        return {"image" : img, "mask" : mas}



#
