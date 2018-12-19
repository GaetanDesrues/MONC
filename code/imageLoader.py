#!/usr/bin/python
# -*- coding: utf-8 -*-

from PIL import Image, ImageOps
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




# cells = DataSample("./data/MMK")
# cells[1]["mask"].show()# .resize(size, Image.LANCZOS)















#
