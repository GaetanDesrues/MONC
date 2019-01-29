import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import torch
import torch.nn.functional as F
from random import uniform

def Tester(test_idx, cows, crop_size, a, device, model):
    model.eval()
    error = 0
    for i in range(len(test_idx)):
        z, zy = PreparationDesDonnees(i, 1, crop_size, cows, a, test_idx)
        X = z.to(device)
        prediction = model(X)
        zy = CorrigerPixels(zy, crop_size, prediction.shape[2])
        y = zy.to(device).long()
        loss = F.cross_entropy(prediction, y)
        error = error + loss.item()
    model.train()
    return error/len(test_idx)




def TesterUneImage(img, model, device):
    model.eval()
    img2 = img.to(device)
    pred = model(img2)
    model.train()
    return pred



def PreparationDesDonnees(i, minibatch, crop_size, cows, a, train_idx):
    z = torch.Tensor(minibatch,1,crop_size,crop_size).zero_() # 1:in_channels
    zy = torch.Tensor(minibatch,crop_size,crop_size).zero_()
#coucou
    for m in range(minibatch): # On parcourt le training set batch par batch
        cow_i = cows[train_idx[i+m]]
        if a<=1: cow_i.Rotation(uniform(1,180))
        elif a<=2: cow_i.SymmetryLeftRight()
        elif a<=3: cow_i.Blur()
        #elif a<=4: cow_i.RandomCrop()
        diCow = cow_i.Resize((crop_size, crop_size))

        imageOriginal = ImageOps.grayscale(diCow['image'])
        maskOriginal = ImageOps.grayscale(diCow['mask'])
        # imageOriginal = diCow['image']
        # maskOriginal = diCow['mask']

        X = transforms.ToTensor()(imageOriginal)
        y = transforms.ToTensor()(maskOriginal)
        # print(X.shape, y.shape)

        z[m:]=X # m ème élément : Tensor X
        zy[m:]=y

    return z, zy



def CorrigerPixels(zy, crop_size, predSize):
    # n_pi = (crop_size - predSize)//2
    # zy = zy[:,n_pi:-n_pi,n_pi:-n_pi] # Essayer d'exécuter en commantant pour voir l'erreur
    return zy



def recadrage(image):
    min = 0
    max = 0
    for m in range(image.shape[0]):
        for i in range(image.shape[1]):
            for j in range (image.shape[2]):
                x = image[m,0,i,j]
                if x<min: min=x
                if x>max: max=x
    return min, max










#
