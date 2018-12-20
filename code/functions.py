import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn.functional as F


def Tester(len_test, cows, crop_size, device, model):
    model.eval()
    error = 0
    len_cows = len(cows)
    for i in range(len_test):
        z, zy = PreparationDesDonnees(len_cows-i, 1, crop_size, cows)
        print(len_cows-i)
        X = z.to(device)
        prediction = model(X)
        # zy = CorrigerPixels(zy, crop_size, prediction.shape[2])
        y = zy.to(device).long()
        loss = F.cross_entropy(prediction, y)
        error = error + loss.item()
    model.train()
    return error/len_test




def TesterUneImage(img, model, device):
    model.eval()
    img2 = img.to(device)
    pred = model(img2)
    model.train()
    return pred



def PreparationDesDonnees(i, minibatch, crop_size, cows):
    z = torch.Tensor(minibatch,1,crop_size,crop_size).zero_() # 1:in_channels
    zy = torch.Tensor(minibatch,crop_size,crop_size).zero_()

    for m in range(minibatch): # On parcourt le training set batch par batch
        diCow = cows[i+m+1]
        imageOriginal = diCow["image"].resize((crop_size, crop_size), Image.LANCZOS) # Steven : à changer
        maskOriginal = diCow["mask"].resize((crop_size, crop_size), Image.LANCZOS) # Steven : à changer

        imageOriginal = ImageOps.grayscale(imageOriginal)
        maskOriginal = ImageOps.grayscale(maskOriginal)

        X = transforms.ToTensor()(imageOriginal)
        y = transforms.ToTensor()(maskOriginal)
        # print(X.shape, y.shape)

        z[m:]=X # m ème élément : Tensor X
        zy[m:]=y

    return z, zy



# def CorrigerPixels(zy, crop_size, predSize):
#     n_pi = (crop_size - predSize)//2
#     zy = zy[:,n_pi:-n_pi,n_pi:-n_pi] # Essayer d'exécuter en commantant pour voir l'erreur
#     return zy
