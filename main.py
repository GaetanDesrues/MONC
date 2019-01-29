#!/usr/bin/python
# -*- coding: utf-8 -*-

from code.unet import UNet
from code.imageLoader import DataLoader, Data, DataSample
from code.optionCompil import OptionCompilation
import code.functions as fc
import code.loss as EssaiLoss
import os
import time
import configparser
from progressbar import ProgressBar, Bar, SimpleProgress
import sys
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
import random
from random import uniform


#Lecture des options shell
options = OptionCompilation()

# TensorBoardX pour les visualisations
writer = SummaryWriter('output/runs/try-15E-1M-5lr')

# Charge le fichier de configurations
config = configparser.ConfigParser()
config.read("config.cfg")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Cuda available : ", torch.cuda.is_available(),"  ---  Starting on", device)

model = UNet(in_channels=1, n_classes=2, padding=True, depth=4,
    up_mode='upsample', batch_norm=True).to(device)

# Check si un modèle existe pour reprendre ou commencer l'apprentissage
# if (bool(config['Model']['saveModel'])):
#     modelSaved = config['Model']['fileName']
#     if (os.path.isfile(os.getcwd()+modelSaved)):
#         model.load_state_dict(torch.load(os.getcwd()+modelSaved))
#     else:
#         print("Attention : le modèle n'existe pas encore et va être créé !")

# Optimisateur pour l'algorithme du gradient
optim = torch.optim.SGD(model.parameters(), lr=5)# lr_scheduler
# optim = torch.optim.Adam(model.parameters() , lr=0.0005)

# Objet représentant les données
cows = DataLoader(
    config['Model']['imgPath'],
    config['Model']['maskPath'],
    config['Model']['file'],
    config['Model']['extension'])
# cows = DataSample("./data/MMK")


# Définition des tailles
len_cows = len(cows)-1
epochs = options.epochs
minibatch = options.minibatch
crop_size = options.cropsize
len_train = int(float(options.lengthtrain) * len_cows)
len_train =  len_train - (len_train%minibatch)
len_test = len_cows - len_train
print("Taille du dataset : "+str(len_cows))
print("Taille du training set : "+str(len_train))
print("Taille du test set : "+str(len_test))

# Une image non utilisée pour tester :
seuleToute = cows[len_cows]
diCow = seuleToute.Resize((crop_size, crop_size))
seuleToute = diCow['image']
imageOriginal = ImageOps.grayscale(seuleToute)
seuleToute = transforms.ToTensor()(imageOriginal)
# print(seuleToute.shape)

model.train()

# erreurMiniBatch = []
# erreurEpoch = []

indices = list(range(1,len_cows))
random.shuffle(indices)
test_idx, train_idx  = indices[len_train:], indices[:len_train]

pBarEpochs = ProgressBar(widgets = ['Epoques : ', SimpleProgress(), '   ', Bar()], maxval = epochs).start()
debut = time.time()

for epoch in range(epochs): # Boucle sur les époques
    pBarEpochs.update(epoch)
    errMoy = 0
    for i in range(1,int(len_train/minibatch)): # parcourt chaque minibatch
        a = uniform(0,5)
        print(i)
        #a = 10
        z, zy = fc.PreparationDesDonnees(i, minibatch, crop_size, cows, a, train_idx)
        X = z.to(device)  # [N, 1, H, W]
        # Forward
        prediction = model(X) # [N, 2, H, W]

        # transforms.ToTensor()(prediction.detach().numpy())
        # prediction = torch.nn.Sigmoid()(prediction)

        # min, max = fc.recadrage(prediction)
        # print(min, max)


        # zy = fc.CorrigerPixels(zy, crop_size, prediction.shape[2])
        y = zy.long().to(device)  # [N, H, W] with class indices (0, 1)
        # Calcul de l'erreur
        LOSS = torch.nn.MSELoss()
        # loss = F.cross_entropy(prediction, y)

        # loss = EssaiLoss.dice_loss2(y, prediction)
        loss = LOSS(prediction[:,1,:,:], y.float())
        errMoy = errMoy + loss.item()
        # On initialise les gradients à 0 avant la rétropropagation
        optim.zero_grad()
        loss.backward()
        # Modification des poids suivant l'algorithme du gradient choisi
        optim.step()

    errMoy = errMoy/epochs


    writer.add_scalar("Erreur sur l'entraînement par époque ", errMoy, epoch)


    # Test sur chaque image restante, cad non utilisée pour l'entrainement
    a = uniform(0,5)
    #a = 10
    errTe = fc.Tester(test_idx, cows, crop_size, a, device, model)
    writer.add_scalar("Erreur sur le test par époque ", errTe, epoch)


    # Tester sur une image pour visualiser la progression globale :
    a = uniform(0,5)
    #a = 10
    imgATester, mask = fc.PreparationDesDonnees(51, 1, crop_size, cows, a, train_idx)
    xx = vutils.make_grid(imgATester, normalize=True, scale_each=True)
    writer.add_image('Image visée', xx, epoch)
    # Prédiction du modèle
    maskPredit = fc.TesterUneImage(imgATester, model, device)
    # x = vutils.make_grid(maskPredit[0,0,:,:], normalize=True, scale_each=True)
    # writer.add_image("Segmentation prédite par le réseau1", x, epoch)

    # maskPredit = fc.TesterUneImage(imgATester, model, device)
    x = vutils.make_grid(maskPredit[0,1,:,:], normalize=True, scale_each=True)
    writer.add_image("Segmentation prédite par le réseau", x, epoch)

    # Masque
    y = vutils.make_grid(mask, normalize=True, scale_each=True)
    writer.add_image("Masque (segmentation) de l'image visée", y, epoch)


    imgg = torch.Tensor(1,1,crop_size,crop_size).zero_()
    # print(seuleToute.shape)
    seuleToute = seuleToute[0,:,:]
    imgg[0:] = seuleToute
    mask = fc.TesterUneImage(imgg, model, device)
    seuleToute = vutils.make_grid(seuleToute, normalize=True, scale_each=True)
    mask = vutils.make_grid(mask[0,1,:,:], normalize=True, scale_each=True)
    writer.add_image("Image", seuleToute, epoch)
    writer.add_image("Prediction", mask, epoch)

    ### Fin de l'époque epoch

fin = time.time()
pBarEpochs.finish()

timer = round((fin - debut)/60, 2)
print(" ------> Temps de l'apprentissage :", timer, "min.")

# if (config['Model']['saveModel']):
path = os.getcwd()+"/output/model/50e_dice_SGD(5e-3).tar"#modelSaved
#torch.save(model.state_dict(), path)

writer.close()
