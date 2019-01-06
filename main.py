#!/usr/bin/python
# -*- coding: utf-8 -*-

from code.unet import UNet
from code.imageLoader import DataLoader, Data
from code.lossFiles import LossFile as LossModule
from code.optionCompil import OptionCompilation
import code.functions as fc
import os
import time
import configparser
from progressbar import ProgressBar, Bar, SimpleProgress
import sys
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
import torchvision.utils as vutils


#Lecture des options shell
options = OptionCompilation() #Haha

# TensorBoardX pour les visualisations
writer = SummaryWriter('output/runs/test-07')#exp-29-11-test')
# arg : Rien pour le nom par défaut, comment='txt' pour ajouter un com à la fin

# Charge le fichier de configurations
config = configparser.ConfigParser()
config.read("config.cfg")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Cuda available : ", torch.cuda.is_available(),"  ---  Starting on", device)

model = UNet(in_channels=1, n_classes=2, padding=True, up_mode='upsample').to(device)

# Check si un modèle existe pour reprendre ou commencer l'apprentissage
# if (bool(config['Model']['saveModel'])):
#     modelSaved = config['Model']['fileName']
#     if (os.path.isfile(os.getcwd()+modelSaved)):
#         model.load_state_dict(torch.load(os.getcwd()+modelSaved))
#     else:
#         print("Attention : le modèle n'existe pas encore et va être créé !")

# Optimisateur pour l'algorithme du gradient
optim = torch.optim.SGD(model.parameters() , lr=1)
# optim = torch.optim.Adam(model.parameters() , lr=0.0001)

# Objet représentant les données
cows = DataLoader(
    config['Model']['imgPath'],
    config['Model']['maskPath'],
    config['Model']['file'],
    config['Model']['extension'])


# Taille totale du dataset
len_cows = len(cows)
print("len : "+str(len_cows))

epochs = options.epochs
len_train = int(float(options.lengthtrain) * len_cows)
crop_size = options.cropsize
minibatch = options.minibatch

model.train()

# erreurMiniBatch = []
# erreurEpoch = []

# Définition du fichier d'erreurs argument=nom du fichier à aller cherche dans config
lossFile = LossModule(str(config['Train']['lossFile']))

pBarEpochs = ProgressBar(widgets = ['Epoques : ', SimpleProgress(), '   ', Bar()], maxval = epochs).start()
debut = time.time()

for epoch in range(epochs): # Boucle sur les époques
    pBarEpochs.update(epoch)
    errMoy = 0
    for i in range(int(len_train/minibatch)): # parcourt chaque minibatch
        z, zy = fc.PreparationDesDonnees(i, minibatch, crop_size, cows)
        X = z.to(device)  # [N, 1, H, W]
        # Forward
        prediction = model(X) # [N, 2, H, W]


        # # Ajout d'une sigmoide en sortie :
        # mmm = torch.nn.Sigmoid()
        #
        # haha = mmm(prediction)
        # # print(prediction.shape)
        # min = 0
        # max = 0
        # for i in range(haha.shape[1]):
        #     for j in range (haha.shape[2]):
        #         x = haha[4,1,i,j]
        #         # print(x)
        #         if x<min: min=x
        #         if x>max: max=x
        #
        # print(min, max)


        zy = fc.CorrigerPixels(zy, crop_size, prediction.shape[2])
        y = zy.long().to(device)  # [N, H, W] with class indices (0, 1)
        # Calcul de l'erreur
        loss = F.cross_entropy(prediction, y)
        errMoy = errMoy + loss.item()
        # On initialise les gradients à 0 avant la rétropropagation
        optim.zero_grad()
        loss.backward()
        # Modification des poids suivant l'algorithme du gradient choisi
        optim.step()

    errMoy = errMoy/epochs

    lossFile.addEpochLoss(epoch,loss.item())
    writer.add_scalar("Erreur sur l'entraînement par époque ", errMoy, epoch)


    # Test sur chaque image restante, cad non utilisée pour l'entrainement
    errTe = fc.Tester(len_train-(len_train%minibatch), cows, crop_size, device, model)
    writer.add_scalar("Erreur sur le test par époque ", errTe, epoch)


    # Tester sur une image pour visualiser la progression globale :
    imgATester, mask = fc.PreparationDesDonnees(len_cows-38, 1, crop_size, cows)
    xx = vutils.make_grid(imgATester, normalize=True, scale_each=True)
    writer.add_image('Image visée', xx, epoch)
    # Prédiction du modèle
    maskPredit = fc.TesterUneImage(imgATester, model, device)
    x = vutils.make_grid(maskPredit[0,1,:,:], normalize=True, scale_each=True)
    writer.add_image("Segmentation prédite par le réseau", x, epoch)
    # Masque
    y = vutils.make_grid(mask, normalize=True, scale_each=True)
    writer.add_image("Masque (segmentation) de l'image visée", y, epoch)


    ### Fin de l'époque epoch


fin = time.time()
pBarEpochs.finish()

timer = round((fin - debut)/60, 2)
print(" ------> Temps de l'apprentissage :", timer, "min.")

# if (config['Model']['saveModel']):
#     path = os.getcwd()+modelSaved
#     torch.save(model.state_dict(), path)

# lossFile.plotLoss()
lossFile.Close()

writer.close()
