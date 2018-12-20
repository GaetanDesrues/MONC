#!/usr/bin/python
# -*- coding: utf-8 -*-

from code.unet import UNet
from code.imageLoader import DataLoader, Data, DataSample
from code.lossFiles import LossFile as LossModule
from code.optionCompil import OptionCompilation
import code.functions as fc
import os
import time
import configparser
# from progressbar import ProgressBar, Bar, SimpleProgress
import sys
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms as transforms


#Lecture des options shell
options = OptionCompilation()

# TensorBoardX pour les visualisations
writer = SummaryWriter('output/runs/Cells/'+options.fileName)

# Charge le fichier de configurations
config = configparser.ConfigParser()
config.read("config.cfg")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Cuda available : ", torch.cuda.is_available(),"  ---  Starting on", device)

model = UNet(in_channels=1, n_classes=2, depth=3, padding=True, up_mode='upsample').to(device)

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
# cows = DataLoader(
#     config['Model']['imgPath'],
#     config['Model']['maskPath'],
#     config['Model']['file'],
#     config['Model']['extension'])

cows = DataSample("./data/MMK")


# Définition des tailles
len_cows = len(cows)
epochs = options.epochs
minibatch = options.minibatch
crop_size = options.cropsize
len_train = int(float(options.lengthtrain) * len_cows)
len_train =  len_train - (len_train%minibatch)
len_test = len_cows - len_train
print("Taille du dataset : "+str(len_cows))
print("Taille du training set : "+str(len_train))
print("Taille du test set : "+str(len_test))

model.train()

# Définition du fichier d'erreurs argument=nom du fichier à aller cherche dans config
# lossFile = LossModule(str(config['Train']['lossFile']))

# pBarEpochs = ProgressBar(widgets = ['Epoques : ', SimpleProgress(), '   ', Bar()], maxval = epochs).start()
debut = time.time()

for epoch in range(epochs): # Boucle sur les époques
    # pBarEpochs.update(epoch)
    errMoy = 0
    for i in range(int(len_train/minibatch)): # parcourt chaque minibatch
        z, zy = fc.PreparationDesDonnees(i, minibatch, crop_size, cows)
        X = z.to(device)  # [N, 1, H, W]
        # Forward
        prediction = model(X) # [N, 2, H, W]
        prediction = transforms.ToTensor()(prediction)

        print(prediction[0,:,30:60,50:80])

        imgATester = prediction[0,:,:,:]
        mask = zy[0:]
        xx = vutils.make_grid(imgATester, normalize=True, scale_each=True)
        writer.add_image('d Entrainement '+str(i), xx, epoch)


        # zy = fc.CorrigerPixels(zy, crop_size, prediction.shape[2])
        y = zy.long().to(device)  # [N, H, W] with class indices (0, 1)
        # Calcul de l'erreur
        loss = F.cross_entropy(prediction, y)
        errMoy = errMoy + loss.item()
        # On initialise les gradients à 0 avant la rétropropagation
        optim.zero_grad()
        loss.backward()
        # Modification des poids suivant l'algorithme du gradient choisi
        optim.step()

    # xx = vutils.make_grid(prediction[0:], normalize=True, scale_each=True)
    # writer.add_image('Image training', xx, epoch)

    errMoy = errMoy/epochs

    # lossFile.addEpochLoss(epoch,loss.item())
    writer.add_scalar("Erreur sur l'entraînement par époque ", errMoy, epoch)


    # Test sur chaque image restante, cad non utilisée pour l'entrainement
    errTe = fc.Tester(len_test, cows, crop_size, device, model)
    writer.add_scalar("Erreur sur le test par époque ", errTe, epoch)


    # Tester sur une image pour visualiser la progression globale :
    imgATester, mask = fc.PreparationDesDonnees(len_cows-24, 1, crop_size, cows)
    xx = vutils.make_grid(imgATester, normalize=True, scale_each=True)
    writer.add_image('c Image visée', xx, epoch)
    # Prédiction du modèle
    maskPredit = fc.TesterUneImage(imgATester, model, device)
    x = vutils.make_grid(maskPredit, normalize=True, scale_each=True)
    writer.add_image("a Segmentation prédite par le réseau", x, epoch)
    # Masque
    y = vutils.make_grid(mask, normalize=True, scale_each=True)
    writer.add_image("b Masque (segmentation) de l'image visée", y, epoch)


    ### Fin de l'époque epoch


fin = time.time()
# pBarEpochs.finish()

timer = round((fin - debut)/60, 2)
print(" ------> Temps de l'apprentissage :", timer, "min.")

# if (config['Model']['saveModel']):
#     path = os.getcwd()+modelSaved
#     torch.save(model.state_dict(), path)

# lossFile.plotLoss()
# lossFile.Close()

writer.close()
