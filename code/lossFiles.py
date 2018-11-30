#!/usr/bin/python
# -*- coding: utf-8 -*-

# Classe pour gérer les fichiers d'erreurs, les afficher ...
import numpy as np
import matplotlib.pyplot as plt

class LossFile():
    def __init__(self,fileName):
        self.fileName=fileName
        with open("output/"+self.fileName, 'w') as fileObject:
            fileObject.write('Epoque '+'Erreur entrainement '+'Erreur test\n')

    def addEpochLoss(self, epoch, training_loss, test_loss=0):
        with open("output/"+self.fileName, 'a') as fileObject:
            fileObject.write(str(epoch) +" "+ str(training_loss) +" "+ str(test_loss)+ "\n")

    def plotLoss(self):
        with open("output/"+self.fileName, 'r') as fileObject:
            next(fileObject)
            datastr=fileObject.read()
            print(datastr)
            datastr=datastr[:-1]
            datastr=datastr.split('\n')
            data=np.empty((0,3))
            for a in datastr:
                b=a.split(" ")
                ligne=np.array([])
                for c in b:
                    ligne=np.append(ligne,[float(c)])
                data=np.append(data,[ligne],axis=0)
            print(data)
        plt.plot(data[:,1], label="Training")
        plt.plot(data[:,2], label="Test")
        plt.legend(bbox_to_anchor=(0.75, 0.5), loc=2, borderaxespad=0.)
        plt.title("Loss - Learning curve")
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.show()

    def Close(self):
        with open("output/"+self.fileName, 'a') as fileObject:
            fileObject.close()
