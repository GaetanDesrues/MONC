# Projet de segmentation d'images par deep learning

***Implémentation du réseau de convolution Unet pour la segmentation d'images IRM***

- Données : ./data
Vaches ou Sarcômes

- Fichiers sources : ./[code](https://github.com/GaetanDesrues/MONC2/tree/gaetan/code)

- Fichiers de sortie : ./output
  - Runs tensorboarad : ./output/runs
  - Saved models : ./output/model

- Paramètres makefile : époques, mini-batch, len train set, learning rate

### Compilation
- CPU : make main
- Plafrim : make plaf
- Tester sur une image : python3 test.py

### Affichage des résultats
tensorboard --logdir ./output/runs --port 6006


