import matplotlib.pyplot as plt
import os
import torch
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from unet import UNet
# from optparse import OptionParser


def PreparationDesDonnees(cow_i, crop_size):
    z = torch.Tensor(1,1,crop_size,crop_size).zero_()
    z[0:] = transforms.ToTensor()(cow_i)[0,:,:]
    return z


class Data():
    def __init__(self,dir_img):
        self.dir_img = dir_img

    def ExtractAsPIL(self):
        img = Image.open(self.dir_img)
        return {'image' : img, 'mask' : img}

    def Resize(self, size):
        img = self.ExtractAsPIL()['image']
        img = img.resize(size, Image.LANCZOS)
        return {'image' : img, 'mask' : img}



# def OptionCompilation():
#     parser = OptionParser()
#     parser.add_option("-m", "--model", type="str", dest="modelPath",
#                         help="Chemin relatif du fichier .tar")
#     parser.add_option("-t", "--targetPath", type="str", dest="targetPath",
#                         help="Chemin relatif du target .png")
#     (options, args) = parser.parse_args()
#     if options.modelPath and options.targetPath :
#         print("Model : "+options.modelPath)
#         print("Target path : "+options.targetPath)
#     else:
#         parser.error("Pas assez d arguments, demander --help")
#     return options





# Options de compilation
# options = OptionCompilation()
modelSaved = "/../../model/model_23.tar"#options.modelPath
imgPath = "./vache.png"#options.targetPath
diCow = Data(imgPath)
vache = diCow.Resize((256, 256))["image"]

model = UNet(in_channels=1, n_classes=2, padding=True, depth=4,
    up_mode='upsample', batch_norm=True)


# Calcul de la prediction du réseau
if (not os.path.isfile(os.getcwd()+modelSaved)):
    print("Attention : le modèle n'existe pas !")
else:
    model.load_state_dict(torch.load(os.getcwd()+modelSaved, map_location='cpu'))
    model.eval()

    img = torch.Tensor(1,1,256,256).zero_()
    img = PreparationDesDonnees(vache, 256)
    prediction = model(img)
    prediction = prediction[0,1:].detach().numpy()

    # plt.imshow(prediction[0,:,:])
    # plt.show()
