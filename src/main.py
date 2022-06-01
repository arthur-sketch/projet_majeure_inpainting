from re import M
import cv2
from matplotlib import pyplot as plt
from math import *
import numpy as np
import time



#init image
img = cv2.imread("images/img_landscape.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#param image
lignes, colonnes, couleur = img.shape



#paramètres zones a inpait
omegaX = 150
omegaY = 1950

omegaHeight = 100
omegaWidth = 1800

startPointOmega = (omegaX, omegaY)
endPointOmega = (omegaX + omegaWidth, omegaY + omegaHeight)


#creation omega
Omega = cv2.rectangle(img, startPointOmega, endPointOmega, color=(255,0,0), thickness=5)






#affichage
plt.figure()
plt.imshow(img)
plt.show()