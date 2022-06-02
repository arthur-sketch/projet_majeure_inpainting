from re import M
import cv2
from matplotlib import pyplot as plt
from math import *
import numpy as np
import time
import scipy as sp



#init image
img = cv2.imread("images/img_landscape.jpg", 0)


#Si en couleur : 
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#param image
# lignes, colonnes, couleur = img.shape



#Sinon

lignes, colonnes = img.shape

#paramètres zones a inpait
omegaX = 150
omegaY = 1950

omegaHeight = 100
omegaWidth = 1800

startPointOmega = (omegaX, omegaY)
endPointOmega = (omegaX + omegaWidth, omegaY + omegaHeight)


#creation omega
Omega = cv2.rectangle(img, startPointOmega, endPointOmega, color=(255,0,0), thickness=5)


#paramètres algo + init variables

N = 100
dT = 0.1 #ce qu'ils ont mit dans le programme


dL = np.zeros( (lignes, colonnes) )
vecN_normed = np.zeros( (lignes, colonnes) ) #jaime pas le nom de la variable mais N est deja pr le nombre d'itérations
epsilon = 0.00000001 #pour ne pas diviser par 0 dans la norme
beta = np.zeros( (lignes, colonnes) )

terme_chelou = np.zeros( (lignes, colonnes) )



#fonction utile

def getValCheloue(i,j):


    if beta[i,j] > 0:

        val = sqrt( min( img[i,j] - img[i,j-1], 0 )**2  +  max( img[i,j] - img[i,j+1], 0 )**2      +     min( img[i,j] - img[i-1,j], 0 )**2  +  max( img[i,j] - img[i+1,j], 0 )**2 )


    else:
        val = sqrt( max( img[i,j] - img[i,j-1], 0 )**2  +  min( img[i,j] - img[i,j+1], 0 )**2      +     max( img[i,j] - img[i-1,j], 0 )**2  +  min( img[i,j] - img[i+1,j], 0 )**2 )


    return val

# def iteration():

#en fait du coup jpense tu peux faire une boucle while pr l'itération mm pas besoin de fonction


################################################# début fonction

L = cv2.Laplacian(img, ddepth=3, ksize=1) #pas trop compris les arguments utilisés
grad_Ny, grad_Nx = np.gradient(img)
norm = ( grad_Ny**2 + grad_Nx**2 )**(1/2)

for i in range(omegaX +1, omegaX + omegaWidth -1):
    for j in range(omegaY+1, omegaY + omegaHeight-1):


        dL[i,j] = np.array( L[i+1, j] - L[i-1, j] , L[i, j+1] - L[i,j-1] )
        vecN_normed[i,j] = np.array( (-grad_Ny[i,j] + grad_Nx[i,j]) / ( norm[i,j] + epsilon ) )


        valCheloue = getValCheloue(i,j)
        terme_chelou[i,j] = valCheloue


beta = dL*vecN_normed



img = img + dT*terme_chelou










#################################################### fin fonction


#affichage
plt.figure()
plt.subplot(1,2,1)
plt.imshow(img, 'gray')

plt.subplot(1,2,2)
plt.imshow(L, 'gray')

plt.show()