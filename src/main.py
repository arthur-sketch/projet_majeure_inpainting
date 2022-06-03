
from re import M
import cv2
from matplotlib import pyplot as plt
from math import *
import numpy as np
import time
import scipy as sp
from copy import deepcopy
# from LoadingProgress import loadingProgress


print('... Intialisation ...')


#init image
img_name = "img_landscape.jpg"
img = cv2.imread('../images/' + img_name, 0) # Lecture en gris

#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Lecture en couleur
#lignes, colonnes, couleur = img.shape # Lecture des params
lignes, colonnes = img.shape

#calculs grandeurs utiles

L = cv2.Laplacian(img, ddepth=3, ksize=5)       # définition de L
grad_Ny, grad_Nx = np.gradient(img)             # définition des gradients
norm = ( grad_Ny**2 + grad_Nx**2 )**(1/2)       # norme du gradient


#paramètres zones a inpait : (omegaY / Height ,omegaX / Width) 
omegaX = 150
omegaY = 1950
omegaHeight = 100
omegaWidth = 1800
startPointOmega = (omegaX, omegaY)
endPointOmega = (omegaX + omegaWidth, omegaY + omegaHeight)


for j in range(omegaX, omegaX+100):
    for i in range(omegaY, omegaY + 100):

        img[i,j] = 0

new_img = deepcopy(img)             
img_copy = deepcopy(img)



#Faire gaffe ici : cv2 dessine bien selon les axes x et y classiques, mais dans le parcours de la boucle, les axes 
#sont inversés (voir l'implémentation lignes 79/80) comme pour les grads

#creation omega
# Omega = cv2.rectangle(img, startPointOmega, endPointOmega, color=(255,0,0), thickness=5)







#init variables


dL = [ [ [0,0] for j in range(colonnes) ] for i in range(lignes) ]
dL = np.array(dL)

vecN_normed = [ [ [0,0] for j in range(colonnes) ] for i in range(lignes) ]
vecN_normed = np.array(vecN_normed)


epsilon = 0.00000001                                #pour ne pas diviser par 0 dans la norme
beta = np.zeros( (lignes, colonnes) )





#fonction utile
def getNormNabla(beta,i,j):
    
    if beta[i,j] > 0:

        val = sqrt( min( img_copy[i,j] - img_copy[i,j-1], 0 )**2  +  max( img_copy[i,j] - img_copy[i,j+1], 0 )**2      +     min( img_copy[i,j] - img_copy[i-1,j], 0 )**2  +  max( img_copy[i,j] - img_copy[i+1,j], 0 )**2 )

    else:

        val = sqrt( max( img_copy[i,j] - img_copy[i,j-1], 0 )**2  +  min( img_copy[i,j] - img_copy[i,j+1], 0 )**2      +     max( img_copy[i,j] - img_copy[i-1,j], 0 )**2  +  min( img_copy[i,j] - img_copy[i+1,j], 0 )**2 )


    return val





#paramètres algo

N = 50          # condition d'arret temporaire, nombre d'itération
count = 0       #nb d'itération dynamique

dT = 0.1        # pas de temps
delta = 0.1     #condition d'arret finale (pour l'instant pas utilisée)
It = 100        #init pour la condition d'arret



print('... Processing ...') 
# loadingProgress(count,N)

t1 = time.time()

while(count < N):
    
    img_copy = new_img

    for j in range(omegaX, omegaX + 100):
        
        for i in range(omegaY , omegaY + 100 ):
        
            dL[i,j] = [ L[i+1, j] - L[i-1, j], L[i, j+1] - L[i, j-1]  ]

            
            vecN_normed[i,j] = [ -grad_Ny[i,j], grad_Nx[i,j] ] / ( norm[i,j] + epsilon )


            beta[i,j] = np.dot( dL[i,j], vecN_normed[i,j] ) 
            norm_nabla = getNormNabla(beta,i,j) 
            

            It = beta[i,j] * norm_nabla

    
            new_img[i,j] = new_img[i,j] + dT * It


    L = cv2.Laplacian(new_img, ddepth=3, ksize=5)               # actualisation de L
    grad_Ny, grad_Nx = np.gradient(new_img)                     # définition des gradients
    norm = ( grad_Ny**2 + grad_Nx**2 )**(1/2)                   # norme du gradient
            
    count+=1
    print("count : ", count)
    # loadingProgress(count, N)


t2 = time.time()
print('\n... End processing ...')
print('\nProcess time : ' + str(t2-t1) + ' s\n')






#affichage
print('... Affichage resultats ...')
plt.figure()
plt.subplot(1,2,1)
plt.title('Image non retouchée')
plt.axis('off')
plt.imshow(img, 'gray')
plt.subplot(1,2,2)
plt.title('Image retouchée')
plt.axis('off')
plt.imshow(new_img, 'gray')
plt.show()


#sos