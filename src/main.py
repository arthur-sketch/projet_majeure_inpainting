
#main.py

#imports
from re import M
import cv2
from matplotlib import pyplot as plt
from math import *
import numpy as np
import time
import scipy as sp
from LoadingProgress import loadingProgress


print('... Intialisation ...')


#init image
img_name = "img_landscape.jpg"
img = cv2.imread("images/" + img_name, 0) # Lecture en gris
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Lecture en couleur
#lignes, colonnes, couleur = img.shape # Lecture des params
lignes, colonnes = img.shape # ptin moi j'ai tjrs écris np.shape(img) mais ozef


#calculs grandeurs utiles
L = cv2.Laplacian(img, ddepth=3, ksize=1) # définition de L
grad_Ny, grad_Nx = np.gradient(img) # définition des gradients
norm = ( grad_Ny**2 + grad_Nx**2 )**(1/2) # norme du gradient
new_img = img.copy() # nouvelle image pour pouvoir comparer les 2


#paramètres zones a inpait : (omegaY / Height ,omegaX / Width) 
omegaX = 150
omegaY = 1950
omegaHeight = 100
omegaWidth = 1800
startPointOmega = (omegaX, omegaY)
endPointOmega = (omegaX + 100, omegaY + 100)

#Faire gaffe ici : cv2 dessine bien selon les axes x et y classiques, mais dans le parcours de la boucle, les axes 
#sont inversés (voir l'implémentation lignes 79/80) comme pour les grads

#creation omega
# Omega = cv2.rectangle(img, startPointOmega, endPointOmega, color=(255,0,0), thickness=5)







#init variables
dL = np.zeros( (lignes, colonnes) )
vecN_normed = np.zeros( (lignes, colonnes) ) #jaime pas le nom de la variable mais N est deja pr le nombre d'itérations
epsilon = 0.00000001 #pour ne pas diviser par 0 dans la norme
beta = np.zeros( (lignes, colonnes) )
terme_chelou = np.zeros( (lignes, colonnes) )



#fonction utile
def getNormNabla(beta,i,j):
    
    if beta[i,j] > 0:

        val = sqrt( min( int(img[i,j]) - int(img[i,j-1]), 0 )**2  +  max( int(img[i,j]) - int(img[i,j+1]), 0 )**2      +     min( int(img[i,j]) - int(img[i-1,j]), 0 )**2  +  max( int(img[i,j]) - int(img[i+1,j]), 0 )**2 )

    else:
        val = sqrt( max( int(img[i,j]) - int(img[i,j-1]), 0 )**2  +  min( int(img[i,j]) - int(img[i,j+1]), 0 )**2      +     max( int(img[i,j]) - int(img[i-1,j]), 0 )**2  +  min( int(img[i,j]) - int(img[i+1,j]), 0 )**2 )

    return val





#paramètres algo
N = 3000 # condition d'arret temporaire, nombre d'itération
count = 0 # nb d'itération dynamique
dT = 0.1 # ce qu'ils ont mit dans le programme // ok pxl
delta = 0.1 #condition d'arret finale
It = 100 #init pour la condition d'arret



print('... Processing ...') 
loadingProgress(count,N)

t1 = time.time()

while(count < N):
    
    for j in range(omegaX, omegaX + 100):
        
        for i in range(omegaY , omegaY + 100 ):
        
            dL[i,j] = np.array( L[i+1, j] - L[i-1, j] , L[i, j+1] - L[i,j-1] )
            
            
            vecN_normed[i,j] = np.array(-grad_Ny[i,j], grad_Nx[i,j])
            vecN_normed[i,j] = vecN_normed[i,j] / ( norm[i,j] + epsilon )
            
            
            beta[i,j] = np.dot(dL[i,j],vecN_normed[i,j])
            norm_nabla = getNormNabla(beta,i,j) 
            #terme_chelou[i,j] = norm_nabla
            It = beta[i,j] * norm_nabla
            #print("AVANT : " + str(new_img[i,j]))
            new_img[i,j] = new_img[i,j] + dT * It
            #print("APRES : " + str(new_img[i,j]))
            
    count+=1
    loadingProgress(count, N)


t2 = time.time()
print('... End processing ...')
print('\nProcess time : ' + str(t2-t1) + ' s\n')
#################################################### fin boucle


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
