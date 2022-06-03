
#main.py

#imports
from re import M
import cv2
from cv2 import CV_16S
from matplotlib import pyplot as plt
from math import *
import numpy as np
import time
import scipy as sp
#from LoadingProgress import loadingProgress


print('... Intialisation ...')


#init image
img_name = "img_landscape.jpg"
img = cv2.imread("images/" + img_name, 0) # Lecture en gris
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Lecture en couleur
#lignes, colonnes, couleur = img.shape # Lecture des params
lignes, colonnes = img.shape # ptin moi j'ai tjrs écris np.shape(img) mais ozef


#paramètres zones a inpait : (omegaY / Height ,omegaX / Width) 
omegaX = 150
omegaY = 1950
omegaHeight = 100
omegaWidth = 1800
startPointOmega = (omegaX, omegaY)
endPointOmega = (omegaX + 100, omegaY + 100)

for j in range(omegaX, omegaX + 100):
    for i in range(omegaY , omegaY + 100 ):
        img[i,j]=0
new_img = img.copy() # nouvelle image pour pouvoir comparer les 2

#calculs grandeurs utiles
# L = cv2.Laplacian(img, ddepth=CV_16S, ksize=7) # définition de L
# print(L)
# print(L.shape)
grad_Ny, grad_Nx = np.gradient(img) # définition des gradients
norm = ( grad_Ny**2 + grad_Nx**2 )**(1/2) # norme du gradient

#Faire gaffe ici : cv2 dessine bien selon les axes x et y classiques, mais dans le parcours de la boucle, les axes 
#sont inversés (voir l'implémentation lignes 79/80) comme pour les grads

#creation omega
# Omega = cv2.rectangle(img, startPointOmega, endPointOmega, color=(255,0,0), thickness=5)







#init variables
dL = np.zeros( shape=(2,lignes, colonnes) ,dtype=object)
vecN_normed = np.zeros( shape=(2,lignes, colonnes),dtype= object ) #jaime pas le nom de la variable mais N est deja pr le nombre d'itérations
epsilon = 0.00001 #pour ne pas diviser par 0 dans la norme
beta = np.zeros( shape=(lignes, colonnes))
terme_chelou = np.zeros( (lignes, colonnes) )
new_img2 = img.copy()




#fonction utile
def getNormNabla(beta,i,j):
    
    if beta[i,j] > 0:

        val = np.sqrt( min( float(new_img2[i,j]) - float(new_img2[i,j-1]), 0 )**2  +  max( float(new_img2[i,j]) - float(new_img2[i,j+1]), 0 )**2      +     min( float(new_img2[i,j]) - float(new_img2[i-1,j]), 0 )**2  +  max( float(new_img2[i,j]) - float(new_img2[i+1,j]), 0 )**2 )

    else:
        val = np.sqrt( max( float(new_img2[i,j]) - float(new_img2[i,j-1]), 0 )**2  +  min( float(new_img2[i,j]) - float(new_img2[i,j+1]), 0 )**2      +     max( float(new_img2[i,j]) - float(new_img2[i-1,j]), 0 )**2  +  min( float(new_img2[i,j]) - float(new_img2[i+1,j]), 0 )**2 )

    return val


def laplac(I,i,j) :
    return I[i+1,j].astype(float)+I[i-1,j].astype(float)+I[i,j+1].astype(float)+I[i,j-1].astype(float)-4*I[i,j].astype(float)


#paramètres algo
N = 20 # condition d'arret temporaire, nombre d'itération
count = 0 # nb d'itération dynamique
dT = 0.1 # ce qu'ils ont mit dans le programme // ok pxl
delta = 0.1 #condition d'arret finale
It = 100 #init pour la condition d'arret






print('... Processing ...') 
#loadingProgress(count,N)

t1 = time.time()


while(count < N):
    
    for j in range(omegaX, omegaX + 100):
        
        for i in range(omegaY , omegaY + 100 ):
        

            #on utilise la formule du laplacien discret L=I[i+1,j]+I[i-1,j]+I[i,j+1]+I[i,j-1]-4*I[i,j]
            dL[:,i,j] =np.array([laplac(new_img2,i+1, j) - laplac(new_img2,i-1, j),laplac(new_img2,i, j+1) - laplac(new_img2,i,j-1)])
            #print("dL : ",dL )

            
            vecN_normed[:,i,j] = np.array([-grad_Ny[i,j],grad_Nx[i,j]])/ ( norm[i,j] + epsilon )
            #print("vecN : ", vecN_normed[:,i,j])
            
            
            beta[i,j] = np.dot(dL[:,i,j],vecN_normed[:,i,j])
            #print(beta[i,j])
            norm_nabla = getNormNabla(beta,i,j)
            #print("normNab : ",norm_nabla) 
            It = beta[i,j] * norm_nabla
            #print("AVANT : " + str(new_img[i,j]))
            new_img[i,j] = new_img[i,j] + dT * It
            #print("APRES : " + str(new_img[i,j]))

    grad_Ny, grad_Nx = np.gradient(new_img) # définition des gradients
    norm = ( grad_Ny**2 + grad_Nx**2 )**(1/2) # norme du gradient
    new_img2 = new_img
    count+=1
    print(count)
    #loadingProgress(count, N)


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
plt.imshow(new_img, 'gray')
#plt.quiver(vecN_normed[0,:,:],-vecN_normed[1,:,:],color='r',units='xy',scale=10)
plt.show()
