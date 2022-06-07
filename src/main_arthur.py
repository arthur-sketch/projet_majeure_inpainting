
#main.py

#imports
from pickletools import uint8
import cv2
from matplotlib import pyplot as plt
from math import *
import numpy as np
import time
import scipy as sp
from copy import deepcopy
from LoadingProgress import loadingProgress


print('... Intialisation ...')


#init image
img_name = "landscape.jpg"
img = cv2.imread("../images/" + img_name, 0)

img = img.astype(float)/255.0

lignes, colonnes = img.shape


#paramètres zones a inpait : (omegaY / Height ,omegaX / Width) 
omegaX = 175
omegaY = 175
omegaHeight = 50
omegaWidth = 50
startPointOmega = (omegaX, omegaY)
endPointOmega = (omegaX + omegaWidth, omegaY + omegaHeight)

for j in range(omegaX, omegaX+ omegaWidth):
    for i in range(omegaY, omegaY + omegaHeight):

        img[i,j] = 0



new_img = deepcopy(img) # nouvelle image pour pouvoir comparer les 2
new_img2 = deepcopy(img)

#calculs grandeurs utiles
# L = cv2.Laplacian(img, ddepth=CV_16S, ksize=7) # définition de L



grad_Ny, grad_Nx = np.gradient(img) # définition des gradients
norm = ( grad_Ny**2 + grad_Nx**2 )**(1/2) # norme du gradient



#creation omega
# Omega = cv2.rectangle(img, startPointOmega, endPointOmega, color=(255,0,0), thickness=5)




#init variables
dL = np.zeros( shape=(2,lignes, colonnes) ,dtype=object)
vecN_normed = np.zeros( shape=(2,lignes, colonnes),dtype= object ) #jaime pas le nom de la variable mais N est deja pr le nombre d'itérations
epsilon = 0.00001 #pour ne pas diviser par 0 dans la norme





#fonction utile
def getNormNabla(I,beta,i,j):
    
    if beta > 0:

        val = np.sqrt( min( round( float(I[i,j]), 8) - round( float(I[i,j-1]), 8), 0 )**2  +  max( round( float(I[i,j]), 8) - round( float(I[i,j+1]), 8), 0 )**2      +     min( round( float(I[i,j]), 8) - round( float(I[i-1,j]), 8), 0 )**2  +  max( round( float(I[i,j]), 8) - round( float(I[i+1,j]), 8), 0 )**2 )

    else:

        val = np.sqrt( max( round( float(I[i,j]), 8) - round( float(I[i,j-1]), 8), 0 )**2  +  min( round( float(I[i,j]), 8) - round( float(I[i,j+1]), 8), 0 )**2      +     max( round( float(I[i,j]), 8) - round( float(I[i-1,j]), 8), 0 )**2  +  min( round( float(I[i,j]), 8) - round( float(I[i+1,j]), 8), 0 )**2 )



    return val


def laplac(I,i,j) :
    return I[i+1,j].astype(float)+I[i-1,j].astype(float)+I[i,j+1].astype(float)+I[i,j-1].astype(float)-4*I[i,j].astype(float)


#paramètres algo
N = 600 # condition d'arret temporaire, nombre d'itération
count = 0 # nb d'itération dynamique
dT = 0.1 # ce qu'ils ont mit dans le programme // ok pxl
It = 100 #init pour la condition d'arret
affichage = False





print('... Processing ...') 
#loadingProgress(count,N)

t1 = time.time()

#pour mettre un flou

while(count < N):

    # try:


    grad_Ny, grad_Nx = np.gradient(new_img2) # définition des gradients
    norm = ( grad_Ny**2 + grad_Nx**2 )**(1/2) # norme du gradient
    
    for j in range(omegaX, omegaX + omegaWidth):
        
        for i in range(omegaY , omegaY + omegaHeight ):
        


            #on utilise la formule du laplacien discret 
            dL[:,i,j] = np.array( [ laplac(new_img2, i+1, j) - laplac(new_img2, i-1, j) , laplac(new_img2, i, j+1) - laplac(new_img2, i, j-1) ] )

            vecN_normed[:,i,j] = np.array([-grad_Ny[i,j],grad_Nx[i,j]])/ ( norm[i,j] + epsilon )
            
            
            beta = np.dot(dL[:,i,j],vecN_normed[:,i,j])
            norm_nabla = getNormNabla(new_img2, beta, i, j)
            It = beta * norm_nabla
            new_img[i,j] = new_img[i,j] + dT * It



            if affichage:

                print("\n\n(i,j) : ", i,j )
                print("dL : ",dL[:,i,j] )
                print("vecN : ", vecN_normed[:,i,j])
                print("beta : ",beta)
                print("normNab : ",norm_nabla) 
                print("It : ", It)
                print("AVANT : ",new_img2[i,j])
                print("APRES : ",new_img[i,j])



    new_img2 = deepcopy(new_img)
    #print(type(new_img2))

    count+=1
    # print(count)
    loadingProgress(count, N)

    # except:
        # loadingProgress(N,N)
        # break

loadingProgress(N,N)

new_img = (new_img*255.0).astype(np.uint8)


t2 = time.time()
print('... End processing ...')
print('\nProcess time : ' + str(round(t2-t1,3)) + ' s\n')


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
