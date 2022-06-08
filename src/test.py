import numpy as np
import cv2


img = 128*np.ones(shape=(200,200))



for i in range(95,105):
    for j in range(95,105):

        img[i,j] = 0


cv2.imwrite("g.jpg", img)




