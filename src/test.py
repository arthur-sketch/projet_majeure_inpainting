import numpy as np
import cv2


img  = cv2.imread("../images/gris.jpg")



for i in range(175,225):
    for j in range(175,225):

        img[i,j] = 0


cv2.imwrite("greyAndWhite.jpg", img)




