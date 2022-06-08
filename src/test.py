import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("../images/landscape.jpg", 0)

plt.figure()
plt.imshow(img, "gray")
plt.show()








