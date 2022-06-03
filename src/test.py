from turtle import shape
import numpy as np




dL = [ [ [0,0] for j in range(2) ] for i in range(3) ]
dL = np.array(dL)
print(dL)

dL[1,1] = [2,2]

print("new : " ,dL)



# [ [[0 0][0 0][0 0]]
#   [[0 0][0 0][0 0]]
#   [[0 0][0 0][0 0]] ]



