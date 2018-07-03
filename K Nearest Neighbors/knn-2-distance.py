'''
Created on Jun 28, 2018
Author: @G_Sansigolo
'''
from math import sqrt
import numpy as np

plot1 = [1,3]
plot2 = [2,5]

euclidian_distance = sqrt((plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2)
#euclidian_distance = np.linalg.norm(np.array(features)-np.array(predict))

print(euclidian_distance)

