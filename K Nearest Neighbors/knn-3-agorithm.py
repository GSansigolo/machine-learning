'''
Created on Jul 28, 2018
Author: @G_Sansigolo
'''

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from collections import Counter
from matplotlib import style

style.use('fivethirtyeight')

dataset = {'k':[[1,2],[2,3],[1,1]], 'r':[[6,5],[7,7],[9,6]]}

new_features = [5,7]

#[[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
#plt.scatter(new_features[0], new_features[1], s=100)
#plt.show()

def k_nearest_neighbors(data, predict, k=3):
	if len(data) >= k:
		warnings.warn('K is set to a value less the total')
	
	return votes_result


