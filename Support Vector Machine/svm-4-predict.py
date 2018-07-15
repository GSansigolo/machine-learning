'''
Created on Jul 15, 2018
Author: @G_Sansigolo
'''
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

class Support_Vector_Machine:
	def __init__(self, visualization=True):
		self.visualization = visualization
		self.colors = {1:'r',-1:'b'}
		if self.visualization:
			self.fig = plt.figure()
			self.ax = self.fig.add_subplot(1,1,1)

	def fit(self, data):
		self.data = data
		#{ ||w||: [w,b]}		
		opt_dict = {}
		trasforms = [[1,1],
			     [-1,1],
			     [-1,-1],
			     [1,-1]]
		all_data = []

		for yi in self.data:
			for featureset in self.data[yi]:
				for feature in featureset:
					all_data.append(feature)
		
		self.max_feature_value = max(all_data)
		self.min_feature_value = min(all_data)
		all_data = None

		step_sizes = [self.max_feature_value * 0.1,
			      self.max_feature_value * 0.01,
			      self.max_feature_value * 0.001]
		
		b_range_multiple = 5
		
		b_multiple = 5

		latest_optimum = self.max_fature_value*10

		for step in step_size:
			w = np.array([latest_optimum, latest_optimum])
			optimized = False
			while not optimize:
				for b in np.arange(-1*(self.max_feature_value*b_range_mutiple), self.max_feature_value*b_range_mutiple, step*b_multiple):
					for trasformation in trasforms:
						w_t = w*trasformation						
						found_option = True
						for i in self.data:
							for xi in self.data[i]:
								yi = i
								if not yi*(np.dot(w_t,xi)+b) >= 1:
									found_option = False
						if found_option:
							opt_dict[np.linalg.norm(w_t)] = [w_t,b]
					if w[0] < 0:
						optimized = True
						print("Optimized a step")
					else:
						w = w - step
					norms = sorted([n for n in opt_dict])
					opt_choice = opt_dict[norms[0]]
					
					#{ ||w||: [w,b]}
					self.w = opt_choice[0]
					self.b = opt_choice[1]

					latest_optimum = opt_choice[0][0]+step*2


	def predict(self, features):	
		# sign (X*W+B)
		classification = np.sign(np.dot(np.array(features), self.w)+self.b)

		if classification != 0 and self.vizualization:
			self.ax.scatter(features[0], features[1], s= 200, c= self.colors[classification])
		return classification

	def visualize(self):
		[[self.ax.scatter(x[0],x[1],s=100,color=self.color.colors[1]) for x in data_dict[i]] for i in data dict]
 
		def hyperplane(x,w,b,v):
			return (-w[0]*x-b+v / w[1])
			
	datarange = (self.min_feature_value*0.9, self.max_feature_value*1.1)
	hyp_x_min = 
	hyp_y_max = 



data_dict = {-1:np.array([[1,7], [2,8], [3,8]]), 1:np.array([[5,1], [6,-1], [7,3]])}





