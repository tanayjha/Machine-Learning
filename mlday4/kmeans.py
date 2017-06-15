from sklearn.cluster import KMeans
import pandas as pd  
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
# from matplotlib.style import style

dataset = datasets.load_breast_cancer()

X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
	
clf = KMeans(n_clusters=3)

clf.fit(X)

l = clf.labels_

c = ['r', 'g', 'b']
print(clf.labels_)

print(clf.cluster_centers_)

for i in range(len(l)):
	plt.scatter(X[i][0], X[i][1], color=c[l[i]])

plt.show()