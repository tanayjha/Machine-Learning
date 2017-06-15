import numpy as np
from sklearn import datasets, cross_validation, preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import svm

dataset = datasets.load_breast_cancer()

X = dataset.data
Y = dataset.target

X = preprocessing.scale(X)

X_Train, X_Test, Y_Train, Y_Test = cross_validation.train_test_split(X, Y, test_size=0.2)
	
model = LinearRegression()
model.fit(X_Train, Y_Train)
print("Linear Regression Score = ", model.score(X_Test, Y_Test))

b = np.round(model.predict(X_Test))

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_Train, Y_Train)

print("KNN Score = ", neigh.score(X_Test, Y_Test))

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_Test, Y_Test)

print("Decision Tree Score = ", clf.score(X_Test, Y_Test))


sv = svm.SVC()
clf = sv.fit(X_Test, Y_Test)

print("SVM Score = ", sv.score(X_Test, Y_Test))