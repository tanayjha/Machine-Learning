import pandas as pd  
import numpy as np  
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import svm


d = pd.read_csv('glass.csv')
#target = pd.DataFrame()
# print(d.tail())


target = d["Type"]
d.drop('Type', 1, inplace=True)

# d.drop("RI", 1, inplace=True)
# d.drop("Fe", 1, inplace=True)
X = np.array(d)
Y = np.array(target)


X_Train, X_Test, Y_Train, Y_Test = cross_validation.train_test_split(d, target, test_size=0.25, random_state=5)

print(X_Train.head())
# model = SVR(kernel="linear")

# selector = RFE(model, 8)
# selector.fit(d, target)
# print(selector.support_)
# print(selector.ranking_)

model = LinearRegression()
model.fit(X_Train, Y_Train)
print("Linear Regression Score = ", model.score(X_Test, Y_Test))

b = np.round(model.predict(X_Test))

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_Train, Y_Train)

print("KNN Score = ", neigh.score(X_Test, Y_Test))

clf = tree.DecisionTreeClassifier()
clf.fit(X_Train, Y_Train)

print("Decision Tree Score = ", clf.score(X_Test, Y_Test))


sv = svm.SVC()
sv.fit(X_Train, Y_Train)

print("SVC Score = ", sv.score(X_Test, Y_Test))

sv = svm.SVR()
# model = SVR(kernel="linear")
# selector = RFE(sv, 5)
# selector.fit(X_Test, Y_Test)

# print(selector.support_)
sv.fit(X_Train, Y_Train)

print("SVR Score = ", sv.score(X_Test, Y_Test))