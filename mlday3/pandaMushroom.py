import pandas as pd  
import numpy as np  
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import svm


d = pd.read_csv('mushrooms.csv')
target = pd.DataFrame()

l = list(d)

print(l)
con = preprocessing.LabelEncoder()
for val in l:
	d[val] = con.fit_transform(d[val])

print(d.head())
target["class"] = d["class"]
d.drop('class', 1, inplace=True)

# d.drop("RI", 1, inplace=True)
# d.drop("Fe", 1, inplace=True)
# preprocessing.scale(d)

X = np.array(d)
Y = np.array(target)

X_Train, X_Test, Y_Train, Y_Test = cross_validation.train_test_split(d, target, test_size=0.25, random_state=5)


# model = SVR(kernel="linear")

# selector = RFE(model, 2)
# selector.fit(X_Train, Y_Train)
# print(selector.support_)
# print(selector.ranking_)

# model = LinearRegression()
# model.fit(X_Train, Y_Train)
# print("Linear Regression Score = ", model.score(X_Test, Y_Test))

# b = np.round(model.predict(X_Test))

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_Train, Y_Train)

X_Test.append(target["class"])

print(X_Test["class"])
# X_Test.loc[X_Test['class'] == some_value]
# print("KNN Score = ", neigh.score(X_Test, Y_Test))

# clf = tree.DecisionTreeClassifier()
# clf.fit(X_Train, Y_Train)

# print("Decision Tree Score = ", clf.score(X_Test, Y_Test))


# sv = svm.SVC()
# sv.fit(X_Train, Y_Train)

# print("SVC Score = ", sv.score(X_Test, Y_Test))

# sv = svm.SVR()

# model = SVR(kernel="linear")
# selector = RFE(sv, 5)
# selector.fit(X_Test, Y_Test)

# print(selector.support_)
# sv.fit(X_Train, Y_Train)

# print("SVR Score = ", sv.score(X_Test, Y_Test))