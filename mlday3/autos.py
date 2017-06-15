import pandas as pd  
import numpy as np  
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import svm
from dateutil import parser
import pickle
# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

d = pd.read_csv('autos.csv', encoding="cp1252")
target = pd.DataFrame()

print(d.info())

d['notRepairedDamage'].fillna(value='not-declared', inplace=True)
d.dropna(inplace=True)

print(d.describe())

# Drop pictures
d = d.drop(['nrOfPictures'], 1)

# Check column and name and decide usefullness
print(d.columns)
print(d.name)
print(d.name.unique())

# High so drop
print(len(d.name.unique()))
d = d.drop(['name'], 1)


print(d.seller.unique()) # Only 2 unique

print(d.groupby('seller').size())  # Mostly privat so 3 anomaly, so we delete those 3 rows and then drop column

d = d[d['seller'] != 'gewerblich']
d = d.drop(['seller'], 1)


print(d.dateCreated.describe()) # format of date created



ageCol = []
for date in d.dateCreated:
	temp = parser.parse(date)
	ageCol.append(temp.year)

# Creating a new field
d['ageOfVehicle'] = ageCol - d.yearOfRegistration

# Can also combine crawled and last seen ad


# print(d.ageOfVehicle.describe())   # Check for anomaly


d = d[(d['ageOfVehicle'] > 1) & d['ageOfVehicle'] < 50]


d = d.drop(['offerType', 'dateCrawled', 'dateCreated', 'lastSeen'], 1)

# Separate ouput

target = d["price"]

# Encoding integers changes the values but keeps the distance same
for col in d.columns:
	d[col] = preprocessing.LabelEncoder().fit_transform(d[col])


d.drop('price', 1, inplace=True)


l = list(d)
print(l)

# print("NAN = ", ans)
X_Train, X_Test, Y_Train, Y_Test = cross_validation.train_test_split(d, target, test_size=0.2, random_state=5)

print(X_Train.head())
# model = SVR(kernel="linear")

# selector = RFE(model, 8)
# selector.fit(d, target)
# print(selector.support_)
# print(selector.ranking_)

# start = time.time()
model = LinearRegression()
model.fit(X_Train, Y_Train)

print(len(d))
# with open('linearReg.picke', 'wb') as f:
# 	pickle.dump(model, f)

# pickle_in = open('linearReg.pickle', 'rb')
# model = pickle.load(pickle_in)

# print(model.feature_importances_)

# print(time.time() - start)
print("Linear Regression Score = ", model.score(X_Test, Y_Test))


clf = RandomForestRegressor()
# model = SVR(kernel="linear")
# selector = RFE(sv, 5)
# selector.fit(X_Test, Y_Test)

# print(selector.support_)

clf.fit(X_Train, Y_Train)
print("Random Forest Score = ", clf.score(X_Test, Y_Test))