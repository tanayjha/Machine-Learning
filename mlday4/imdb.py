import pandas as pd
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
from sklearn import svm
import numpy as np
from sklearn.ensemble import RandomForestRegressor
d = pd.read_csv('movie_metadata.csv')

print(len(d))

# print(d.head())


columnList = list(d.columns)

for val in columnList: 
	try:
		d[val].fillna(value=d[val].mean(), inplace=True)
	except TypeError:
		d[val].fillna(value='unknown', inplace=True)
	d[val] = preprocessing.LabelEncoder().fit_transform(d[val])

ans = d['imdb_score']
d.drop(['imdb_score'], 1, inplace=True)

d.drop(['director_name'], 1, inplace=True)
d.drop(['actor_1_name'], 1, inplace=True)
d.drop(['actor_2_name'], 1, inplace=True)
d.drop(['actor_3_name'], 1, inplace=True)
d.drop(['movie_title'], 1, inplace=True)
d.drop(['title_year'], 1, inplace=True)
d.drop(['movie_imdb_link'], 1, inplace=True)

print(len(d))
# X_Train, X_Test, Y_Train, Y_Test = cross_validation.train_test_split(d, ans, test_size=0.5, random_state=5)

number_of_samples = len(d)
np.random.seed(0)
random_indices = np.random.permutation(number_of_samples)
num_training_samples = int(number_of_samples*0.3)
# print(d[[5000]])
x_train = d.loc[random_indices[:num_training_samples], :]
y_train=ans.loc[random_indices[:num_training_samples]]
x_test=d.loc[random_indices[num_training_samples:], :]
y_test=ans.loc[random_indices[num_training_samples:]]
y_Train=list(y_train)


model = LinearRegression()
model.fit(x_train, y_train)
print("Linear regression score = ", model.score(x_test, y_test))

model = svm.SVR(kernel="linear")
model.fit(x_train, y_train)
print("SVR score = ", model.score(x_test, y_test))

