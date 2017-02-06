import numpy as np
from sklearn import datasets, cross_validation, preprocessing
from sklearn.linear_model import LinearRegression

dataset = datasets.load_diabetes()

X = dataset.data
Y = dataset.target

X = preprocessing.scale(X)

X_Train, X_Test, Y_Train, Y_Test = cross_validation.train_test_split(X, Y, test_size=0.2)

model = LinearRegression()
model.fit(X_Train, Y_Train)

b = model.predict(X_Test)

print("Coefficients = ", model.coef_)
print("Predictions = ", b)
print("Actual Output = ", Y_Test)
print("Score = ", model.score(X_Test, Y_Test))