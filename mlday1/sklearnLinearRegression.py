import numpy as np
from sklearn import datasets, cross_validation, preprocessing	
from sklearn.linear_model import LinearRegression

# Loading the boston dataset
dataset = datasets.load_boston()
X = dataset.data # data holds the features	 
Y = dataset.target # target holds the output

print(X[0:2])
# Scaling the data
# Gaussian distribution with zero mean and unit variance
X = preprocessing.scale(X)	
print(X[0:2])

#Separate training and testing
X_Train, X_Test, Y_Train, Y_Test = cross_validation.train_test_split(X, Y, test_size=0.02)

model = LinearRegression()
# self : returns an instance of self.
model.fit(X_Train, Y_Train)

# Returns predicted values.
b = model.predict(X_Test)

# Calculate the score
# The coefficient R^2 is defined as (1 - u/v), where u is the regression sum of squares ((y_true - y_pred) ** 2).sum() 
# and v is the residual sum of squares ((y_true - y_true.mean()) ** 2).sum(). 
# Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). 
# A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.
score = model.score(X_Test, Y_Test)
print("Score = ", score)

# Printing the value of the features
print("Coefficients = ", model.coef_)
# The coeffecients vary 

print("Predictions = ", b)
print("Actual Output = ", Y_Test)