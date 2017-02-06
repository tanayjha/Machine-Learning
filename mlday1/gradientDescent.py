import numpy as np
import matplotlib.pyplot as plt

# Setting the learning rate alpha
learning_rate = 0.01

# Setting the initial value of the parameters
a0 = 2
a1 = 2

# Calculates the derivative of the cost function w.r.t a0
def derivativeA0(X, Y):
	val = 0
	for i in range(len(X)):
		val += 2*(a0 + a1*X[i] - Y[i])
	return val

# Calculates the derivative of the cost function w.r.t a1
def derivativeA1(X, Y):
	val = 0
	for i in range(len(X)):
		val += 2*(a0 + a1*X[i] - Y[i])*X[i]
	return val

# Single feature list
X = np.array([1, 2, 3, 4, 5, 6])

# Output list
Y = np.array([1.5, 4, 4, 6, 7, 8])

# Plots the point itself
plt.scatter(X, Y)

# Running gradient descent 199 times
for i in range(0, 200):
	tempa0 = a0 - learning_rate*derivativeA0(X, Y)
	tempa1 = a1 - learning_rate*derivativeA1(X, Y)
	# Performing simultaneous updates 
	a0 = tempa0
	a1 = tempa1
print("Parameters = ", a0, a1)
X1 = 0
pred1 = a0 + a1*X1
 
X2 = 10
pred2 = a0 + a1*X2

X_Line = (X1, X2)
Y_Line = (pred1, pred2)

# Plotting the best fit line
plt.plot(X_Line, Y_Line)
plt.show()

