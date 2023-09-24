import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import math
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
target = target.reshape(target.shape[0], 1)
data = np.append(data, target, axis=1)

# Function to scale the data


def scale(x):
    scaler = MinMaxScaler()
    scaled_x = scaler.fit_transform(x)
    return scaled_x


scaled_data = scale(data)
scaled_target = scaled_data[:, -1]
scaled_data = np.delete(scaled_data, -1, 1)

# Add a column of one's
onesArr = np.ones([len(scaled_data), 1])
scaled_data = np.append(onesArr, scaled_data, axis=1)

# Split into training and test set
x_train, x_test, y_train, y_test = train_test_split(scaled_data, scaled_target,
                                                    test_size=0.3,
                                                    random_state=42)

# Function to compute the mean squared error


def MSE(coeff, x, y, m):
    total_sq_err = 0
    sum_err = 0
    yhat = np.zeros((m, 1))
    errArray = np.zeros((1, 2))

    # Find yhat give a set of coefficients and input x
    for i in range(m):
        for j in range(len(x[0])):
            yhat[i, 0] += coeff[j]*x[i, j]

    # Calculate total error between yhat and y
    for i in range(m):
        total_sq_err += (yhat[i]-y[i])**2

    # Find the mean squared error
    meanSqError = (1/m)*total_sq_err

    return meanSqError

# Function to multiply matrices


def matrix_mul(m1, m2):
    result = np.matmul(m1, m2)
    return result+


# Implement the normal equation


def normal_equation_train(xTrain, yTrain):
    xT = np.transpose(xTrain)
    transMul = matrix_mul(xT, xTrain)

    invProd = np.linalg.inv(transMul)

    xTy = matrix_mul(xT, yTrain)

    coeff = matrix_mul(invProd, xTy)

    return coeff

# Test function


def linearReg_test(xTest, coeff):
    rows = len(xTest)
    cols = len(xTest[0])
    predictions = np.zeros((rows, 1))
    for i in range(rows):
        for j in range(cols):
            predictions[i, 0] += coeff[j]*xTest[i, j]
    return predictions


# Find the coefficients using the training set
coefficients = normal_equation_train(x_train, y_train)

# Test the coefficients
yOutput = linearReg_test(x_test, coefficients)

# Find the mean squared error
meanSqErr = MSE(coefficients, x_test, y_test, len(x_test))
print("Mean squared error: ", meanSqErr)
