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

def scale(x):
    scaler = MinMaxScaler()
    scaled_x = scaler.fit_transform(x)
    return scaled_x

scaled_data = scale(data)
scaled_target = scaled_data[:,-1]
scaled_data = np.delete(scaled_data, -1, 1)
# Split into training and test set
x_train, x_test, y_train, y_test = train_test_split(scaled_data, scaled_target, test_size = 0.7, 
                                                         random_state = 42)
def MSE (coeff, x, y, m):
    total_sq_err = 0
    sum_err = 0
    yhat = np.zeros((m,1))
    errArray = np.zeros((1,2))
    
    # Add the first coeff to yhat
    for i in range(m):
        yhat[i,0] = coeff[0,0]
         
    #Find yhat give a set of coefficients and input x
    for i in range(m):
        for j in range(len(x[0])):
            yhat[i,0] += coeff[0,j+1]*float(x[i,j])
            
        
    # Calculate total error between yhat and y
    for i in range(m):
        total_sq_err += (yhat[i]-y[i])**2
        sum_err += (yhat[i]-y[i])
    
    # Find the mean squared error
    meanSqError = (1/m)*total_sq_err
    
    errArray = np.concatenate((meanSqError, sum_err), axis=0)
    return errArray
        
def gradDescent_train(xTrain, yTrain, alpha, maxEpoch):
    rows = len(xTrain)
    cols = len(xTrain[0])
    coeff = np.zeros((1,cols+1))
    yPred = np.zeros((rows,1))
    errArray = MSE(coeff, xTrain, yTrain, rows)
    loss = errArray[0]
    sumError = errArray[1]
    epoch = 0
    
    while loss > 0.01 and epoch < maxEpoch:
        for i in range(rows):
            for j in range(cols+1):
                # The first coefficient is a special case
                # because it's an intercept
                if j == 0:
                    coeff[0,j] = coeff[0,j] - alpha*(1/rows)*sumError 
                else:
                    coeff[0,j] = coeff[0,j] - alpha*(1/rows)*sumError*(xTrain[i,j-1]**rows)
            
            # After the first training example, we will have a new set of coefficient
            errArray = MSE(coeff, xTrain, yTrain, rows)
            loss = errArray[0]
            sumError = errArray[1]
        epoch += 1
            
                
    return coeff

coefficients = gradDescent_train(x_train, y_train, 0.01, 10)

def gradDescent_test(xTest,coeff):
    rows = len(xTest)
    cols = len(xTest[0])
    predictions = np.zeros((rows,1))
    
    for i in range(rows):
        for j in range(cols):
            predictions[i,0] += coeff[0,j+1]*xTest[i,j]
            predictions[i,0] += coeff[0,0]
            
    return predictions

yOutput = gradDescent_test(x_test, coefficients)

#print(yOutput)

errArray = MSE(coefficients, x_test, y_test, len(x_test))  
print("Mean squared error: ", errArray[0])     
    
    
    
    
    
        
        