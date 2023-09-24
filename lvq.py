# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import scipy.stats
import numpy as np
import math

# load data
irisData = load_iris()

# create feature and target arrays
x = irisData.data
y = irisData.target

# Split into training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                    random_state=42)

# Training function


def lvq_train(xTrain, yTrain, maxEpoch, alpha):
    rows = len(xTrain)
    cols = len(xTrain[0])
    epoch = 0
    codeVectors = np.zeros((3, cols))
    distances = np.zeros(3)
    yPredict = 0
    sum = 0
    pointDistances = np.zeros((1, cols))

    # Choose 3 unique code vectors
    for i in range(3):
        indices = np.where(yTrain == i)
        codeVectors[i] = xTrain[indices[0][0]]

    while epoch < maxEpoch:

        # Comput the learning rate for each epoch
        alpha = alpha*(1-(epoch/maxEpoch))
        for i in range(rows):
            for j in range(3):
                for l in range(cols):
                    # Compute the distance
                    pointDistances[0, l] = (
                        xTrain[i, l] - codeVectors[j, l])**2
                sum = np.sum(pointDistances[0])
                distances[j] = math.sqrt(sum)
            minIndex = distances.argmin()

            # Assign the class of the vector with the
            # smallest distance to the input data point
            yPredict = minIndex

            # Update the code vector
            if yPredict == yTrain[i]:
                for l in range(cols):
                    x = codeVectors[minIndex, l]
                    codeVectors[minIndex, l] = x + alpha*(xTrain[i, l] - x)
            else:
                for l in range(cols):
                    x = codeVectors[minIndex, l]
                    codeVectors[minIndex, l] = x - alpha*(xTrain[i, l] - x)
        epoch += 1
    return codeVectors

# Test function


def lvq_test(codeVectors, xTest):
    rows = len(xTest)
    cols = len(xTest[0])
    predLabels = np.zeros(rows)
    pointDistances = np.zeros((1, cols))
    distances = np.zeros(3)

    for i in range(rows):
        for j in range(3):
            for l in range(cols):
                pointDistances[0, l] = (xTest[i, l] - codeVectors[j, l])**2
            sum = np.sum(pointDistances[0])
            distances[j] = math.sqrt(sum)
        predLabels[i] = distances.argmin()
    return predLabels


weight = lvq_train(x_train, y_train, 100, 0.01)
test = lvq_test(weight, x_test)

print(test)
print("Accuracy: ", accuracy_score(test, y_test))
