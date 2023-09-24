#Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import scipy.stats
import numpy as np
import math 



# Loading data
irisData = load_iris()

# Create feature and target arrays
x = irisData.data
y = irisData.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
             x, y, test_size = 0.7, random_state=42)




def knn (xTest, xTrain, yTrain, k):
  rows = len(xTest)
  cols = len(xTest[0])
  tRow = len(xTrain)
  yPredict = np.zeros(rows)
  distances = np.zeros(len(xTrain))
  sum = 0
  pointDistances = np.zeros((1,cols))
  
  for i in range(rows):
    for j in range(tRow):
      for l in range(cols):
        pointDistances[0,l] = (xTest[i,l]-xTrain[j,l])**2
      sum = np.sum(pointDistances[0])
      distances[j] = math.sqrt(sum)
    yPredict[i] = yTrain[distances.argmin()]
  return yPredict

yOutput = knn(X_test, X_train, y_train, 1)

print(yOutput)
print("Accuracy: ", accuracy_score(yOutput,y_test))