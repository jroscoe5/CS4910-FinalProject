#Description: This program provides the functionality for testing
#various machine learning algorithms in predicting the outcome
#of a dota 2 match based on game mode, type, and team compositions.

import numpy as np
from numpy import genfromtxt
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import data_clean
import csv

#Clean train/test data
#Data is rows of [result, game mode, game type, ...-> team comp]
#This takes part takes a while.
if __name__ == '__main__':
  data_clean.clean('data\dota2Test.csv','CleanTest.csv')
  data_clean.clean('data\dota2Train.csv','CleanTrain.csv')

testRaw = genfromtxt('CleanTest.csv', delimiter=',')
trainRaw = genfromtxt('CleanTrain.csv', delimiter=',')
testSize = len(testRaw)
trainSize = len(trainRaw)
fSize = 115 #Num of features 

#Store tags
testTags = np.zeros(testSize)
trainTags = np.zeros(trainSize)

#Store features
testF = np.zeros([testSize, fSize])
trainF = np.zeros([trainSize, fSize])

#Pull out test tags and features
for i in range(0, testSize - 1):
  charCode = 0 #The character ID
  charCount = 0 #How many characters out of 10 stored
  testTags[i] = testRaw[i][0] #Store game result
  testF[i] = testRaw[i][1:] #Store mode

#Pull out train tags and features
for i in range(0, trainSize - 1):
  charCode = 0 #The character ID
  charCount = 0 #How many characters out of 10 stored
  trainTags[i] = trainRaw[i][0] #Store game result
  trainF[i] = trainRaw[i][1:] #Store mode

#% missed error of classifier 
def error(pred, real):
  e = 0;
  for i in range(0, len(pred)-1):
    if pred[i] != real[i]:
      e = e + 1
  return e/len(pred)
  
#MULTI-LAYER PERCEPTRON
#Layers is a tuple as defined for sklearn MLP classifier
def testMLPModel(lays, activation, sol, trainFeatures, trainTags, testFeatures, testTags):
  M = MLPClassifier(activation=activation, solver=sol, hidden_layer_sizes=lays);
  M.fit(trainFeatures, trainTags);
  return error(M.predict(testFeatures), testTags)

#RANDOM FOREST 
def testRFModel(trees, maxDepth, trainFeatures, trainTags, testFeatures, testTags):   
  M = RandomForestClassifier(n_estimators=trees, max_depth=maxDepth)
  M.fit(trainFeatures, trainTags);
  return error(M.predict(testFeatures), testTags)

#KNN 
def testKNNModel(n, trainFeatures, trainTags, testFeatures, testTags):
  M = KNeighborsClassifier(n_neighbors=n)
  M.fit(trainFeatures, trainTags);
  return error(M.predict(testFeatures), testTags)
  
#LOGISTIC REGRESSION
def testLRModel(sol, trainFeatures, trainTags, testFeatures, testTags):
  M = LogisticRegression(solver=sol, multi_class='auto')
  M.fit(trainFeatures, trainTags);
  return error(M.predict(testFeatures), testTags)
  
#TESTING MLP  
#All these variations of MLP where around 40% with little variance
#print(testMLPModel((20), 'identity', 'lbfgs', trainF, trainTags, testF, testTags))
#print(testMLPModel((40), 'identity', 'lbfgs', trainF, trainTags, testF, testTags))
#print(testMLPModel((100), 'identity', 'lbfgs', trainF, trainTags, testF, testTags))
#print(testMLPModel((20 , 20), 'identity', 'lbfgs', trainF, trainTags, testF, testTags))
#print(testMLPModel((20), 'identity', 'lbfgs', trainF, trainTags, testF, testTags))
#print(testMLPModel((20), 'logistic', 'lbfgs', trainF, trainTags, testF, testTags))
#print(testMLPModel((20), 'tanh', 'lbfgs', trainF, trainTags, testF, testTags))
#print(testMLPModel((20), 'relu', 'lbfgs', trainF, trainTags, testF, testTags))

#print(testMLPModel((20), 'identity', 'sgd', trainF, trainTags, testF, testTags))
#print(testMLPModel((20), 'identity', 'adam', trainF, trainTags, testF, testTags))

#TESTING RANDOM FOREST
#Trees with depth 2 have about 45% error
#print(testRFModel(100, 2, trainF, trainTags, testF, testTags))
#print(testRFModel(50, 2, trainF, trainTags, testF, testTags))
#print(testRFModel(20, 2, trainF, trainTags, testF, testTags))

#Trees with depth 20 preform the best at 42%
#print(testRFModel(20, 5, trainF, trainTags, testF, testTags))
#print(testRFModel(20, 10, trainF, trainTags, testF, testTags))
#print(testRFModel(20, 20, trainF, trainTags, testF, testTags))
#print(testRFModel(20, 50, trainF, trainTags, testF, testTags))
#print(testRFModel(20, 100, trainF, trainTags, testF, testTags))

#TESTING KNN
#All n values had around 47% error
#print(testKNNModel(1, trainF, trainTags, testF, testTags))
#print(testKNNModel(2, trainF, trainTags, testF, testTags))
#print(testKNNModel(5, trainF, trainTags, testF, testTags))

#TESTING LOGISTIC REGRESSION
#All had 40% error with almost no variance
#print(testLRModel('newton-cg', trainF, trainTags, testF, testTags))
#print(testLRModel('lbfgs', trainF, trainTags, testF, testTags))
#print(testLRModel('liblinear', trainF, trainTags, testF, testTags))
#print(testLRModel('sag', trainF, trainTags, testF, testTags))
#print(testLRModel('saga', trainF, trainTags, testF, testTags))

#One hidden layer 228 relu
#print(testMLPModel((228), 'relu', 'lbfgs', trainF, trainTags, testF, testTags))
#Two hidden layer both 228 relu
#print(testMLPModel((228 ,228), 'relu', 'lbfgs', trainF, trainTags, testF, testTags))
#One hidden layer 114 relu
#print(testMLPModel((114), 'relu', 'lbfgs', trainF, trainTags, testF, testTags))
#Two hidden layer both 114 relu
#print(testMLPModel((114 ,114), 'relu', 'lbfgs', trainF, trainTags, testF, testTags))
#One hidden layer 57 relu
#print(testMLPModel((57), 'relu', 'lbfgs', trainF, trainTags, testF, testTags))
#Two hidden layer both 57 relu
#print(testMLPModel((57 ,57), 'relu', 'lbfgs', trainF, trainTags, testF, testTags))
