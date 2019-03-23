# main.py

import numpy as np
import pandas as pd
import sklearn.linear_model as linmod

trainData = pd.read_csv("dota2Train.csv", header=None)
testData = pd. read_csv("dota2Test.csv", header=None)

# Remove practice, tutorial, and co-op with AI games
trainData.drop(trainData[trainData[3] == 1 ].index , inplace=True)
trainData.drop(trainData[trainData[3] == 3 ].index , inplace=True)
trainData.drop(trainData[trainData[3] == 4 ].index , inplace=True)

testData.drop(testData[testData[3] == 1 ].index , inplace=True)
testData.drop(testData[testData[3] == 3 ].index , inplace=True)
testData.drop(testData[testData[3] == 4 ].index , inplace=True)

train_Y = trainData.iloc[:, 0]
train_X = trainData.iloc[::, 1:114]

test_Y = testData.iloc[:, 0]
test_X = testData.iloc[::, 1:114]

logreg = linmod.LogisticRegression(C=1e5, solver='lbfgs')

logreg.fit(test_X, test_Y)

pred_Y = logreg.predict(test_X)

N = len(pred_Y)
sum = 0.0

for i in range(N):
    if(pred_Y[i] != test_Y[i]):
        sum += 1
        
print(sum / N)
