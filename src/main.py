# main.py

import numpy as np
import pandas as pd
import sklearn.linear_model as linmod
import sklearn.neural_network as NN

def error(exact, pred):
    N = len(exact)
    sum = 0.0
    misclass = 0

    for i in range(N):
        if(exact[i] != pred[i]):
            sum += 1.0
            misclass += 1

    print(sum / N)
    print(misclass)

trainData = pd.read_csv("dota2Train.csv", header=None)
testData = pd. read_csv("dota2Test.csv", header=None)

# Remove practice, tutorial, and co-op with AI games
trainData.drop(trainData[trainData[3] == 1 ].index , inplace=True)
trainData.drop(trainData[trainData[3] == 3 ].index , inplace=True)
trainData.drop(trainData[trainData[3] == 4 ].index , inplace=True)

testData.drop(testData[testData[3] == 1 ].index , inplace=True)
testData.drop(testData[testData[3] == 3 ].index , inplace=True)
testData.drop(testData[testData[3] == 4 ].index , inplace=True)

train_Y = np.array(trainData.iloc[:, 0])
train_X = np.array(trainData.iloc[::, 4::])

test_Y = np.array(testData.iloc[:, 0])
test_X = np.array(testData.iloc[::, 4::])

logreg = linmod.LogisticRegression()
logreg.fit(test_X, test_Y)

pred = logreg.predict(test_X)
error(pred, test_Y)

mlp = NN.MLPClassifier(hidden_layer_sizes=(228,))
mlp.fit(train_X, train_Y)

pred = mlp.predict(test_X)
error(pred, test_Y)

mlp = NN.MLPClassifier(hidden_layer_sizes=(228, 228))
mlp.fit(train_X, train_Y)

pred = mlp.predict(test_X)
error(pred, test_Y)

mlp = NN.MLPClassifier(hidden_layer_sizes=(114,))
mlp.fit(train_X, train_Y)

pred = mlp.predict(test_X)
error(pred, test_Y)

mlp = NN.MLPClassifier(hidden_layer_sizes=(114,114))
mlp.fit(train_X, train_Y)

pred = mlp.predict(test_X)
error(pred, test_Y)

mlp = NN.MLPClassifier(hidden_layer_sizes=(57,))
mlp.fit(train_X, train_Y)

pred = mlp.predict(test_X)
error(pred, test_Y)

mlp = NN.MLPClassifier(hidden_layer_sizes=(57,57))
mlp.fit(train_X, train_Y)

pred = mlp.predict(test_X)
error(pred, test_Y)

mlp = NN.MLPClassifier(hidden_layer_sizes=(29,))
mlp.fit(train_X, train_Y)

pred = mlp.predict(test_X)
error(pred, test_Y)

mlp = NN.MLPClassifier(hidden_layer_sizes=(29,29))
mlp.fit(train_X, train_Y)

pred = mlp.predict(test_X)
error(pred, test_Y)
