import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import random
get_ipython().run_line_magic('matplotlib', 'inline')
from numpy import genfromtxt

class NeuralNetwork(object):
    def __init__(self):
        self.inputs = 4
        self.outputs = 1
        self.hidden = 2
        self.W1 = np.random.randn(self.inputs, self.hidden)
        self.W2 = np.random.randn(self.hidden, self.outputs)

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def sigmoidDerivative(self, z):
        return np.exp(-z) / ((1 + np.exp(-z))**2)

    def feedForward(self, X):
        self.z2 = X @ self.W1
        self.a2 = self.sigmoid(self.z2)
        self.z3 = self.a2 @ self.W2
        self.yhat = self.sigmoid(self.z3)
        return self.yhat

    def functionError(self, X, y):
        self.yhat = self.feedForward(X)
        cost = 0.5 * sum((y - self.yhat)**2)
        return cost

    def functionErrorDeriv(self, X, y):
        self.yhat = self.feedForward(X)
        self.delta3 = np.multiply(-(y - self.yhat), self.sigmoidDerivative(self.z3))
        self.djdW2 = np.dot(self.a2.T, self.delta3)
        self.delta2 = self.delta3 @ self.W2.T * self.sigmoidDerivative(self.z2)
        self.djdW1 = X.T @ self.delta2
        return self.djdW1, self.djdW2

    def getWeights(self):
        data = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return data

    def setWeights(self, data):
        W1_i = 0
        W1_f = self.hidden * self.inputs
        self.W1 = np.reshape(data[W1_i:W1_f], (self.inputs , self.hidden))
        W2_f = W1_f + self.hidden*self.outputs
        self.W2 = np.reshape(data[W1_f:W2_f], (self.hidden, self.outputs))

    def getGradients(self, X, y):
        djdW1, djdW2 = self.functionErrorDeriv(X, y)
        return np.concatenate((djdW1.ravel(), djdW2.ravel()))

from scipy import optimize
class Trainer:
    def __init__(self, aNet):
        self.NN = aNet

    def updateWeights(self, params):
        self.NN.setWeights(params)
        self.Costs.append(self.NN.functionError(self.X, self.y))
        self.CostsTest.append(self.NN.functionError(self.Xtest, self.ytest))

    def getWeightsNN(self, params, X, y):
        self.NN.setWeights(params)
        cost = self.NN.functionError(X, y)
        grad = self.NN.getGradients(X,y)
        return cost, grad

    def train(self, X, y, Xtest, ytest):
        self.X = X
        self.y = y
        self.Xtest = Xtest
        self.ytest = ytest
        self.Costs = []
        self.CostsTest = []
        weights = self.NN.getWeights()

        ops = {'maxiter': 2000, 'disp' : True}

        output = optimize.minimize(self.getWeightsNN, weights, jac=True, method='BFGS',                                  args=(X, y), options=ops, callback=self.updateWeights)

        self.NN.setWeights(output.x)
        self.results = output

data = pd.read_csv("NCAAWGymRankings.csv")
data

nn = NeuralNetwork()
results = np.array(data)
XGym = np.genfromtxt("NCAAWGymRankings.csv", delimiter = ',')
DataX = XGym[1:19666, 1:6]
random.shuffle(DataX)
print(DataX)

# Train
TrainXGym = np.array((DataX[:15732, :4]), dtype=float)
TrainYGym = np.array((DataX[:15732, 4:]), dtype=float)
# Test
TestXGym = np.array((DataX[15732:, :4]), dtype = float)
TestYGym = np.array((DataX[15732:, 4:]), dtype = float)
# Normalization Train
TrainXGym = TrainXGym/np.amax(TrainXGym, axis = 0)
TrainYGym = TrainYGym/np.amax(TrainYGym, axis = 0)
# Normalization Test
TestXGym = TestXGym/np.amax(TestXGym, axis = 0)
TestYGym = TestYGym/np.amax(TestYGym, axis = 0)

t = Trainer(nn)
t.train(TrainXGym, TrainYGym, TestXGym, TestYGym)

plt.plot(t.Costs)
plt.plot(t.CostsTest)
plt.grid(1)
plt.ylabel("Total Eror")
plt.xlabel("Iterations")
