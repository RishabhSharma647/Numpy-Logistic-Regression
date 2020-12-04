#Regularised Logistic Regression for Perov/Non-Perov Classification

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

dataset = pd.read_csv('PerovNonPerovFINALANN.csv')
y = dataset.iloc[0:622,0].values


for i in range(1,10):
    vars()['x' + str(i)] = dataset.iloc[0:622, i].values


X = [[1, x1[i], x2[i], x3[i], x4[i], x5[i], x6[i], x7[i], x8[i], x9[i]] for i in range(622)]

theta = np.random.normal(0, 0.5, len(X[1]))
m = len(x1)
lmbda = 0.01
lr = 0.1
num_epochs = 200

# Train and test split
test_idx = random.sample(range(0, len(y)), 70)
y_train = [y[i] for i in range(len(y)) if i not in test_idx]
X_train = [X[i] for i in range(len(X)) if i not in test_idx]

y_test = [y[i] for i in range(len(y)) if i in test_idx]
X_test = [X[i] for i in range(len(X)) if i in test_idx]


def sigmoid(theta, x):
    return 1/(1 + np.exp(-np.dot(theta,x)))

def single_cost(htheta, y):
    return -y*np.log(htheta)-(1-y)*np.log(1-htheta)

def total_cost(theta):
    htheta_all = [sigmoid(theta,X_train[i]) for i in range(len(X_train))]
    return (1/m)*sum([single_cost(htheta_all[i], y_train[i]) for i in range(len(htheta_all))])

def percent_accuracy_training(theta):
    htheta_all = [round(pred) for pred in [sigmoid(theta,X_train[i]) for i in range(len(X_train))]]
    return len([i for i in range(len(htheta_all)) if htheta_all[i] == y_train[i]])/len(y_train)

def percent_accuracy_testing(theta):
    htheta_all = [round(pred) for pred in [sigmoid(theta,X_test[i]) for i in range(len(X_test))]]
    return len([i for i in range(len(htheta_all)) if htheta_all[i] == y_test[i]])/len(y_test)
    
all_epochs, trainset_costs, training_accuracies, testing_accuracies = [[] for _ in range(4)]


for epoch in range(num_epochs):
    dtheta = list(np.zeros(len(theta)))
    
    for j in range(len(dtheta)):
        dtheta[j] = -(lr/m)*sum([(sigmoid(theta, X_train[i])-y_train[i])*X_train[i][j] for i 
              in range(len(y_train))])-(lmbda/m)*theta[j]
    
    for k in range(len(theta)):
        theta[k] += dtheta[k]
    
    all_epochs.append(epoch)
    print("Epoch: {}".format(epoch))
    trainset_costs.append(total_cost(theta))
    print("Total Cost : {}".format(total_cost(theta)))
    training_accuracies.append(percent_accuracy_training(theta))
    print("Percent Training Accuracy: {}".format(percent_accuracy_training(theta)))
    testing_accuracies.append(percent_accuracy_testing(theta))
    print("Percent Testing Accuracy: {}".format(percent_accuracy_testing(theta)))
    
    
# Plot results
plt.plot(all_epochs, trainset_costs)
plt.xlabel("Epoch")
plt.ylabel("Training Cost")
plt.title("Training Cost vs. Epoch")

plt.plot(all_epochs, training_accuracies)
plt.plot(all_epochs, testing_accuracies)
plt.legend(["Training set accuacy", "Testing set accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Train and Test Set Accuracy (%) vs. Epoch")



    
    
    
