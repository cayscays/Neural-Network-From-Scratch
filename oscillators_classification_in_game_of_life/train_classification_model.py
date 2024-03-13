"""
Author:       cayscays
Date:         December 2021
Version:      1
Description:  Trains a neural network to classify oscillators within Conway's Game of Life. The network architecture comprises:
              - An input layer consisting of 49 neurons.
              - Two hidden layers, each with 7 neurons.
              - An output layer consisting of a single neuron.
"""

import matplotlib.pyplot as plt
import pandas as pd

import dataset
from neural_network.neural_network import NeuralNetwork

SEED = 10

# Network's architecture:
INPUT_SIZE = [49]
HIDDEN_LAYERS_SIZES = [7, 7]
LABELS = [1]

# Optimization parameters:
LEARNING_RATE = 0.5
amount_of_epochs = 20
batch_size = 1

# Initiate and train the neural network
nn = NeuralNetwork(INPUT_SIZE, HIDDEN_LAYERS_SIZES, LABELS, LEARNING_RATE, amount_of_epochs, batch_size,
                   dataset.data, SEED)
nn.train()

# Initiate epochs for the x axes of the graphs
epochs = []
for i in range(amount_of_epochs):
    epochs.append(i)

# Plot error and accuracy graphs:
fig, axs = plt.subplots(1, 2)

error_graph = pd.DataFrame(nn.errors, epochs)
error_graph.plot(title="Error", kind='line', xlabel='Number of epochs', ax=axs[0], ylabel="Error")

accuracy_graph = pd.DataFrame(nn.accuracy, epochs)
accuracy_graph.plot(title="Accuracy", kind='line', xlabel='Number of epochs', ax=axs[1], ylabel="Accuracy")
plt.tight_layout()
plt.show()

print("The test accuracy is " + str(nn.accuracy['test'][-1]) + "%")
