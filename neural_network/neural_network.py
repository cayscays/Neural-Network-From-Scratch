"""
Author:       cayscays
Date:         December 2021
Version:      1
Description:  A neural network implemented from scratch
"""

import random
import numpy as np


class NeuralNetwork:
    """
    A fully connected neural network
    """

    def __init__(self, input_size, hidden_layers_sizes, output_size,
                 learning_rate, amount_of_epochs, batch_size, data, seed):
        """
            Initializes the neural network.

            Args:
                input_size (list): The number of neurons in the input layer.
                hidden_layers_sizes (list): The number of neurons in the hidden layers.
                output_size (list): The number of neurons in the output layer.
                learning_rate (float): The learning rate for the network.
                amount_of_epochs (int): The number of epochs to train the network for.
                batch_size (int): The size of the batch (currently supports batch size of 1).
                data (list): The data for training and testing.
                seed (int): The seed for random initialization.
        """
        random.seed(seed)
        self.all_layers_sizes = input_size + hidden_layers_sizes + output_size
        self.learning_rate = learning_rate
        self.amount_of_epochs = amount_of_epochs
        self.batch_size = batch_size  # For future batch features. currently supports batch size of 1.

        self.errors = {'training': [], 'test': []}
        self.accuracy = {'training': [], 'test': []}
        self.epochs = []

        # Initiates random weights
        self.weights = []
        for i in range(len(self.all_layers_sizes) - 1):
            self.weights.append(np.random.rand(self.all_layers_sizes[i + 1], self.all_layers_sizes[i]))

        self.values = []
        for layer in self.all_layers_sizes:
            self.values.append(np.zeros(layer))

        self.delta = []
        for i in range(1, len(self.values)):
            self.delta.append(np.zeros(self.values[i].shape))

        # divide the data to test and training:
        random.shuffle(data)
        n = int(len(data) / 2)
        self.training_data = data[n:]
        self.test_data = data[:n]

    def sigmoid(self, vals):
        """
        Calculates the sigmoid function for the given values.

        Args:
            vals (numpy.array): Input values.

        Returns:
            numpy.array: Result of applying the sigmoid function to the input values.
        """
        return 1 / (np.exp(-vals) + 1.0)

    def get_label_id(self, output):
        """
        Determines the label id based on the output value.

        Args:
            output (numpy.array): Output value.

        Returns:
            int: Label ID.
        """
        if output[0] > 0.5:
            return 1
        else:
            return 0

    def forward_pass_single_input(self, single_input):
        """
        Performs a forward pass for a single input through the network.

        This method updates all neuron's values.
        Immediately after the input layer there is no activation function.


        Args:
            single_input (list): Input data.

        Returns:
            int: Predicted label id.
        """
        self.values[0] = np.array(single_input)
        # forward pass
        for i in range(1, len(self.all_layers_sizes)):
            self.values[i] = self.sigmoid(self.weights[i - 1] @ self.values[i - 1])
        return self.get_label_id(self.values[-1])

    def backpropagation(self, correct_label):
        """
        Updates the weights of the network using the backpropagation algorithm.

        Args:
            correct_label (numpy.array): The correct label for the current input.
        """
        output = self.values[len(self.values) - 1]
        self.delta[len(self.delta) - 1] = (correct_label - output) * output * (1 - output)

        for l in range(len(self.delta) - 2, -1, -1):
            for i in range(len(self.delta[l])):
                temp = 0
                for j in range(len(self.delta[l + 1])):
                    temp += (self.weights[l + 1][j][i] * self.delta[l + 1][j])
                self.delta[l][i] = self.values[l + 1][i] * (1 - self.values[l + 1][i]) * temp

        # update the weights
        for j in range(len(self.weights)):
            for i in range(len(self.weights[j])):
                # one line at a time:
                self.weights[j][i] += self.learning_rate * self.values[j] * self.delta[j][i]

    def calculate_single_run_error(self, target, output):
        """
        Calculates the error for a single run.

        Args:
            target (numpy.array): Target values.
            output (numpy.array): Output values.

        Returns:
            float: Error value.
        """
        error = 0
        for i in range(len(output)):
            error += (target[i] - output[i]) ** 2
        return error

    def run_epoch(self):
        """
        Runs an epoch of training on all the data.
        Updates error, accuracy, and weights.
        """
        training_error = 0
        test_error = 0
        training_accuracy = 0
        test_accuracy = 0
        for i in range(len(self.training_data)):
            # 1 for correct, 0 for incorrect
            training_accuracy += 1 + self.forward_pass_single_input(self.training_data[i][0]) - \
                                 self.training_data[i][1][0]

            self.backpropagation(self.training_data[i][1])
            training_error += self.calculate_single_run_error(self.training_data[i][1], self.values[-1])

            # 1 for correct, 0 for incorrect
            test_accuracy += 1 + self.forward_pass_single_input(self.test_data[i][0]) - self.test_data[i][1][0]

            test_error += self.calculate_single_run_error(self.test_data[i][1], self.values[-1])

        test_error /= len(self.test_data)
        training_error /= len(self.training_data)
        self.errors['training'].append(training_error)
        self.errors['test'].append(test_error)

        test_accuracy /= len(self.test_data)
        training_accuracy /= len(self.training_data)
        test_accuracy *= 100
        training_accuracy *= 100
        self.accuracy['training'].append(training_accuracy)
        self.accuracy['test'].append(test_accuracy)

    def train(self):
        """
        Trains the neural network.
        """
        for i in range(self.amount_of_epochs):
            self.run_epoch()
