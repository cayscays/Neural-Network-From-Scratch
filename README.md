# Neural Network Implementation and Applications

## Description
This repository provides a Python implementation of a fully connected neural network, offering flexible customization of network architecture. As the creator of this repository, I've developed two projects that utilize this neural network for classification tasks. Continue reading for further information.. 

## Table of Contents
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Project 1](#project-1)
- [Project 2: Game of Life Oscillator Classifier](#project-2-game-of-life-oscillator-classifier)

## Key Features
- Flexible network architecture customization, allowing easy adjustment of layer sizes and hidden layers.
- Adaptive learning rate.


## Technologies Used
- Python 3.10
- Jupyter
- Numpy 1.22.1
- Pandas
- Matplotlib

## Project 1:

## Project 2: Game of Life Oscillator Classifier
### Overview:
The project aims to classify oscillators within Conway's Game of Life. 

### Dataset Description:
[The dataset](https://github.com/cayscays/oscillators-7x7-dataset-game-of-life/) contains a list of 7x7 oscillators in Conway's Game of Life. The oscillators in the dataset have a maximum period of 15 generations, with the grid borders extending to infinity.

### Data Preprocessing:
Given the inherently clean nature of the dataset, minimal preprocessing was required before training the classification model. I used [utility functions from my dataset repository](https://github.com/cayscays/oscillators-7x7-dataset-game-of-life/tree/main/data_management) to enrich the dataset by incorporating non-oscillators and diversifying entries through grid state flipping. Each entry was then flattened and transformed into a list, aligning with the network's input requirements for further analysis and modelling.

### Model Architecture and Training:
The model architecture encompasses a fully connected neural network trained using backpropagation. It includes the following attributes:
Input layer: Consists of 49 neurons representing the cell states of the input data.
Hidden layers: Comprises two layers, each containing 7 neurons.
Output layer: One neuron. The threshold for classifying oscillators and non-oscillators is set at 0.5. Above it, the pattern is an oscillator.
Learning rate: 0.1.
Activation function: Sigmoid.
Training epochs: 20 epochs.

### Results:
After running the neural network for 20 epochs, an accuracy of 91.20% was achieved. This indicates the model's effectiveness in classifying 7x7 oscillators within Conway's Game of Life. The high accuracy attained demonstrates the capability of even a small neural network to learn and generalize patterns.

---
Thank you for reviewing my neural network and classification projects!

Created by [cayscays](https://github.com/cayscays/).
