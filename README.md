# Neural Network from Scratch

## Description
This repository provides an implementation of a fully connected neural network, offering flexible customization of network architecture, allowing easy adjustment of layer sizes and hidden layers. As the creator of this repository, I've developed two projects that utilize this neural network for classification tasks. Continue reading for further information.

## Table of Contents
- [Repository Contents](#repository-contents)
- [Technologies Used](#technologies-used)
- [Project 1: Chronic Kidney Disease Classification](#project-1-chronic-kidney-disease-classification)
  - Overview
  - Dataset Description
  - Data Preprocessing
  - Model Architecture and Training
  - Results
- [Project 2: Oscillators Classification in Game of Life](#project-2-oscillators-classification-in-game-of-life)
  - Overview
  - Dataset Description
  - Data Preprocessing
  - Model Architecture and Training
  - Results


## Repository Contents
1. **neural_network/**: The implementation on the neural network.
2. **project1/** :
3. **oscillators_classification_in_game_of_life/**: This project focuses on classifying oscillators within a cellular automaton using the neural network.


## Technologies Used
- Python 3.10
- Jupyter
- Numpy 1.22.1
- Pandas
- Matplotlib
---
## Project 1: Chronic Kidney Disease Classification
### Overview:
This project focuses on classifying Chronic Kidney Disease (CKD) in individuals. The classification result will be true for CKD cases and false for non-CKD cases.

### Dataset Description:
Chronic Kidney Disease dataset by L. Rubini, P. Soundarapandian, and P. Eswaran, donated on 7/2/2015. Retrieved from https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease. Licensed under CC BY 4.0. 

### Data Preprocessing:

### Model Architecture and Training:
The model architecture is a fully connected neural network trained using backpropagation. It includes the following attributes:
- Input layer: 22 neurons.
- Hidden layers: One layer, containing 5 neurons.
- Output layer: One neuron. 
- Learning rate: 0.5.
- Random seed: 10.
- Epochs: 100.
- Activation function: Sigmoid.

### Results:

- **Test Accuracy**: After training the neural network, a test accuracy of 95.36% was achieved. This high accuracy demonstrates the effectiveness of the model in classifying CKD cases, indicating its reliability in distinguishing between CKD and non-CKD individuals.

- **Sensitivity**: The most important aspect of sensitivity for this classification task is its ability to correctly classify positive cases, ensuring that individuals with the condition are not missed, thereby preventing delayed treatment. False positives, while still important, could undergo further testing to confirm their positive or negative status, while false negative are less likely to undergo further testing.

- **Overfitting**: No overfitting was observed in the training process. Please refer to the accompanying graphs for a visual representation of the training and test performance.


<img src="chronic_kidney_disease_classification/raw_results/accuracy.png"  height="200">

<img src="chronic_kidney_disease_classification/raw_results/error.png"  height="200">


---
## Project 2: Oscillators Classification in Game of Life
### Overview:
The project aims to classify oscillators within Conway's Game of Life. 

### Dataset Description:
[The dataset](https://github.com/cayscays/oscillators-7x7-dataset-game-of-life/) contains a list of 7x7 oscillators in Conway's Game of Life. The oscillators in the dataset have a maximum period of 15 generations, with the grid borders extending to infinity.

### Data Preprocessing:
Given the inherently clean nature of the dataset, minimal preprocessing was required before training the classification model. I used utility functions from my [dataset repository](https://github.com/cayscays/oscillators-7x7-dataset-game-of-life/tree/main/data_management) to enrich the dataset by incorporating non-oscillators and diversifying entries through grid state flipping. Each entry was then flattened and transformed into a list, aligning with the network's input requirements for further analysis and modelling.

### Model Architecture and Training:
The model architecture is a fully connected neural network trained using backpropagation. It includes the following attributes:
- Input layer: 49 neurons representing the cell states of the input data.
- Hidden layers: Two layers, each containing 7 neurons.
- Output layer: One neuron. The threshold for classifying oscillators and non-oscillators is set at 0.5. Above it, the pattern is an oscillator.
- Learning rate: 0.5.
- Random seed: 10.
- Epochs: 20.
- Activation function: Sigmoid.

### Results:
After running the neural network for 20 epochs, a test accuracy of 97.25806451612902% was achieved. This indicates the model's effectiveness in classifying 7x7 oscillators within Conway's Game of Life. The high accuracy attained demonstrates the capability of even a small neural network to learn and generalize patterns.

<img src="oscillators_classification_in_game_of_life/raw_results/accuracy%20and%20error.png"  height="250">


---
Thank you for reviewing my neural network and classification projects!

Created by [cayscays](https://github.com/cayscays/).
