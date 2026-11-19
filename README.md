# DSLR (Data Science Logistic Regression)

  Implementation of a Multi-Class Logistic Regression model from scratch in Python for Hogwarts House prediction, including multiple gradient descent optimizers and convergence visualization as bonus features.

# Description

  A comprehensive data science project from the 42 curriculum. It introduces Exploratory Data Analysis (EDA) and implements a One-vs-All Logistic Regression classifier from scratch in Python—without external ML training libraries like Scikit-Learn.
  The project explores data visualization, standardization, cost functions, gradient descent optimization, classification metrics, and bonus optimizer comparisons.

---

## Table of Contents
1. About The Project
2. Features and Architecture
3. Prerequisites & Installation
4. Usage and Execution
  - Exploratory Data Analysis & Visualization
  - Training the Model
  - Predicting Hogwarts Houses
  - Evaluating Performance
  - Bonus: Optimizers & Loss Convergence Comparison
5. In-Depth Mathematical Concepts

---

## About The Project
The goal of this project is to dive into Data Science by analyzing student records from Hogwarts, handling missing data, visualizing features, and building a machine learning classifier from scratch.

Since logistic regression is inherently binary, the program uses the One-vs-All (OvR) strategy to classify students into one of four houses: Gryffindor, Hufflepuff, Ravenclaw, or Slytherin.

The core prediction relies on the Sigmoid function applied to a linear combination of features:
z = X * theta
h_theta(x) = 1 / (1 + exp(-z))

As a bonus, the classifier is encapsulated in a reusable `Logreg` class supporting **Batch**, **Mini-Batch** (`minibatch` / `mbsgd`), and **Stochastic Gradient Descent** (`sgd`), along with **Binary Cross-Entropy** loss tracking across epochs.

---

## Features & Architecture

- describe.py: A custom implementation of pandas' describe() function. Computes count, mean, std, min, percentiles (25%, 50%, 75%), and max for all numerical features.
- histogram.py: Plots histograms for each Hogwarts course to discover which course is homogeneous across all houses (e.g., Care of Magical Creatures).
- scatter_plot.py: Generates a scatter plot to identify two features that are similar / correlated (e.g., Astronomy and Defense Against the Dark Arts).
- pair_plot.py: Visualizes pairwise relationships across all features using seaborn or matplotlib and save it in 'Pair_plot.png'.
- logreg.py (Bonus): Reusable `Logreg` class implementing sigmoid activation, weight updates for `batch`, `minibatch` (`mbsgd`), and `sgd` optimizers, clipped Binary Cross-Entropy (`_compute_loss`), `loss_history` tracking, `fit`, and `predict_proba`.
- logreg_train.py: Implements Gradient Descent with Z-score normalization to iteratively train binary classifiers for each house. Saves the optimal weights (and normalization parameters) into a JSON file.
- logreg_predict.py: Loads the trained weights, normalizes the test dataset, computes probabilities for each house, applies the argmax rule, and outputs a houses.csv prediction file.
- Bonus training & comparison script: Runs One-vs-All training across `batch`, `minibatch`, and `sgd`, exports prediction CSVs, reports accuracy, and plots a 2x2 grid comparing `loss_history` per house.

---

## Prerequisites & Installation

Make sure you have Python 3 and pip installed on your machine. 

### Installing Dependencies:
You will need numpy, pandas, and matplotlib (and seaborn if used in pair_plot.py) for data manipulation and visualization:
pip install numpy pandas matplotlib seaborn

---

## Usage and Execution Guide

1. Exploratory Data Analysis (describe.py):
Displays summary statistics for the training dataset.
python show_data/describe.py datasets/dataset_train.csv

2. Data Visualization:
Explore feature distributions and correlations.
python show_data/histogram.py 
python show_data/scatter_plot.py
python show_data/pair_plot.py 

3. Training the Model (logreg_train.py):
Trains the logistic regression models using gradient descent and saves weights to weights.json.
python logistic_regression/logreg_train.py datasets/dataset_train.csv

4. Predicting Houses (logreg_predict.py):
Generates house predictions for the test dataset and creates houses.csv.
python logistic_regression/logreg_predict.py datasets/dataset_test.csv logistic_regression/weights.json

5. Bonus — Comparing Optimizers (`batch`, `minibatch`, `sgd`):
Trains One-vs-All models with different optimizers and learning rates/epochs, writes CSV outputs
(e.g. `logistic_regression/batch.csv`, `logistic_regression/minibatch.csv`, `logistic_regression/sgd.csv`), 
prints accuracy, and displays a 2x2 subplot comparing Binary Cross-Entropy loss curves for Gryffindor, Slytherin, Ravenclaw, and Hufflepuff.
python logistic_regression/logreg_bonus.py

---

## In-Depth Mathematical Concepts

1. Z-Score Normalization:
To ensure gradient descent converges smoothly and features are on the same scale, data is standardized:
x_norm = (x - mu) / sigma

2. Sigmoid Activation Function:
Maps linear scores (z) into probabilities between 0 and 1:
h_theta(x) = 1 / (1 + exp(-z))

3. Binary Cross-Entropy Loss & Gradient Descent Optimizers:
Measures average prediction penalty across samples (with probability clipping to avoid `log(0)`):
J(theta) = -(1 / m) * sum(y^(i) * log(p^(i)) + (1 - y^(i)) * log(1 - p^(i)))

Iteratively updates weight vector theta using learning rate `alpha`:
- **Batch**: computes the gradient over all `m` samples once per epoch.
- **Mini-Batch (`minibatch` / `mbsgd`)**: shuffles data each epoch and updates weights on slices of size `batch_size`.
- **Stochastic (`sgd`)**: shuffles data each epoch and updates weights sample by sample (`1` row at a time).

4. One-vs-All (OvR) & Argmax Decision Rule:
For K classes, train K separate binary classifiers. For a given student, compute the probability for each house and select the class with the highest probability using np.argmax(axis=1):
Predicted House = argmax_k P(House_k | x)