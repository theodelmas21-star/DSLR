# DSLR (Data Science Logistic Regression)

    Implementation of a Multi-Class Logistic Regression model from scratch in Python for Hogwarts House prediction.

# Description

    > A comprehensive data science project from the 42 curriculum. It introduces Exploratory Data Analysis (EDA) and implements a One-vs-All Logistic Regression classifier from scratch in Python—without external ML training libraries like Scikit-Learn.
  The project explores data visualization, standardization, cost functions, gradient descent optimization, and classification metrics.

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
5. In-Depth Mathematical Concepts

---

## About The Project
The goal of this project is to dive into Data Science by analyzing student records from Hogwarts, handling missing data, visualizing features, and building a machine learning classifier from scratch.

Since logistic regression is inherently binary, the program uses the One-vs-All (OvR) strategy to classify students into one of four houses: Gryffindor, Hufflepuff, Ravenclaw, or Slytherin.

The core prediction relies on the Sigmoid function applied to a linear combination of features:
z = X * theta
h_theta(x) = 1 / (1 + exp(-z))

---

## Features & Architecture

- describe.py: A custom implementation of pandas' describe() function. Computes count, mean, std, min, percentiles (25%, 50%, 75%), and max for all numerical features.
- histogram.py: Plots histograms for each Hogwarts course to discover which course is homogeneous across all houses (e.g., Care of Magical Creatures).
- scatter_plot.py: Generates a scatter plot to identify two features that are similar / correlated (e.g., Astronomy and Defense Against the Dark Arts).
- pair_plot.py: Visualizes pairwise relationships across all features using seaborn or matplotlib and save it in 'Correlation between courses.png'.
- logreg_train.py: Implements Gradient Descent with Z-score normalization to iteratively train binary classifiers for each house. Saves the optimal weights into a weights.json file.
- logreg_predict.py: Loads the trained weights, normalizes the test dataset, computes probabilities for each house, applies the argmax rule, and outputs a houses.csv prediction file.
- precision.py (or evaluation script): Computes the accuracy and error rate by comparing predictions against ground truth labels.

---

## Prerequisites & Installation

Make sure you have Python 3 and pip installed on your machine. 

### Installing Dependencies:
You will need numpy, pandas, matplotlib, and seaborn for data manipulation and visualization:
pip install numpy pandas matplotlib seaborn

---

## Usage and Execution Guide

1. Exploratory Data Analysis (describe.py):
Displays summary statistics for the training dataset.
python describe.py datasets/dataset_train.csv

2. Data Visualization:
Explore feature distributions and correlations.
python histogram.py 
python scatter_plot.py
python pair_plot.py 

3. Training the Model (logreg_train.py):
Trains the logistic regression models using gradient descent and saves weights to weights.json.
python logreg_train.py datasets/dataset_train.csv

4. Predicting Houses (logreg_predict.py):
Generates house predictions for the test dataset and creates houses.csv.
python logreg_predict.py datasets/dataset_test.csv weights.json

5. Evaluating Precision (precision.py):
Compares predictions against actual labels to check the model's accuracy.
python precision.py houses.csv datasets/dataset_truth.csv

---

## In-Depth Mathematical Concepts

1. Z-Score Normalization:
To ensure gradient descent converges smoothly and features are on the same scale, data is standardized:
x_norm = (x - mu) / sigma

2. Sigmoid Activation Function:
Maps linear scores (z) into probabilities between 0 and 1:
h_theta(x) = 1 / (1 + exp(-z))

3. Gradient Descent Optimization:
Iteratively updates weight vector theta using the derivative of the cost function with a learning rate alpha:
theta_j := theta_j - alpha * (1 / m) * sum((h_theta(x^(i)) - y^(i)) * x_j^(i))

4. One-vs-All (OvR) & Argmax Decision Rule:
For K classes, train K separate binary classifiers. For a given student, compute the probability for each house and select the class with the highest probability using np.argmax(axis=1):
Predicted House = argmax_k P(House_k | x)