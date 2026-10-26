# Machine Learning Classifiers from Scratch in C++

An educational C++ implementation of three common classification algorithms: K-Nearest Neighbors (KNN), Gaussian Naive Bayes, and a linear Support Vector Machine (SVM). The classifiers are written using only the C++ standard library, without any machine learning packages.

## Project overview

Each classifier is a C++ class with a `fit` method for training and `predict` methods for inference. The three classes share the same interface, so they can be trained and tested on the same data in the same way.

The file was written with assistance from an AI coding assistant for guidance and debugging, as noted in the source file header.

## Algorithms

### K-Nearest Neighbors

- Stores the training data.
- Predicts by measuring the Euclidean distance from a new point to every training point and taking the most common label among the `k` closest points.

### Gaussian Naive Bayes

- Estimates the mean and standard deviation of each feature for each class during training.
- Predicts by using the Gaussian probability density function and picking the class with the highest combined probability.

### Linear SVM

- Learns a weight vector and bias using gradient descent on the hinge loss with L2 regularization.
- Predicts the class from the sign of the decision value.

Gaussian likelihood is the probability of seeing a feature value given that its class has a certain mean and spread. Hinge loss is the penalty an SVM uses to push points to the correct side of the decision boundary with some margin.

## Repository contents

```text
machine-learning-classifiers-from-scratch-c-/
├── README.md
└── ML_Classifiers.cpp
```

## Setup and usage

This project requires a C++17 compiler such as `g++` or `clang++`.

```bash
g++ -std=c++17 -O2 ML_Classifiers.cpp -o ml_classifiers
./ml_classifiers
```

The `main` function trains all three classifiers on a small sample dataset and prints each model's predictions for two test points.

## Limitations and future improvements

- The demo uses a three-sample toy dataset, so it shows how the classes work but does not measure accuracy.
- The SVM is binary only and the label mapping is hard-coded to `Male` and `Female`.
- There is no train/test split, cross-validation, or feature scaling.
- Future improvements could include loading data from a CSV file, reporting accuracy, precision, and recall, supporting multiple classes, and comparing results against a library such as scikit-learn.

## Disclaimer

This project is for educational purposes only.
