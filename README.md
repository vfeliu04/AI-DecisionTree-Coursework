# Artificial Intelligence Coursework - Decision Trees

This repository contains the coursework for COMP2611 - Artificial Intelligence, focusing on the implementation and evaluation of decision tree models for classification tasks. The coursework demonstrates the end-to-end process of training, optimizing, and evaluating decision trees using Python and the scikit-learn library.

## Overview

The project involves the following key tasks:
- **Data Loading and Preprocessing**: Loading data from a CSV file, handling missing values, and preparing the dataset for model training.
- **Training a Decision Tree Model**: Implementing a decision tree classifier and training it on the preprocessed data.
- **Model Evaluation**: Assessing the model's performance on a testing set using metrics like accuracy and recall.
- **Pruning and Optimization**: Applying cost complexity pruning to the decision tree to find an optimal balance between model complexity and accuracy.
- **Feature Importance Analysis**: Identifying the most significant features contributing to the model's predictions.

## File Descriptions

- `decision_tree_coursework.py`: Main Python script containing the implementation of data preprocessing, model training, and evaluation functions.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn

## Usage

1. Clone the repository
2. Navigate to the project directory
3. Run the script (ensure you have the necessary data file, e.g., `DT.csv`, in the same directory)


## Tasks and Functions

- **`load_data(file_path)`:** Loads data from a CSV file specified by `file_path`.
- **`filter_data(data)`:** Filters out rows with missing values.
- **`statistics_data(data)`:** Calculates and returns the coefficient of variation for each feature in the dataset.
- **`split_data(data)`:** Splits the dataset into training and testing sets.
- **`train_decision_tree(x_train, y_train, ccp_alpha)`:** Trains a decision tree model with an optional cost complexity pruning parameter.
- **`evaluate_model(model, x_test, y_test)`:** Evaluates the trained model on the testing set and returns performance metrics.
- **`optimal_ccp_alpha(x_train, y_train, x_test, y_test)`:** Determines the optimal cost complexity pruning parameter.
- **`tree_depths(model)`:** Returns the depth of the given decision tree model.
- **`important_feature(x_train, y_train)`:** Identifies and returns the most important feature used by the decision tree.

## Contributing

This project is part of an academic coursework assignment. Therefore it is not fully mine.
