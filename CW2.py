# COMP2611-Artificial Intelligence-Coursework#2 - Descision Trees

# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.tree import export_text
import warnings
import os

# STUDENT NAME: Vicente Feliu
# STUDENT EMAIL: sc22vf@leeds.ac.uk

# Print tree structure
def print_tree_structure(model, header_list):
    
    tree_rules = export_text(model, feature_names=header_list[:-1])
    
    print(tree_rules)
    
# Task 1 [10 marks]: Load the data from the CSV file and give back the number of rows
def load_data(file_path, delimiter=','):
    
    # Init variables
    num_rows, data, header_list= None, None, None

    # Check if the file exists
    if not os.path.isfile(file_path):
        warnings.warn(f"Task 1: Warning - CSV file '{file_path}' does not exist.")
        return None, None, None

    # Read the CSV file using pandas
    df = pd.read_csv(file_path, delimiter=delimiter)

    # Convert DataFrame to a numpy array
    data = df.values

    # Get the number of rows
    num_rows = data.shape[0]

    # Extract header list
    header_list = df.columns.tolist()

    return num_rows, data, header_list

# Task 2[10 marks]: Give back the data by removing the rows with -99 values 
def filter_data(data):
    
    # Init variables
    filtered_data=[None]*1
    data = np.array(data)
    
    # Find the rows where any column has a value of -99
    rows_with_missing_values = np.any(data == -99, axis=1)
    
    # Invert the condition to keep rows without -99 values
    filtered_data = data[~rows_with_missing_values]

    # Return filtered data
    return filtered_data

# Task 3 [10 marks]: Data statistics, return the coefficient of variation for each feature, make sure to remove the rows with nan before doing this. 
def statistics_data(data):
    
    # Init variables and filter data
    coefficient_of_variation=None
    data=filter_data(data)
    
    # Calculate the mean and standard deviation for each feature
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    
    # Calculate the coefficient of variation for each feature, where a parameter to avoid division by zero is used,
    # filling those cases with zeros instead.
    coefficient_of_variation = np.divide(std_dev, mean, out=np.zeros_like(std_dev), where=mean!=0)

    return coefficient_of_variation

# Task 4 [10 marks]: Split the dataset into training (70%) and testing sets (30%), 
# use train_test_split of scikit-learn to make sure that the sampling is stratified, 
# meaning that the ratio between 0 and 1 in the label column stays the same in train and test groups.
# Also when using train_test_split function from scikit-learn make sure to use "random_state=1" as an argument. 
def split_data(data, test_size=0.3, random_state=1):
    # Init variables
    x_train, x_test, y_train, y_test=None, None, None, None
    
    # Set numpy's random num generator to a fixed value of 1
    np.random.seed(1)
    
    # Assuming the last column of the data is the label
    X = data[:, :-1]
    y = data[:, -1]
    
    # Split the data into training and testing sets while maintaining label ratios
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    return x_train, x_test, y_train, y_test

# Task 5 [10 marks]: Train a decision tree model with cost complexity parameter of 0
def train_decision_tree(x_train, y_train,ccp_alpha=0):
    # Init variable
    model=None
    
    # Initialize the DecisionTreeClassifier model with the given ccp_alpha
    model = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
    
    # Fit the model to the training data
    model.fit(x_train, y_train)
    
    return model

# Task 6 [10 marks]: Make predictions on the testing set 
def make_predictions(model, X_test):
    # Init variable
    y_test_predicted=None
    
    # Use the trained model to predict the labels of the test set
    y_test_predicted = model.predict(X_test)
    
    return y_test_predicted

# Task 7 [10 marks]: Evaluate the model performance by taking test dataset and giving back the accuracy and recall 
def evaluate_model(model, x, y):
    
    # Init variables
    accuracy, recall=None,None
    
    # Use the model to make predictions on the given dataset
    y_pred = model.predict(x)
    
    # Calc accuracy
    accuracy = accuracy_score(y, y_pred)
    
    # Calc recall
    recall = recall_score(y, y_pred, pos_label=1)
    
    return accuracy, recall

# Task 8 [10 marks]: Write a function that gives the optimal value for cost complexity parameter
# which leads to simpler model but almost same test accuracy as the unpruned model (+-1% of the unpruned accuracy)
def optimal_ccp_alpha(x_train, y_train, x_test, y_test):
    # Train an unpruned decision tree to get the baseline accuracy
    unpruned_model = DecisionTreeClassifier(random_state=1)
    unpruned_model.fit(x_train, y_train)
    baseline_accuracy = accuracy_score(y_test, unpruned_model.predict(x_test))
    
    # Init variables for search
    ccp_alpha_increment = 0.001
    current_ccp_alpha = 0
    best_ccp_alpha = 0
    
    # Search for the optimal ccp_alpha
    while True:
        current_ccp_alpha += ccp_alpha_increment
        model = DecisionTreeClassifier(ccp_alpha=current_ccp_alpha, random_state=1)
        model.fit(x_train, y_train)
        current_accuracy = accuracy_score(y_test, model.predict(x_test))
        
        # Check if the accuracy is outside the ±1% range of baseline accuracy
        if abs(current_accuracy - baseline_accuracy) > 0.01 * baseline_accuracy:
            break
        else:
            # Update the best ccp_alpha
            best_ccp_alpha = current_ccp_alpha
            
    return best_ccp_alpha

# Task 9 [10 marks]: Write a function that gives the depth of a decision tree that it takes as input.
def tree_depths(model):
    # Init variable
    depth=None
    
    # Get the depth of the tree
    depth = model.get_depth()
    
    return depth

 # Task 10 [10 marks]: Feature importance 
def important_feature(x_train, y_train,header_list):
    # Init variable
    best_feature=None
    ccp_alpha_increment = 0.001
    ccp_alpha = 0
    
    while True:
        # Train a decision tree model with the current ccp_alpha
        model = DecisionTreeClassifier(random_state=1, ccp_alpha=ccp_alpha)
        model.fit(x_train, y_train)
        
        # Check the depth of the tree, and find the most important feature
        if model.get_depth() == 1:
            feature_importances = model.feature_importances_
            best_feature_index = feature_importances.argmax()
            best_feature = header_list[best_feature_index]
            break
        else:
            ccp_alpha += ccp_alpha_increment
    
    return best_feature

# Example usage (Template Main section):
if __name__ == "__main__":
    # Load data
    file_path = "DT.csv"
    num_rows, data, header_list = load_data(file_path)
    print(f"Data is read. Number of Rows: {num_rows}"); 
    print("-" * 50)

    # Filter data
    data_filtered = filter_data(data)
    num_rows_filtered=data_filtered.shape[0]
    print(f"Data is filtered. Number of Rows: {num_rows_filtered}"); 
    print("-" * 50)

    # Data Statistics
    coefficient_of_variation = statistics_data(data_filtered)
    print("Coefficient of Variation for each feature:")
    for header, coef_var in zip(header_list[:-1], coefficient_of_variation):
        print(f"{header}: {coef_var}")
    print("-" * 50)
    # Split data
    x_train, x_test, y_train, y_test = split_data(data_filtered)
    print(f"Train set size: {len(x_train)}")
    print(f"Test set size: {len(x_test)}")
    print("-" * 50)
    
    # Train initial Decision Tree
    model = train_decision_tree(x_train, y_train)
    print("Initial Decision Tree Structure:")
    print_tree_structure(model, header_list)
    print("-" * 50)
    
    # Evaluate initial model
    acc_test, recall_test = evaluate_model(model, x_test, y_test)
    print(f"Initial Decision Tree - Test Accuracy: {acc_test:.2%}, Recall: {recall_test:.2%}")
    print("-" * 50)
    # Train Pruned Decision Tree
    model_pruned = train_decision_tree(x_train, y_train, ccp_alpha=0.002)
    print("Pruned Decision Tree Structure:")
    print_tree_structure(model_pruned, header_list)
    print("-" * 50)
    # Evaluate pruned model
    acc_test_pruned, recall_test_pruned = evaluate_model(model_pruned, x_test, y_test)
    print(f"Pruned Decision Tree - Test Accuracy: {acc_test_pruned:.2%}, Recall: {recall_test_pruned:.2%}")
    print("-" * 50)
    # Find optimal ccp_alpha
    optimal_alpha = optimal_ccp_alpha(x_train, y_train, x_test, y_test)
    print(f"Optimal ccp_alpha for pruning: {optimal_alpha:.4f}")
    print("-" * 50)
    # Train Pruned and Optimized Decision Tree
    model_optimized = train_decision_tree(x_train, y_train, ccp_alpha=optimal_alpha)
    print("Optimized Decision Tree Structure:")
    print_tree_structure(model_optimized, header_list)
    print("-" * 50)
    
    # Get tree depths
    depth_initial = tree_depths(model)
    depth_pruned = tree_depths(model_pruned)
    depth_optimized = tree_depths(model_optimized)
    print(f"Initial Decision Tree Depth: {depth_initial}")
    print(f"Pruned Decision Tree Depth: {depth_pruned}")
    print(f"Optimized Decision Tree Depth: {depth_optimized}")
    print("-" * 50)
    
    # Feature importance
    important_feature_name = important_feature(x_train, y_train,header_list)
    print(f"Important Feature for Fraudulent Transaction Prediction: {important_feature_name}")
    print("-" * 50)
        
# References: 
"""Arora, S. 2020. Cost Complexity Pruning in Decision Trees | Decision Tree. Analytics Vidhya.
    [Online]. [Accessed 16 April 2024]. Available from:
    https://www.analyticsvidhya.com/blog/2020/10/cost-complexity-pruning-decision-trees/."""

"""L, G.D. 2023. Five Methods for Data Splitting in Machine Learning. Medium.
    [Online]. [Accessed 1 April 2024]. Available from:
    https://medium.com/@tubelwj/five-methods-for-data-splitting-in-machine-learning-27baa50908ed#:~:text=Data%20splitting%20is%20a%20crucial."""



