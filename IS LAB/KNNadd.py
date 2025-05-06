# Simple Neural Network Lab Exercise with Diabetes Dataset and Cross-Validation
# ## Objective:
# - Train a neural network to predict diabetes using the popular Pima Indians Diabetes Dataset.
from sklearn.datasets import load_diabetes
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import numpy as np
# - Use cross-validation for model evaluation.
# ## Libraries Needed:
# - scikit-learn
# - tensorflow (for the neural network)
# - pandas (for data handling)
# - numpy (for numerical operations)
# Import necessary libraries
# ## Question 1: What is the goal of this lab exercise?
# ## Answer 1:
# Load Diabetes dataset (For this case, we'll use sklearn's diabetes dataset as an example)
# In practice, you can load the 'pima-indians-diabetes.csv' dataset.
diabetes_data = load_diabetes()
# Prepare the features (X) and target variable (y)
X = diabetes_data.data
y = diabetes_data.target
# Binarize target (1 = Positive, 0 = Negative)
y = np.where(y > np.median(y), 1, 0)
# Split the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# Feature scaling (standardizing the dataset)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Initialize the Neural Network classifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
# Perform cross-validation
cv_scores = cross_val_score(
    mlp, X_train_scaled, y_train, cv=5)  # 5-fold crossvalidation
# Print cross-validation scores
print(f"Cross-validation scores: {cv_scores}")
print(f"Average cross-validation score: {cv_scores.mean():.4f}")
# ## Question 2: How is the diabetes dataset loaded and used?
# ## Answer 2:
# target values, so we binarized it by setting values greater than the median to `1` (positive class) and others to `0` (negative class).
# Train the model on the entire training dataset
mlp.fit(X_train_scaled, y_train)
# Evaluate the model on the test dataset
test_accuracy = mlp.score(X_test_scaled, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")
# ## Question 3: What is cross-validation, and why is it important in this exercise?
# ## Answer 3:
# ## Question 4: What does the `MLPClassifier` represent in this context?
# ## Answer 4:
# ## Question 5: What role does `StandardScaler` play in this neural network model?
# ## Answer 5:
# ## Question 6: What can be inferred from the test accuracy and cross-validation scores?
# ## Answer 6:
# ## Question 7: What is the result?
# ## Answer 7
