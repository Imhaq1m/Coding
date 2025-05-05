# Data Upload and Reading
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Read the uploaded dataset (Assuming wdbc.dat is in a CSV format with headers)
# You can modify the separator based on the dataset format (',', '\t', etc.)
df = pd.read_csv('wdbc.data', header=None)

# Data Inspection
# Inspect the first few rows of the dataset
print(df.head())
# Assuming that the first column is the ID and the second column is the target label
# and the rest are features
df.columns = ['ID', 'Target'] + \
    [f'Feature{i}' for i in range(1, df.shape[1] - 1)]

# Data preprocessing
# Drop the ID column as it is not needed for model training
df = df.drop(columns=['ID'])
# Convert the target labels ('M' = Malignant, 'B' = Benign) to numerical values (1 = Malignant, 0 = Benign)
df['Target'] = df['Target'].map({'M': 1, 'B': 0})
# Split the data into features (X) and target labels (y)
X = df.drop(columns=['Target'])
y = df['Target']
# Feature Scaling
# Standardize the features (important for neural networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Model Training
# Create an instance of MLPClassifier with three hidden layers, each containing 10 neurons, and a maximum of 1000 iterations
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
# Train the MLPClassifier on the scaled training data
mlp.fit(X_train, y_train.values.ravel())

# Model Evaluation
# Use the trained model to make predictions on the testing data
predictions = mlp.predict(X_test)
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
# Print the confusion matrix to evaluate the performance of the model
print(confusion_matrix(y_test, predictions))
# Print the classification report to get detailed metrics such as precision, recall, and F1-score
print(classification_report(y_test, predictions))
# Visualize the confusion matrix using seaborn heatmap
class_labels = ['Benign', 'Malignant']
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d',
            cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
# plt.show()
