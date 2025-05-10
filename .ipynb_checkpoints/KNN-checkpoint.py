from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#datasets from sklearn to load the iris dataset
#pandas for data manipulation and analysis
#numpy for numerical opeartions
#matplotlib.pyplot for data visualization
#Sets the plotting style to ggplot for better aesthetics

iris = datasets.load_iris()
type(iris)
#Loads the Iris dataset into the variable iris
#Checks the type of the iris object

X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
print(df.head())
#Extracts the feature data (X) and target labels (y) from the iris dataset
#Creates a DataFrame from the feature data (X) with column names according to the feature names in the dataset 
#Prints the first few rows of the DataFrame for a quick preview

type(iris.data), type(iris.target)
#Checks the type of the feature data and the target

samples = iris.data
print(samples)
#Assigns the feature data to the variable samples
#Prints the feature data for inpsection

from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
y_kmc = model.fit(samples)
#imports the KMeans clustering algorithm from sklean
#Initializes a KMeans model with n_clusters=3 
#Fits the KMeans model to the feature data

labels = model.predict(samples)
print(labels)
#Predicts the cluster labels fr the feature data using the trained KMeans model
#Prints the cluster labels, which indicates the cluster assignments for each sample.

new_samples = np.array([[5.6, 2.8, 3.9, 1.1], [5.7, 2.6, 3.8, 1.3], [4.7, 3.2, 1.3, 0.2]])
#Defines new samples as a NumPy array,

xs = samples[:,0]
ys = samples[:,2]
plt.scatter(xs, ys, c=labels)
plt.show()
#Extracts the first feature as xs variable and the third feature as ys for visualization
#Creates a scatter plot of the data points, coloring them by their cluster labels
#Displays the plot using plt.show()
