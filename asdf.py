import pandas as pd
# Location of dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
# Read dataset to pandas dataframe
irisdata = pd.read_csv(url, names=names)
irisdata.head()

X = irisdata.iloc[:, 0:4] #Assign data from first four columns to X variable
y = irisdata.select_dtypes(include=[object]) #Assign data from first fifth columns to y variable
y.head() #Display data from first fifth of y variable
y.Class.unique() #Display unique values in y series


from sklearn import preprocessing
le = preprocessing.LabelEncoder() #encode categorical labels into numeric values
y = y.apply(le.fit_transform) #Applies the LabelEncoder to transform the categorical column into numeric values.
y.Class.unique() #Displays the unique encoded values in the Class column after transformation

from sklearn.model_selection import train_test_split
# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.preprocessing import StandardScaler
# Create an instance of StandardScaler to standardize the features
scaler = StandardScaler()
# Fit the scaler to the training data and transform the training data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
# Transform the testing data using the same scaler
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier
# Create an instance of MLPClassifier with three hidden layers, each containing 10 neurons, and a maximum of 1000 iterations
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
# Train the MLPClassifier on the scaled training data
mlp.fit(X_train, y_train.values.ravel())
# Display the configuration of the trained MLPClassifier
MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
# Use the trained model to make predictions on the testing data
predictions = mlp.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
# Print the confusion matrix to evaluate the performance of the model
print(confusion_matrix(y_test, predictions))
# Print the classification report to get detailed metrics such as precision, recall, and F1-score
print(classification_report(y_test, predictions))
