import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("results.csv")


def getresults(row):
    if row['home_score'] > row['away_score']:
        return 'Home win'  # return home win
    elif row['home_score'] < row['away_score']:
        return 'Away win'  # return away win
    else:
        return "Draw"  # return draw


# Creates new column result:'Home win', 'Away win', or 'Draw'
df['results'] = df.apply(getresults, axis=1)

features = ['home_team', 'away_team',
            'tournament', 'city', 'country']
X = df[features].copy()
y = df['results'].copy()

labels_encoders = {}
for col in X.columns:
    # assings a unique number to each team ,tournament, city, country, and neutral
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    labels_encoders[col] = le

le_target = LabelEncoder()
# Change 'Home win', 'Away win', and 'Draw' to numerical values
y = le_target.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 80% data used for training, and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# max_depth=5 limits the complexity to prevent overfitting
# model = DecisionTreeClassifier(max_depth=5, random_state=42)
# model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
# model = LogisticRegression(max_iter=1000, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')

'''
mlp_model = MLPClassifier(hidden_layer_sizes=(
    100,), max_iter=1000, random_state = 42)
'''
print(df['results'].value_counts())
cv_knnscores = cross_val_score(
    knn_model, X_scaled, y, cv=5)  # Cross-validation 5 times

'''
cv_mlpscores = cross_val_score(
    mlp_model, X_scaled, y, cv=5)  # Cross-validation 5 times
'''
print(
    f"Cross-validation Accuracy (KNN): {cv_knnscores.mean():.4f} +/- {cv_knnscores.std():.4f}")

'''
print(
    f"Cross-validation Accuracy (MLP): {cv_mlpscores.mean():.4f} +/- {cv_mlpscores.std():.4f}")
'''
knn_model.fit(X_train, y_train)

knn_y_pred = knn_model.predict(X_test)

print("\nKNN Classification Report:")
print(classification_report(y_test, knn_y_pred))
kaccuracy = accuracy_score(y_test, knn_y_pred)
print(f'KNN Accuracy: {kaccuracy * 100:.2f}%')

print("KNN Confusion Matrix:")
print(confusion_matrix(y_test, knn_y_pred))

'''
mlp_model.fit(X_train, y_train)

mlp_y_pred = mlp_model.predict(X_test)
print("\nMLP Classification Report:")
print(classification_report(y_test, mlp_y_pred))
maccuracy = accuracy_score(y_test, mlp_y_pred)
print(f'MLP Accuracy: {maccuracy * 100:.2f}%')

print("MLP Confusion Matrix:")
print(confusion_matrix(y_test, mlp_y_pred))
'''
