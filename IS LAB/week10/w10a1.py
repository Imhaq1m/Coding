import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
sns.set(color_codes=True)

df = pd.read_csv('data.csv')
'''
print(df.head(5))
print(df.tail(5))
'''
df = df.drop(['Engine Fuel Type', 'Market Category', 'Vehicle Style', 'Popularity', 'Number of Doors', 'Vehicle Size'], axis=1)
print(df.dtypes)

df = df.rename(columns={"Engine HP": "HP", "Engine Cylinders": "Cylinders", "Transmission Type" : "Transmission"})
print(df.shape)
duplicate_rows_df = df[df.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)
df = df.drop_duplicates()
print(df.head(5))
print(df.count())
print(df.isnull().sum())
sns.boxplot(x=df['MSRP'])

Q1 = df.select_dtypes(include=['number']).quantile(0.25)
Q3 = df.select_dtypes(include=['number']).quantile(0.75)
IQR = Q3 - Q1
print(IQR)

df.Make.value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title("Number of cars by make")
plt.ylabel('Number of cars')
plt.xlabel('Make')

plt.figure(figsize=(10,5))
c=df.select_dtypes(include=['number']).corr()
sns.heatmap(c,cmap="BrBG", annot=True)

fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(df['HP'], df['MSRP'])
ax.set_xlabel('HP')
ax.set_ylabel('MSRP')
plt.show()


