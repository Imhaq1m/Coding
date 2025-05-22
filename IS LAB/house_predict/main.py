import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

df = pd.read_csv('train.csv')
X = df.select_dtypes(include=['number'])
X = X.drop(columns=['Id', 'SalePrice'])
y = df['SalePrice']

model = RandomForestRegressor(random_state=0)
model.fit(X, y)

df_final = pd.read_csv('test.csv')
X_final = df_final.select_dtypes(include=['number'])
X_final = X_final.drop(columns=['Id'])
pred_final = model.predict(X_final)

submission = pd.DataFrame({
    'Id': df_final['Id'],
    'SalePrice': pred_final
})

print(submission)

submission.to_csv('submission.csv', index=False)
