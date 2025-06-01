import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os

os.makedirs('outputs/models', exist_ok=True)

data = pd.read_csv('HousePrices.csv')

X = data[['GrLivArea', 'OverallQual', 'GarageCars', 'TotalBsmtSF']]
y = data['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, 'outputs/models/best_model.pkl')

print("Model is now trained and saved as outputs/models/best_model.pkl")
