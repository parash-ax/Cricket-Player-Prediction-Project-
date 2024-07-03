import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

download_path = 'C:\Users\JACK\Downloads\archive.zip\Batting'  

df = pd.read_csv(download_path + 'ODI data.csv')

X = df[['Matches', 'Runs']]
y = df['Batting Average']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

new_player = np.array([[20, 800]])
predicted_avg = model.predict(new_player)
print(f'Predicted Batting Average: {predicted_avg[0]}')
