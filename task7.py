# task7_walmart_forecast.py

import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# 1. Load Dataset
df = pd.read_csv("Walmart.csv")  # change to your dataset path

# 2. Parse date column (day-month-year)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# 3. Sort by Date
df = df.sort_values('Date')

# 4. Create time-based features
df['day'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year
df['day_of_week'] = df['Date'].dt.dayofweek

# 5. Lag features
df['sales_lag_1'] = df['Weekly_Sales'].shift(1)
df['sales_lag_2'] = df['Weekly_Sales'].shift(2)

# 6. Rolling mean features
df['rolling_mean_3'] = df['Weekly_Sales'].shift(1).rolling(window=3).mean()
df['rolling_mean_7'] = df['Weekly_Sales'].shift(1).rolling(window=7).mean()

# Drop NaN values from lag/rolling creation
df = df.dropna()

# 7. Train-test split (time-aware: last 20% for testing)
split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

# 8. Features & Target
feature_cols = ['day', 'month', 'year', 'day_of_week', 'sales_lag_1', 'sales_lag_2', 'rolling_mean_3', 'rolling_mean_7']
X_train = train_df[feature_cols]
y_train = train_df['Weekly_Sales']
X_test = test_df[feature_cols]
y_test = test_df['Weekly_Sales']

# 9. Model (LightGBM)
model = LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# 10. Predictions
y_pred = model.predict(X_test)

# 11. Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.2f}")

# 12. Plot Actual vs Predicted
plt.figure(figsize=(12, 6))
plt.plot(test_df['Date'], y_test, label='Actual Sales', color='blue')
plt.plot(test_df['Date'], y_pred, label='Predicted Sales', color='red')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.title('Walmart Sales Forecast - Actual vs Predicted')
plt.legend()
plt.show()
