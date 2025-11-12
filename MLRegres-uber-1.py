import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


data = pd.read_csv("uber.csv")


data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"], errors="coerce")


print("Missing values (before):")
print(data.isnull().sum())


data = data.dropna().reset_index(drop=True)

print("\nMissing values (after dropna):")
print(data.isnull().sum())


Q1 = data["fare_amount"].quantile(0.25)
Q3 = data["fare_amount"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\nFare IQR bounds: lower={lower_bound:.2f}, upper={upper_bound:.2f}")
data_no_outliers = data[(data["fare_amount"] >= lower_bound) & (data["fare_amount"] <= upper_bound)].copy()
print(f"Rows before outlier removal: {len(data)}, after: {len(data_no_outliers)}")


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.boxplot(x=data["fare_amount"])
plt.title('Fare Amount (original)')
plt.subplot(1,2,2)
sns.boxplot(x=data_no_outliers["fare_amount"])
plt.title('Fare Amount (without outliers)')
plt.tight_layout()
plt.show()


corr_df = data_no_outliers.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix (numeric features)')
plt.show()


X = data_no_outliers[['pickup_longitude', 'pickup_latitude',
                      'dropoff_longitude', 'dropoff_latitude',
                      'passenger_count']].copy()
y = data_no_outliers['fare_amount'].copy()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


y_pred_lr = lr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print("\nModel results:")
print(f"Linear Regression  -> R2: {r2_lr:.4f}   RMSE: {rmse_lr:.4f}")
print(f"Random Forest      -> R2: {r2_rf:.4f}   RMSE: {rmse_rf:.4f}")
