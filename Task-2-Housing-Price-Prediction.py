# Task 2: Real Estate Housing Price Prediction
# Internship: Machine Learning and Data Science - Datheon
# Author: Nafris

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("TASK 2: HOUSING PRICE PREDICTION")

# 1. Load Dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name='MedHouseVal')
print("Dataset Shape:", X.shape)
print(X.head())
print(y.describe())

# 2. EDA - Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation - California Housing')
plt.tight_layout()
plt.savefig('Task2_Correlation.png')
plt.show()

# 3. EDA - Distribution
plt.figure(figsize=(8, 5))
sns.histplot(y, kde=True, color='green')
plt.title('Distribution of Median House Value')
plt.xlabel('Median House Value')
plt.tight_layout()
plt.savefig('Task2_Distribution.png')
plt.show()

# 4. EDA - MedInc vs Price
plt.figure(figsize=(8, 5))
plt.scatter(X['MedInc'], y, alpha=0.3, color='blue')
plt.title('Median Income vs House Price')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.tight_layout()
plt.savefig('Task2_Scatter.png')
plt.show()

# 5. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 6. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Models
models = {
    "OLS": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1, max_iter=5000)
}

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv = cross_val_score(model, X, y, cv=5, scoring='r2').mean()
    print(f"\n===== {name} =====")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"CV R2 Mean: {cv:.4f}")
    results.append([name, rmse, mae, r2])

results_df = pd.DataFrame(results, columns=['Model', 'RMSE', 'MAE', 'R2_Score'])
print("\nFINAL COMPARISON")
print(results_df.sort_values(by='R2_Score', ascending=False))

# 8. Comparison Chart
plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x='Model', y='R2_Score', palette='viridis')
plt.title('Model Comparison - R2 Score')
plt.ylabel('R2 Score')
plt.ylim(0.5, 0.7)
plt.tight_layout()
plt.savefig('Task2_Comparison.png')
plt.show()

# 9. Actual vs Predicted - Ridge Best Model
best_model = Ridge(alpha=1.0)
best_model.fit(X_train_scaled, y_train)
y_pred_best = best_model.predict(X_test_scaled)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_best, alpha=0.4, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs Predicted - Ridge')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.tight_layout()
plt.savefig('Task2_Actual_vs_Predicted.png')
plt.show()

# 10. Feature Importance
coef = pd.Series(best_model.coef_, index=X.columns).sort_values()
plt.figure(figsize=(8, 5))
coef.plot(kind='barh', color='teal')
plt.title('Feature Importance - Ridge Coefficients')
plt.xlabel('Coefficient Value')
plt.tight_layout()
plt.savefig('Task2_Feature_Importance.png')
plt.show()

print("\nKey Insight: Median Income is strongest predictor. Ridge handles multicollinearity best.")
print("Task 2 Completed Successfully!")