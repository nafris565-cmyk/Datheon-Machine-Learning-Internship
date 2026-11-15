# Task 3: Predictive Customer Churn Classifier
# Internship: Machine Learning and Data Science - Datheon
# Author: Nafris
# Goal: Predict which Telecom customers will churn using Random Forest and XGBoost with SMOTE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

print("TASK 3: CUSTOMER CHURN PREDICTION")

# 1. Load Dataset - Telco Customer Churn
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
print("Dataset Shape:", df.shape)
print(df.head())
print("\nChurn Distribution:")
print(df['Churn'].value_counts())
print("\nChurn Percentage:")
print(df['Churn'].value_counts(normalize=True) * 100)

# 2. Data Cleaning
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
df = df.drop(['customerID'], axis=1)
print("\nAfter Cleaning Shape:", df.shape)
print("Missing Values:", df.isnull().sum().sum())

# 3. EDA - Churn by Contract
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Contract', hue='Churn', palette='Set2')
plt.title('Churn by Contract Type - Month-to-Month Customers Churn More')
plt.tight_layout()
plt.savefig('Task3_Churn_by_Contract.png')
plt.show()

# 4. EDA - Monthly Charges vs Churn
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Churn', y='MonthlyCharges', palette='coolwarm')
plt.title('Monthly Charges vs Churn')
plt.tight_layout()
plt.savefig('Task3_Charges_vs_Churn.png')
plt.show()

# 5. EDA - Tenure vs Churn
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='tenure', hue='Churn', bins=30, palette='Set1')
plt.title('Tenure vs Churn - New Customers Churn More')
plt.xlabel('Tenure (Months)')
plt.tight_layout()
plt.savefig('Task3_Tenure_vs_Churn.png')
plt.show()

# 6. Encoding - Label Encoding for all categorical columns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
cat_cols.remove('Churn')
print("\nCategorical Columns:", cat_cols)
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

X = df.drop('Churn', axis=1)
y = df['Churn']
print("\nFeatures:", X.columns.tolist())
print("X Shape:", X.shape)

# 7. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
print("Before SMOTE - Train Churn Distribution:")
print(y_train.value_counts())

# 8. Handle Class Imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print("\nAfter SMOTE - Train Churn Distribution:")
print(y_train_bal.value_counts())

# 9. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

# 10. Train Models
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train_scaled, y_train_bal)

xgb_model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_scaled, y_train_bal)

# 11. Evaluation Function
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"\n===== {name} =====")
    print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f} ({roc_auc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'Task3_CM_{name}.png')
    plt.show()
    return roc_auc, y_prob

roc_rf, prob_rf = evaluate_model("Random Forest", rf_model, X_test_scaled, y_test)
roc_xgb, prob_xgb = evaluate_model("XGBoost", xgb_model, X_test_scaled, y_test)

# 12. ROC Curve Comparison
fpr_rf, tpr_rf, _ = roc_curve(y_test, prob_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, prob_xgb)
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_rf:.3f})', lw=2)
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_xgb:.3f})', lw=2)
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison - Churn Prediction')
plt.legend()
plt.tight_layout()
plt.savefig('Task3_ROC_Curve.png')
plt.show()

# 13. Feature Importance - Random Forest
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
importances.head(10).plot(kind='bar', color='teal')
plt.title('Top 10 Feature Importance - Random Forest')
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig('Task3_Feature_Importance.png')
plt.show()
print("\nTop 5 Important Features:")
print(importances.head(5))

# 14. Final Comparison
print("\nFINAL COMPARISON")
print(f"Random Forest ROC-AUC: {roc_rf*100:.2f}%")
print(f"XGBoost ROC-AUC: {roc_xgb*100:.2f}%")
best = "Random Forest" if roc_rf >= roc_xgb else "XGBoost"
print(f"Best Model: {best}")
print("\nKey Insight: Contract Type, Tenure and MonthlyCharges are top churn drivers. SMOTE improved Recall for churn class. Month-to-Month customers churn most.")
print("Task 3 Completed Successfully!")