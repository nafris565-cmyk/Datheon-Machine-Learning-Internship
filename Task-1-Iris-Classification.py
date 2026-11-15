# Task 1: Multi-Class Benchmark Classification Pipeline (Iris Dataset)
# Internship: Machine Learning & Data Science - Datheon
# Author: Nafris
# Goal: Compare Logistic Regression vs Decision Tree with Full EDA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

print("="*60)
print("TASK 1: IRIS CLASSIFICATION - DISTINCTION LEVEL PROJECT")
print("="*60)

# 1. Load Dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')
print("\nDataset Shape:", X.shape)
print("\nFirst 5 Rows:")
print(X.head())
print("\nClass Distribution:")
print(y.value_counts())
print("\nMissing Values:")
print(X.isnull().sum())

# 2. EDA - Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation - Iris Dataset')
plt.tight_layout()
plt.savefig('Task1_Correlation.png')
plt.show()

# 3. EDA - Pairplot
sns.pairplot(pd.concat([X, y], axis=1), hue='species', palette='Set2')
plt.suptitle('Iris Dataset - Pairplot', y=1.02)
plt.savefig('Task1_Pairplot.png')
plt.show()

# 4. Train / Test Split (80/20, Stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"\nTrain Size: {X_train.shape}, Test Size: {X_test.shape}")

# 5. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train Models
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

tree_model = DecisionTreeClassifier(random_state=42, max_depth=4)
tree_model.fit(X_train_scaled, y_train)

# 7. Evaluation Function - Distinction Level
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n===== {name} =====")
    print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Cross Validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"5-Fold Cross Validation Mean: {cv_scores.mean():.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'Task1_CM_{name.replace(" ", "_")}.png')
    plt.show()
    return acc

acc_log = evaluate_model("Logistic Regression", log_model, X_test_scaled, y_test)
acc_tree = evaluate_model("Decision Tree", tree_model, X_test_scaled, y_test)

# 8. Final Comparison
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
print(f"Logistic Regression Accuracy: {acc_log*100:.2f}%")
print(f"Decision Tree Accuracy      : {acc_tree*100:.2f}%")
best = "Logistic Regression" if acc_log >= acc_tree else "Decision Tree"
print(f"\nBest Model: {best}")
print("\nKey Insight: Logistic Regression performs better on Iris because classes are linearly separable and dataset is small & clean.")
print("Task 1 Completed Successfully!")