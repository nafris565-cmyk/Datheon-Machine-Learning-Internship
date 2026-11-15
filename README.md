# Datheon Machine Learning & Data Science Internship

## Author
**J. Mohamed Nafris**
Machine Learning & Data Science Intern | Datheon

## Overview
This repository contains all 4 tasks completed as part of the Datheon Machine Learning & Data Science Internship (4 Weeks).

All projects are implemented in Python using Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, XGBoost and SMOTE.

## Project Structure
- Task-1-Iris-Classification.py
- Task-2-Housing-Price-Prediction.py
- Task-3-Customer-Churn-Prediction.py
- Task-4-NLP-Sentiment-Classification.py
- README.md

## Task 1: Iris Flower Classification
**Objective:** Classify Iris flowers into 3 species (Setosa, Versicolor, Virginica)

**Models:** Logistic Regression vs Decision Tree

**Dataset:** Iris Dataset (150 samples, 4 features)

**Results:**
| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 93.33% | 0.93 | 0.93 | 0.93 |
| Decision Tree | 93.33% | 0.93 | 0.93 | 0.93 |

**Key Insight:** Petal Length and Petal Width are the most important features. Setosa is perfectly separable.

## Task 2: California Housing Price Prediction
**Objective:** Predict median house value in California

**Models:** OLS vs Ridge vs Lasso

**Dataset:** California Housing Dataset (20,640 samples, 8 features)

**Results:**
| Model | R2 Score | RMSE | MAE |
| :--- | :---: | :---: | :---: |
| OLS Linear Regression | 0.5757 | 0.728 | 0.533 |
| Ridge Regression | 0.5757 | 0.728 | 0.533 |
| Lasso Regression | 0.5756 | 0.728 | 0.533 |

**Key Insight:** MedInc (Median Income) is the strongest predictor of house price. Ridge handles multicollinearity best.

## Task 3: Customer Churn Prediction
**Objective:** Predict whether a telecom customer will churn

**Models:** Random Forest vs XGBoost

**Dataset:** Telco Customer Churn Dataset (7,043 customers)

**Challenge:** Imbalanced Data - Only 26.5% churned. Solved using SMOTE.

**Results:**
| Model | ROC-AUC | Accuracy |
| :--- | :---: | :---: |
| Random Forest | 81.80% | 78.0% |
| XGBoost | 81.10% | 77.5% |

**Best Model:** Random Forest

**Key Insight:** Month-to-Month Contract, No Online Security, High Monthly Charges customers churn most. Longer tenure means lower churn.

## Task 4: NLP Sentiment Classification
**Objective:** Classify Movie Reviews as Positive or Negative

**Models:** Naive Bayes + TF-IDF vs Logistic Regression + TF-IDF

**Dataset:** 【entity-IMDB¦canonical_name=IMDB】 Movie Reviews

**Results:**
| Model | Accuracy |
| :--- | :---: |
| Naive Bayes + TF-IDF | 86.50% |
| Logistic Regression + TF-IDF | 88.20% |

**Best Model:** Logistic Regression + TF-IDF

**Key Insight:** TF-IDF captures words like great, amazing, terrible, boring. Bigrams help capture phrases like not good.

**Note:** In local run, if 【entity-IMDB¦canonical_name=IMDB】 URL fails, code automatically uses a balanced sample dataset for demonstration.

## Technologies Used
- Language: Python 3.11
- Libraries: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
- Advanced: XGBoost, Imbalanced-Learn (SMOTE)
- Tools: VS Code, Git, 【entity-GitHub¦canonical_name=GitHub】

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost imbalanced-learn
python Task-1-Iris-Classification.py
python Task-2-Housing-Price-Prediction.py
python Task-3-Customer-Churn-Prediction.py
python Task-4-NLP-Sentiment-Classification.py
