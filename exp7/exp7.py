"""
Experiment 7: Bivariate Analysis - Linear and Logistic Regression Modeling

AIM:
To perform Bivariate Analysis on the UCI Diabetes Dataset and Pima Indians Diabetes Dataset using
Linear Regression and Logistic Regression.

SOFTWARE REQUIREMENTS:
- Python: Version 3.13.2
- Jupyter Notebook: Version 7.3.2
- Packages: pandas, numpy, seaborn, matplotlib, scikit-learn
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score

def run_experiment_7():
    # Determine base directory for data files
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    uci_path = os.path.join(base_dir, "uci_diabetes.csv")
    if not os.path.exists(uci_path):
        uci_path = os.path.join(base_dir, "uci_diabetes (3).csv")
        
    pima_path = os.path.join(base_dir, "pima_diabetes.csv")
    if not os.path.exists(pima_path):
        pima_path = os.path.join(base_dir, "pima_diabetes (3).csv")

    # 1. Load the Datasets
    uci_diabetes = pd.read_csv(uci_path)
    pima_diabetes = pd.read_csv(pima_path)

    # 2. Display first few rows
    print("UCI Diabetes Dataset Sample:")
    print(uci_diabetes.head())
    print("\nPima Indians Diabetes Dataset Sample:")
    print(pima_diabetes.head())

    # 3. Perform Linear Regression (Glucose vs. BMI)
    def linear_regression_analysis(df, x_column, y_column, dataset_name, save_filename=None):
        X = df[[x_column]] # Independent variable
        Y = df[y_column]   # Dependent variable

        model = LinearRegression()
        model.fit(X, Y)
        Y_pred = model.predict(X)

        r2 = r2_score(Y, Y_pred)

        print(f"\nLinear Regression ({dataset_name} - Predicting {y_column} using {x_column}):")
        print(f"R2 Score: {r2:.4f}")

        # Plot
        plt.figure(figsize=(8, 5))
        plt.scatter(X, Y, color='blue', alpha=0.7, label='Actual Data')
        plt.plot(X, Y_pred, color='red', linewidth=2, label='Regression Line')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(f"Linear Regression ({dataset_name}): {x_column} vs. {y_column}")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        if save_filename:
            save_path = os.path.join(base_dir, save_filename)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved plot to '{save_filename}'")
        
        plt.close()

    # Apply Linear Regression on both datasets
    linear_regression_analysis(uci_diabetes, "Glucose", "BMI", "UCI Diabetes", "uci_linear_regression.png")
    linear_regression_analysis(pima_diabetes, "Glucose", "BMI", "Pima Indians Diabetes", "pima_linear_regression.png")

    # 4. Perform Logistic Regression (Predicting Diabetes Outcome)
    def logistic_regression_analysis(df, features, target, dataset_name):
        X = df[features]
        Y = df[target]
        
        # Splitting dataset
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(Y_test, Y_pred)
        print(f"\nLogistic Regression ({dataset_name} - Predicting {target} using {features}):")
        print(f"Accuracy Score: {accuracy:.4f}")

    # Select features and target
    features = ["Glucose", "BloodPressure", "BMI", "Age"]
    target = "Outcome"

    # Apply Logistic Regression on both datasets
    logistic_regression_analysis(uci_diabetes, features, target, "UCI Diabetes")
    logistic_regression_analysis(pima_diabetes, features, target, "Pima Indians Diabetes")

if __name__ == "__main__":
    run_experiment_7()
