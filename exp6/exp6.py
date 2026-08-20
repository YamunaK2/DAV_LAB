"""
Experiment 6: Statistical Analysis Using Diabetes Datasets - Univariate Analysis

AIM:
To analyze the Diabetes dataset from UCI and the Pima Indians Diabetes dataset using univariate
statistical methods, including Frequency, Mean, Median, Mode, Variance, Standard Deviation,
Skewness, and Kurtosis.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def run_experiment_6():
    # Determine base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    uci_path = os.path.join(base_dir, "uci_diabetes.csv")
    pima_path = os.path.join(base_dir, "pima_diabetes.csv")
    
    # 1. Import Datasets
    uci_diabetes = pd.read_csv(uci_path)
    pima_diabetes = pd.read_csv(pima_path)
    
    # 2. Display Dataset Samples
    print("UCI Diabetes Dataset Sample:")
    print(uci_diabetes.head())
    print("\nPima Indians Diabetes Dataset Sample:")
    print(pima_diabetes.head())
    
    # 3. Define Relevant Numerical Columns
    numerical_columns = [
        "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
        "DiabetesPedigreeFunction", "Age"
    ]
    
    # 4. Univariate Analysis Function
    def univariate_analysis(df, columns):
        stats = {}
        for col in columns:
            stats[col] = {
                "Mean": np.mean(df[col]),
                "Median": np.median(df[col]),
                "Mode": df[col].mode()[0],
                "Variance": np.var(df[col], ddof=1),
                "Standard Deviation": np.std(df[col], ddof=1),
                "Skewness": skew(df[col]),
                "Kurtosis": kurtosis(df[col])
            }
        return pd.DataFrame(stats).T

    # 5. Perform Univariate Analysis
    uci_stats = univariate_analysis(uci_diabetes, numerical_columns)
    pima_stats = univariate_analysis(pima_diabetes, numerical_columns)
    
    # 6. Display Results
    print("\nUCI Diabetes Dataset Statistics:")
    print(uci_stats)
    print("\nPima Indians Diabetes Dataset Statistics:")
    print(pima_stats)

if __name__ == "__main__":
    run_experiment_6()
