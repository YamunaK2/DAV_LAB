"""
Experiment 8: Multiple Regression Analysis

AIM:
To perform multiple regression analysis on the UCI Diabetes and Pima Indians Diabetes datasets to
predict BMI based on multiple independent variables.

SOFTWARE REQUIREMENTS:
- Python: Version 3.13.2
- Jupyter Notebook: Version 7.3.2
- Packages: pandas, numpy, scikit-learn
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def run_experiment_8():
    base_dir=os.path.dirname(os.path.abspath(__file__))
    uci=pd.read_csv(os.path.join(base_dir,'uci_diabetes.csv'))
    pima=pd.read_csv(os.path.join(base_dir,'pima_diabetes.csv'))
    features=['Glucose','BloodPressure','Age']; target='BMI'
    def analyze(df,name):
        X=df[features]; y=df[target]
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
        model=LinearRegression().fit(X_train,y_train)
        pred=model.predict(X_test)
        r2=r2_score(y_test,pred)
        print(f"{name} - Multiple Regression R2 Score: {r2:.4f}")
    analyze(uci,'UCI Diabetes Dataset')
    analyze(pima,'Pima Indians Diabetes Dataset')

if __name__=='__main__':
    run_experiment_8()
