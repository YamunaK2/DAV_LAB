"""
Experiment 9: Comparison of Analysis Results Between UCI and Pima Diabetes Datasets

AIM:
To compare statistical analysis and model performance between the UCI Diabetes and Pima Indians Diabetes datasets.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score

def run_experiment_9():
    base=os.path.dirname(os.path.abspath(__file__))
    uci=pd.read_csv(os.path.join(base,'uci_diabetes.csv'))
    pima=pd.read_csv(os.path.join(base,'pima_diabetes.csv'))
    features=['Glucose','BloodPressure','Age']; target='BMI'
    log_features=['Glucose','BloodPressure','BMI','Age']; log_target='Outcome'
    for name,df in [('UCI Diabetes Dataset',uci),('Pima Indians Diabetes Dataset',pima)]:
        print(f'\n{name} Univariate Summary:')
        print(df[['Glucose','BloodPressure','BMI','Age']].describe().loc[['mean','50%','std']])
        Xtr,Xte,ytr,yte=train_test_split(df[features],df[target],test_size=.2,random_state=42)
        lm=LinearRegression().fit(Xtr,ytr)
        lr2=r2_score(yte,lm.predict(Xte))
        Xtr,Xte,ytr,yte=train_test_split(df[log_features],df[log_target],test_size=.2,random_state=42)
        log=LogisticRegression(max_iter=1000).fit(Xtr,ytr)
        acc=accuracy_score(yte,log.predict(Xte))
        print(f'Linear Regression R2: {lr2:.4f}')
        print(f'Logistic Regression Accuracy: {acc*100:.2f}%')
    print('\nComparison completed based on the same train/test procedure and selected features.')

if __name__=='__main__':
    run_experiment_9()
