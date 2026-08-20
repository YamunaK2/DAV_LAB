"""
Experiment 14: Building and Validating Linear Models

AIM:
To build and validate Linear Regression Models using the UCI and Pima Indians Diabetes datasets.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def run_experiment_14():
    base=os.path.dirname(os.path.abspath(__file__))
    uci=pd.read_csv(os.path.join(base,'uci_diabetes.csv'))
    pima=pd.read_csv(os.path.join(base,'pima_diabetes.csv'))
    features=['Glucose','BloodPressure','BMI']; target='Age'
    for name,df in [('UCI Diabetes Dataset',uci),('Pima Indians Diabetes Dataset',pima)]:
        X_train,X_test,y_train,y_test=train_test_split(df[features],df[target],test_size=.2,random_state=42)
        model=LinearRegression().fit(X_train,y_train)
        pred=model.predict(X_test)
        r2=r2_score(y_test,pred); mse=mean_squared_error(y_test,pred); mae=mean_absolute_error(y_test,pred)
        print(f'{name} - Linear Regression Results:')
        print(f'R2 Score: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}\n')
        plt.figure(figsize=(7,5)); plt.scatter(y_test,pred,alpha=.7); plt.xlabel('Actual Age'); plt.ylabel('Predicted Age'); plt.title(f'{name} - Actual vs Predicted'); plt.grid(True,alpha=.3)
        filename='uci_linear_model_validation.png' if name.startswith('UCI') else 'pima_linear_model_validation.png'
        plt.savefig(os.path.join(base,filename),dpi=150,bbox_inches='tight'); plt.close()

if __name__=='__main__':
    run_experiment_14()
