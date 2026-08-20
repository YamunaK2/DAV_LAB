"""
Experiment 12: Performing T-Test on Diabetes Datasets

AIM:
To perform an independent T-test on the UCI Diabetes and Pima Indians Diabetes datasets to compare means.
"""
import os
import pandas as pd
from scipy.stats import ttest_ind

def run_experiment_12():
    base=os.path.dirname(os.path.abspath(__file__))
    uci=pd.read_csv(os.path.join(base,'uci_diabetes.csv'))
    pima=pd.read_csv(os.path.join(base,'pima_diabetes.csv'))
    cols=['Glucose','BloodPressure','BMI']
    results={}
    for col in cols:
        t,p=ttest_ind(uci[col],pima[col],equal_var=False)
        results[col]={'T-statistic':t,'P-value':p}
    print('\nT-test Results:\n',pd.DataFrame(results).T)

if __name__=='__main__':
    run_experiment_12()
