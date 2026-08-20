"""
Experiment 11: Hypothesis Testing - Z-Test on UCI Diabetes Dataset

AIM:
To perform a Z-test on the UCI Diabetes dataset to determine whether the mean Glucose level differs from 100.
"""
import os
import pandas as pd
from statsmodels.stats.weightstats import ztest

def run_experiment_11():
    base=os.path.dirname(os.path.abspath(__file__))
    df=pd.read_csv(os.path.join(base,'uci_diabetes.csv'))
    z_stat,p_value=ztest(df['Glucose'],value=100)
    print(f'Z-Statistic: {z_stat:.4f}')
    print(f'P-Value: {p_value:.4f}')
    alpha=.05
    if p_value<alpha:
        print('Reject the null hypothesis: The mean Glucose level is significantly different from 100.')
    else:
        print('Fail to reject the null hypothesis: No significant difference in mean Glucose level.')

if __name__=='__main__':
    run_experiment_11()
