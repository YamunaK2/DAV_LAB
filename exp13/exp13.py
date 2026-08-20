"""
Experiment 13: ANOVA on Diabetes Datasets

AIM:
To perform One-Way ANOVA on the UCI Diabetes and Pima Indians Diabetes datasets.
"""
import os
import pandas as pd
from scipy.stats import f_oneway

def run_experiment_13():
    base=os.path.dirname(os.path.abspath(__file__))
    uci=pd.read_csv(os.path.join(base,'uci_diabetes.csv'))
    pima=pd.read_csv(os.path.join(base,'pima_diabetes.csv'))
    cols=['Glucose','BloodPressure','BMI']
    results={}
    for col in cols:
        f,p=f_oneway(uci[col],pima[col])
        results[col]={'F-statistic':f,'P-value':p}
    result_df=pd.DataFrame(results).T
    print('\nANOVA Results:\n',result_df)
    print('\nDecision at alpha=0.05:')
    for col,row in result_df.iterrows():
        print(f"{col}: {'Significant difference' if row['P-value']<0.05 else 'No significant difference'}")

if __name__=='__main__':
    run_experiment_13()
