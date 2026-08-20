"""
Experiment 10: Data Visualization - Normal Curves on UCI Diabetes Dataset

AIM:
To visualize the distribution of key numerical attributes in the UCI Diabetes dataset using normal curves.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def run_experiment_10():
    base=os.path.dirname(os.path.abspath(__file__))
    df=pd.read_csv(os.path.join(base,'uci_diabetes.csv'))
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,col in zip(axes,['Glucose','BMI']):
        ax.hist(df[col], bins=15, density=True, alpha=0.6)
        df[col].plot(kind='kde', ax=ax)
        x=np.linspace(df[col].min(),df[col].max(),100)
        ax.plot(x,norm.pdf(x,df[col].mean(),df[col].std()),label='Normal Curve')
        ax.set_title(f'Normal Curve - {col}')
        ax.legend()
    plt.tight_layout()
    path=os.path.join(base,'normal_curves.png'); plt.savefig(path,dpi=150); print(f"Saved plot to '{os.path.basename(path)}'")
    plt.show(); plt.close()

if __name__=='__main__':
    run_experiment_10()
