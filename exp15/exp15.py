"""
Experiment 15: Building and Validating Logistic Models

AIM:
To build and validate Logistic Regression Models for predicting diabetes presence using the UCI and Pima Indians Diabetes datasets.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def run_experiment_15():
    base=os.path.dirname(os.path.abspath(__file__))
    uci=pd.read_csv(os.path.join(base,'uci_diabetes.csv'))
    pima=pd.read_csv(os.path.join(base,'pima_diabetes.csv'))
    features=['Glucose','BloodPressure','BMI']; target='Outcome'
    matrices=[]
    for name,df in [('UCI Diabetes Dataset',uci),('Pima Indians Diabetes Dataset',pima)]:
        X_train,X_test,y_train,y_test=train_test_split(df[features],df[target],test_size=.2,random_state=42)
        model=LogisticRegression(max_iter=1000).fit(X_train,y_train)
        pred=model.predict(X_test)
        print(f'{name} - Logistic Regression Results:')
        print(f'Accuracy: {accuracy_score(y_test,pred):.4f}, Precision: {precision_score(y_test,pred,zero_division=0):.4f}, Recall: {recall_score(y_test,pred,zero_division=0):.4f}, F1 Score: {f1_score(y_test,pred,zero_division=0):.4f}')
        matrices.append((name,confusion_matrix(y_test,pred)))
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,(name,cm) in zip(axes,matrices):
        sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=ax); ax.set_title(f'{name} - Confusion Matrix'); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.tight_layout(); plt.savefig(os.path.join(base,'logistic_confusion_matrices.png'),dpi=150,bbox_inches='tight'); plt.show(); plt.close()

if __name__=='__main__':
    run_experiment_15()
