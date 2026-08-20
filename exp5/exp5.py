"""
Experiment 5: Exploring Descriptive Analytics Using the Iris Dataset

AIM:
To explore descriptive analytics using the Iris dataset with Python's Pandas and Seaborn libraries.
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run_experiment_5():
    # Determine base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, 'iris_dataset(2d).csv')
    
    # 1. Load dataset
    df = pd.read_csv(dataset_path)
    
    # 2. Display basic information and summary statistics
    print("Basic Information:")
    df.info()
    print("\nSummary Statistics:")
    print(df.describe())
    
    # 3. Perform univariate analysis - species count
    print("\nSpecies Count:")
    print(df['species'].value_counts())
    
    # 4. Visualize data distributions using histograms
    df.hist(figsize=(8, 6), edgecolor='black')
    plt.suptitle('Feature Distributions')
    plt.savefig(os.path.join(base_dir, 'histograms.png'), bbox_inches='tight')
    plt.close()
    print("\nSaved histogram feature distributions to 'histograms.png'")
    
    # 5. Boxplot for Sepal Length
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x='species', y='sepal length (cm)')
    plt.title('Sepal Length Comparison')
    plt.savefig(os.path.join(base_dir, 'sepal_length_boxplot.png'), bbox_inches='tight')
    plt.close()
    print("Saved sepal length boxplot to 'sepal_length_boxplot.png'")
    
    # 6. Pairplot to analyze feature relationships
    pair_plot = sns.pairplot(df, hue='species')
    pair_plot.savefig(os.path.join(base_dir, 'pairplot.png'), bbox_inches='tight')
    plt.close()
    print("Saved feature pairplot to 'pairplot.png'")

if __name__ == "__main__":
    run_experiment_5()
