"""
Experiment 3: Data Analysis and Manipulation using Pandas
"""

import os
import pandas as pd

def run_experiment_3():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'data.csv')
    
    # Load dataset into a DataFrame
    df = pd.read_csv(csv_path)

    # Display first and last few rows
    print("--- First 5 rows ---")
    print(df.head())
    print("\n--- Last 5 rows ---")
    print(df.tail())

    # Check data types and general info
    print("\n--- DataFrame Info ---")
    df.info()

    # Summary statistics
    print("\n--- Summary Statistics ---")
    print(df.describe())

    # Handle missing values
    df.fillna(df.select_dtypes(include='number').mean(), inplace=True)

    # Create a new column
    df['new_column'] = df['existing_column'] * 2

    # Create a Series and perform operations
    series = df['existing_column']
    print("\n--- Series Addition ---")
    print(series + 10)

    # Filter rows based on conditions
    filtered_df = df[(df['existing_column'] > 50) & (df['another_column'] < 100)]
    print("\n--- Filtered DataFrame ---")
    print(filtered_df)

    # Grouping and aggregation
    grouped = df.groupby('category_column')['numeric_column'].mean()
    print("\n--- Grouped Mean ---")
    print(grouped)

    # Sorting
    df_sorted = df.sort_values(by='numeric_column', ascending=False)
    print("\n--- Sorted DataFrame ---")
    print(df_sorted)

    # Boolean masking
    masked_df = df[df['numeric_column'] > df['numeric_column'].median()]
    print("\n--- Masked DataFrame ---")
    print(masked_df)

    # Remove duplicates and drop missing values
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # Create a new DataFrame with selected columns
    subset_df = df[['column1', 'column2']]

    # Save the new DataFrame to a CSV file
    out_path = os.path.join(base_dir, 'filtered_data.csv')
    subset_df.to_csv(out_path, index=False)
    print("\nSaved subset DataFrame to 'filtered_data.csv'")

    # Compute summary statistics
    print("\n--- Final Aggregations ---")
    print("Total sum:", df['numeric_column'].sum())
    print("Mean:", df['numeric_column'].mean())
    print("Standard Deviation:", df['numeric_column'].std())

if __name__ == "__main__":
    run_experiment_3()
