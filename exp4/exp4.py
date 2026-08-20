"""
Experiment 4: Reading Data from Text Files, Excel, and the Web

AIM:
To read and process data from various sources, including text files, Excel spreadsheets, and web-based
data, using Python's Pandas library.
"""

import os
import pandas as pd

def run_experiment_4():
    # Determine base directory of script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    csv_file = os.path.join(base_dir, 'Google_data (2b.c1).csv')
    excel_file = os.path.join(base_dir, 'data (2c2).xlsx')
    web_url = 'https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv'
    
    # 1. Read data from CSV file
    text_df = pd.read_csv(csv_file, encoding='utf-8')
    
    # 2. Read data from Excel file
    excel_df = pd.read_excel(excel_file, sheet_name='Sheet1')
    
    # 3. Read data from web-based source
    web_df = pd.read_csv(web_url)
    
    # 4. Display the first few rows of the datasets
    print("--- Text (CSV) Data Head ---")
    print(text_df.head())
    print("\n--- Excel Data Head ---")
    print(excel_df.head())
    print("\n--- Web Data Head ---")
    print(web_df.head())
    
    # 5. Handle missing values if present
    text_df.ffill(inplace=True)
    excel_df.bfill(inplace=True)
    web_df.dropna(inplace=True)
    
    # 6. Save processed data into new file formats
    out_csv = os.path.join(base_dir, 'processed_text.csv')
    out_excel = os.path.join(base_dir, 'processed_excel.xlsx')
    text_df.to_csv(out_csv, index=False)
    excel_df.to_excel(out_excel, index=False)
    print("\nSuccessfully saved processed data to 'processed_text.csv' and 'processed_excel.xlsx'")

if __name__ == "__main__":
    run_experiment_4()
