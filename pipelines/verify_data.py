import pandas as pd
import os

DATA_PATH = "data/diabetic_data.csv"

def verify_data():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Data loaded successfully.")
        print(f"Shape: {df.shape}")
        print("Columns:", df.columns.tolist()[:10], "...")
        print("\nMissing values:\n", df.isnull().sum().sum())
        print("\nFirst 5 rows:\n", df.head())
        
        # Check target column 'readmitted'
        if 'readmitted' in df.columns:
            print("\nTarget distribution (readmitted):")
            print(df['readmitted'].value_counts())
        else:
            print("Warning: 'readmitted' column not found!")

    except Exception as e:
        print(f"Failed to load data: {e}")

if __name__ == "__main__":
    verify_data()
