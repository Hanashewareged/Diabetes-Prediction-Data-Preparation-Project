import pandas as pd
class DataExploration:
    def load_data(self, filepath):
        if filepath.endswith(".csv"):
            return pd.read_csv(filepath)
        elif filepath.endswith(".parquet"):
            return pd.read_parquet(filepath)
        elif filepath.endswith(".xlsx") or filepath.endswith(".xls"):
            return pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format. Use CSV, Parquet, XLSX, or XLS.")

    def explore_data(self, data):
        print("\n--- Dataset Overview ---")
        print(data.head())
        print("\n--- Info ---")
        data.info()
        print("\n--- Summary Statistics ---")
        print(data.describe())

    def identify_issues(self, data):
        cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in cols_with_zero:
            if col in data.columns:
                print(f"{col}: {(data[col] == 0).sum()} zeros found.")
