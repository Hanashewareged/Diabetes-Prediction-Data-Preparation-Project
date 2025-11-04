
import pandas as pd

def save_data(data: pd.DataFrame, filepath: str):
    """
    Save a DataFrame to CSV, Excel, or Parquet.
    """
    try:
        if filepath.endswith(".csv"):
            data.to_csv(filepath, index=False)
        elif filepath.endswith(".xlsx") or filepath.endswith(".xls"):
            data.to_excel(filepath, index=False)
        elif filepath.endswith(".parquet"):
            data.to_parquet(filepath, index=False)
        else:
            raise ValueError("Unsupported file format. Use .csv, .xlsx, or .parquet")
        print(f"✅ Cleaned data successfully saved to '{filepath}'")
    except Exception as e:
        print(f"❌ Error saving cleaned data: {e}")
