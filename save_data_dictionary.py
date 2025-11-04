import pandas as pd
def save_data_dictionary(data: pd.DataFrame, filepath: str):
    
    try:
        data_dict = pd.DataFrame({
            "Feature": data.columns,
            "Data Type": data.dtypes.astype(str),
            "Unique Values": [data[col].nunique() for col in data.columns],
            "Missing Values": [data[col].isnull().sum() for col in data.columns],
            "Min": [data[col].min() if pd.api.types.is_numeric_dtype(data[col]) else "" for col in data.columns],
            "Max": [data[col].max() if pd.api.types.is_numeric_dtype(data[col]) else "" for col in data.columns],
            "Mean": [data[col].mean() if pd.api.types.is_numeric_dtype(data[col]) else "" for col in data.columns],
            "Median": [data[col].median() if pd.api.types.is_numeric_dtype(data[col]) else "" for col in data.columns],
            "Std": [data[col].std() if pd.api.types.is_numeric_dtype(data[col]) else "" for col in data.columns],
            "Example Value": [data[col].iloc[0] for col in data.columns]
        })

        if filepath.endswith(".csv"):
            data_dict.to_csv(filepath, index=False)
        elif filepath.endswith(".xlsx") or filepath.endswith(".xls"):
            data_dict.to_excel(filepath, index=False)
        else:
            raise ValueError("Unsupported file format. Use .csv or .xlsx")

        print(f"✅ Data dictionary successfully saved to '{filepath}'")

    except Exception as e:
        print(f"❌ Error saving data dictionary: {e}")
