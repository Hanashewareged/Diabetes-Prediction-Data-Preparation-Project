import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
from scipy import stats

class DataCleaning:
    def missing_value_analysis(self, data):
        cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        for col in cols_with_zero:
            if col in data.columns:
                data[col] = data[col].replace(0, np.nan)
            else:
                print(f"Warning: Column '{col}' not found in dataset.")
        
        print("\nMissing values per column:")
        print(data.isnull().sum())
        
        # Visualize missing data
        msno.matrix(data)
        plt.show()
        msno.heatmap(data)
        plt.show()
        
        return data

    def impute_missing(self, data, strategy='median'):
        for col in data.select_dtypes(include=np.number).columns:
            if strategy == 'median':
                data[col] = data[col].fillna(data[col].median())
            elif strategy == 'mean':
                data[col] = data[col].fillna(data[col].mean())
            elif strategy == 'mode':
                data[col] = data[col].fillna(data[col].mode()[0])
            else:
                raise ValueError("strategy must be 'median', 'mean', or 'mode'")
        return data

    def treat_outliers(self, data, method='IQR'):
        for col in data.select_dtypes(include=np.number).columns:
            if method == 'IQR':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                data[col] = np.clip(data[col], lower, upper)
            elif method == 'Z-score':
                z_scores = np.abs(stats.zscore(data[col]))
                data = data[z_scores < 3]  # remove rows with z-score > 3
            else:
                raise ValueError("method must be 'IQR' or 'Z-score'")
        return data
