import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

class FeatureEngineering:
    # Feature creation
    def feature_engineering(self, data):
        # Age groups
        bins_age = [0, 34, 60, np.inf]
        labels_age = ['young', 'middle-aged', 'senior']
        data['AgeGroup'] = pd.cut(data['Age'], bins=bins_age, labels=labels_age)

        # BMI categories
        bins_bmi = [0, 18.5, 24.9, 29.9, np.inf]
        labels_bmi = ['underweight', 'normal', 'overweight', 'obese']
        data['BMICategory'] = pd.cut(data['BMI'], bins_bmi, labels=labels_bmi)

        # Glucose categories
        bins_glucose = [0, 139, 199, np.inf]
        labels_glucose = ['normal', 'prediabetes', 'diabetes']
        data['GlucoseCategory'] = pd.cut(data['Glucose'], bins=bins_glucose, labels=labels_glucose)
        return data

    # Encoding
    def encode_features(self, data):
        le = LabelEncoder()
        for col in ['AgeGroup', 'BMICategory', 'GlucoseCategory']:
            if col in data.columns:
                data[col] = le.fit_transform(data[col])
        return data

    # Scaling
    def scale_features(self, data, method='standard'):
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        return data
