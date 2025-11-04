import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

class DataBalancing:
    # Class distribution
    def analyze_class_distribution(self, y):
        class_counts = y.value_counts()
        print("Class distribution:\n", class_counts)
        imbalance_ratio = class_counts.min() / class_counts.max()
        print(f"Imbalance ratio: {imbalance_ratio:.2f}")

        plt.figure(figsize=(6, 4))
        class_counts.plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title('Target Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.show()

        return imbalance_ratio

    # Balancing using SMOTE
    def balance_data(self, X, y, method='SMOTE'):
        if method == 'SMOTE':
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            print("After balancing:\n", y_resampled.value_counts())
            return X_resampled, y_resampled
        else:
            raise ValueError("Only SMOTE is supported currently.")
