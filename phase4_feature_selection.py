import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA

class FeatureSelection:
    # Correlation Matrix
    def correlation_matrix(self, data):
        corr = data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()
        return corr

    # Select K Best
    def select_k_best_features(self, X, y, k=5):
        if y.dtype != int:
            y = y.astype(int)
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()]
        print(f"Top {k} features: {list(selected_features)}")
        return selected_features

    # PCA
    def perform_pca(self, X, n_components=None):
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = explained_variance.cumsum()

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(explained_variance) + 1),
                 cumulative_variance, marker='o', linestyle='--')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.grid(True)
        plt.show()

        print("Explained variance:", explained_variance)
        return X_pca, explained_variance, cumulative_variance
