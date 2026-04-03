#!/usr/bin/env python3
"""
Market state discovery via clustering.
"""
import sys
sys.path.append('.')
from elite_feature_engineer import EliteFeatureEngineer
from data_utils import get_data_range
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class MarketStateDiscoverer:
    """
    Discover market states via clustering of feature space.
    """
    
    def __init__(self, symbol: str, n_clusters: int = 5, method: str = 'kmeans'):
        self.symbol = symbol
        self.n_clusters = n_clusters
        self.method = method
        self.engineer = EliteFeatureEngineer(symbol)
        self.scaler = StandardScaler()
        self.pca = None
        self.clusterer = None
        self.cluster_labels = None
        self.features = None
        
    def load_data(self, start_date: pd.Timestamp, end_date: pd.Timestamp):
        """Load feature dataset."""
        result = self.engineer.create_dataset(start_date, end_date)
        if isinstance(result, tuple):
            X, y = result
        else:
            # Fallback: result is DataFrame with target column?
            # Assuming last column is target
            X = result.iloc[:, :-1]
            y = result.iloc[:, -1]
        
        if X.empty:
            return None, None
        
        # Store features and target
        self.features = X
        self.target = y
        
        return X, y
    
    def prepare_for_clustering(self, X: pd.DataFrame, use_pca: bool = True, n_components: int = 10):
        """Prepare features for clustering."""
        # Handle missing values
        X_clean = X.dropna(axis=1, how='any')
        if X_clean.empty:
            raise ValueError("All features have missing values after dropping.")
        
        # Standardize
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Optional PCA for dimensionality reduction
        if use_pca:
            self.pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
            X_reduced = self.pca.fit_transform(X_scaled)
            print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        else:
            X_reduced = X_scaled
        
        return X_reduced, X_clean.columns
    
    def cluster(self, X_reduced: np.ndarray):
        """Apply clustering."""
        if self.method == 'kmeans':
            self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        elif self.method == 'gmm':
            self.clusterer = GaussianMixture(n_components=self.n_clusters, random_state=42)
        elif self.method == 'dbscan':
            self.clusterer = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.cluster_labels = self.clusterer.fit_predict(X_reduced)
        
        # For DBSCAN, -1 is noise; we'll assign noise to its own cluster
        if self.method == 'dbscan':
            noise_mask = self.cluster_labels == -1
            if noise_mask.any():
                # Assign noise to separate cluster
                self.cluster_labels[noise_mask] = self.cluster_labels.max() + 1
                self.n_clusters = len(np.unique(self.cluster_labels))
        
        print(f"Clustering complete. Found {len(np.unique(self.cluster_labels))} clusters.")
        print(f"Cluster sizes: {np.bincount(self.cluster_labels)}")
        
        return self.cluster_labels
    
    def analyze_clusters(self, X: pd.DataFrame, y: pd.Series):
        """Analyze cluster characteristics."""
        if self.cluster_labels is None:
            raise ValueError("Must run clustering first.")
        
        results = []
        
        for cluster_id in range(self.n_clusters):
            mask = self.cluster_labels == cluster_id
            if mask.sum() == 0:
                continue
            
            cluster_X = X[mask]
            cluster_y = y[mask]
            
            # Compute cluster statistics
            stats = {
                'cluster': cluster_id,
                'samples': mask.sum(),
                'percent': mask.sum() / len(X) * 100,
                'up_rate': cluster_y.mean() if len(cluster_y) > 0 else 0,
            }
            
            # Compute mean of key features for interpretation
            key_features = ['vol_24h', 'return_24h', 'rsi', 'ma_distance_50', 
                           'volume_ratio_24h', 'prev_hour_return', 'hour_sin']
            
            for feat in key_features:
                if feat in cluster_X.columns:
                    stats[f'mean_{feat}'] = cluster_X[feat].mean()
            
            results.append(stats)
        
        return pd.DataFrame(results)
    
    def visualize_clusters(self, X_reduced: np.ndarray, save_path: str = 'clusters.png'):
        """Visualize clusters (first 2 PCA components)."""
        if self.cluster_labels is None:
            raise ValueError("Must run clustering first.")
        
        plt.figure(figsize=(12, 10))
        
        # Plot clusters
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                             c=self.cluster_labels, cmap='tab20', alpha=0.6, s=30)
        
        plt.title(f'Market State Clusters - {self.symbol}')
        plt.xlabel('PC1' if self.pca else 'Feature 1')
        plt.ylabel('PC2' if self.pca else 'Feature 2')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Cluster visualization saved to {save_path}")

def main():
    """Run market state discovery for BTC."""
    symbol = 'BTC/USDT'
    start_date, end_date = get_data_range(symbol)
    # Use last 30 days for speed
    quick_start = end_date - pd.Timedelta(days=30)
    
    print(f"Market State Discovery for {symbol}")
    print(f"Period: {quick_start} to {end_date}")
    
    discoverer = MarketStateDiscoverer(symbol, n_clusters=4, method='kmeans')
    
    # Load data
    X, y = discoverer.load_data(quick_start, end_date)
    if X is None:
        print("Failed to load data.")
        return
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} features.")
    
    # Prepare for clustering
    X_reduced, feature_names = discoverer.prepare_for_clustering(X, use_pca=True, n_components=10)
    
    # Cluster
    labels = discoverer.cluster(X_reduced)
    
    # Analyze clusters
    cluster_df = discoverer.analyze_clusters(X, y)
    
    print("\nCluster Analysis:")
    print(cluster_df.to_string())
    
    # Visualize
    discoverer.visualize_clusters(X_reduced, 'market_clusters.png')
    
    # Save cluster assignments
    results_df = pd.DataFrame({
        'timestamp': X.index,
        'cluster': labels,
        'target': y.values
    })
    results_df.to_csv('cluster_assignments.csv', index=False)
    print("\nCluster assignments saved to cluster_assignments.csv")
    
    # Analyze per-cluster predictability
    print("\nPer-cluster predictability:")
    for cluster_id in range(discoverer.n_clusters):
        mask = labels == cluster_id
        if mask.sum() == 0:
            continue
        
        cluster_up_rate = y[mask].mean()
        samples = mask.sum()
        
        # Compute distance from 0.5 (predictability)
        predictability = abs(cluster_up_rate - 0.5)
        
        print(f"Cluster {cluster_id}: {samples:4d} samples, up_rate={cluster_up_rate:.3f}, "
              f"predictability={predictability:.3f}")
    
    # Determine optimal number of clusters via elbow method
    print("\nComputing elbow method for k-means...")
    inertias = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_reduced)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True, alpha=0.3)
    plt.savefig('elbow_method.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Elbow method plot saved to elbow_method.png")

if __name__ == '__main__':
    main()