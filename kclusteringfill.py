import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.cluster import KMeans

class KClusteringFill(BaseEstimator,TransformerMixin):
    """
    Perform K-Means clustering on data with missing values.

    Args:
      X: An [n_samples, n_features] array of data to cluster.
      n_clusters: Number of clusters to form.
      max_iter: Maximum number of EM iterations to perform.
      select_columns : Columns to perform Transformation

    Returns:
      labels: An [n_samples] vector of integer labels.
      df: Copy of X with the missing values filled in.
    """
    
    def __init__(self,n_clusters=5,select_columns=None,max_iter=10):
        self.n_clusters=n_clusters
        self.max_iter=max_iter
        self.select_column=select_column
    def fit(self,X,y=None):
        self.mu=np.nanmean(X[self.select_column], 0, keepdims=1)
        self.missing = ~np.isfinite(X[self.select_column])
        self.X_hat = np.where(self.missing, self.mu, X[self.select_column])
        return self
    def transform(self,X):
        df=X.copy()
        if self.X_hat.ndim == 1:
            self.X_hat=self.X_hat.reshape(-1,1)
        for i in range(self.max_iter):
            if i > 0:
                # initialize KMeans with the previous set of centroids. this is much
                # faster and makes it easier to check convergence (since labels
                # won't be permuted on every iteration), but might be more prone to
                # getting stuck in local minima.
                cls = KMeans(self.n_clusters, init=prev_centroids)
            else:
                # do multiple random initializations in parallel
                cls = KMeans(self.n_clusters, n_jobs=-1)

            # perform clustering on the filled-in data
            labels = cls.fit_predict(self.X_hat)
            centroids = cls.cluster_centers_

            # fill in the missing values based on their cluster centroids
            self.X_hat[self.missing] = centroids[labels][self.missing]

            # when the labels have stopped changing then we have converged
            if i > 0 and np.all(labels == prev_labels):
                print('::::::::::convergence::::::::::')
                break

            prev_labels = labels
            prev_centroids = cls.cluster_centers_
            df[self.select_column]=self.X_hat
        return df,labels
