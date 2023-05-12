import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt


class SpectralClustering:
    
    def __init__(self, k):
        self.k = k
        self.clustering = None

    def fit(self, X):
        sigma = 0.1
        W = np.exp(-np.sqrt(cdist(X, X, 'sqeuclidean')) ** 2 / (2 * sigma ** 2))
        D = np.diag(np.sum(W, axis=1))
        L = D - W

        eigenvals, eigenvecs = np.linalg.eigh(L)
        eigenvecs = eigenvecs[:, np.argsort(eigenvals)[1:self.k+1]]

        self.clustering = KMeans(n_clusters=self.k)
        self.clustering.fit(eigenvecs)
    
    def predict(self):
        return self.clustering.labels_


# Example
X_circles1, y_circles1 = make_circles(n_samples=1000, noise=0.03, factor=0.2, random_state=0)
X_circles2, y_circles2 = make_circles(n_samples=1000, noise=0.03, factor=0.6, random_state=0)
X_circles = np.append(X_circles1, X_circles2, axis=0)

SC = SpectralClustering(k=3)
SC.fit(X_circles)
clusters = SC.predict()

plt.scatter(X_circles[:, 0], X_circles[:, 1], c=clusters)
plt.show()



