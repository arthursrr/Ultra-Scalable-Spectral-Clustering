import time
import numpy as np

from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.neighbors import radius_neighbors_graph
from scipy.sparse import csgraph
from scipy.linalg import eig

class SC():
    def __init__(self):
        return
    
    def predict(self, Features, n_clusters):
        #Stage 1
        A = radius_neighbors_graph(Features,0.4,mode='distance', metric='minkowski', p=2, metric_params=None, include_self=False).toarray()

        #Stage 2
        L = csgraph.laplacian(A, normed=False)
        B = np.sum(A,axis=0)

        #Stage 3
        _, eigvec = eig(L)
        return KMeans(n_clusters=n_clusters, max_iter=100).fit(eigvec[:, :n_clusters-1]).labels_