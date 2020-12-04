import time

import numpy as np
from numpy import matlib

from sklearn.cluster import KMeans
import scipy

class USPEC:
	def __init__(self):
		return

	def __getRepresentivesByRandomSelection(self, data, pSize):
		N = data.shape[0]
		if pSize > N:
			pSize = N
		
		selectIdx = np.random.permutation(np.arange(N))[:pSize]
		randSelect = []
		for i in selectIdx:
			randSelect.append(data[i,:])

		return np.array(randSelect)

	def __getRepresentativesByHybridSelection(self, data, pSize, cntTimes=10):
		N = data.shape[0]
		bigPSize = cntTimes * pSize
		
		if pSize > N: 
			pSize = N
		if bigPSize > N: 
			bigPSize = N

		#random selection
		np.random.seed(int(time.time()))
		RpBigPdata = self.__getRepresentivesByRandomSelection(data, bigPSize)
		
		#KNN selection
		RpData = KMeans(n_clusters=pSize, max_iter=cntTimes).fit(RpBigPdata)
		
		return RpData
	
	def __sparse(self, i, j, v, m, n):
		'''
		Create and compressing a matrix that have many zeros
		Parameters:
			i: 1-D array representing the index 1 values 
				Size n1
			j: 1-D array representing the index 2 values 
				Size n1
			v: 1-D array representing the values 
				Size n1
			m: integer representing x size of the matrix >= n1
			n: integer representing y size of the matrix >= n1
		Returns:
			s: 2-D array
				Matrix full of zeros excepting values v at indexes i, j
		'''
		return scipy.sparse.csr_matrix((v, (i, j)), shape=(m, n))

	def __dist(self, A, B, C):
		for i in range(A.shape[0]):
			for j in range(A.shape[1]):
				A[i,j] = np.linalg.norm(B[i]-C[j])

	def predict(self, Features, N_representations, N_clusters, Knn=10, cntTimes=10):
		#Stage 1
		RpData = self.__getRepresentativesByHybridSelection(Features, N_representations)
		
		#Stage 2
		N = Features.shape[0]

		cntRepCls = int(np.floor(np.sqrt(RpData.cluster_centers_.shape[0])))
		AprData = KMeans(n_clusters=cntRepCls, max_iter=600).fit(RpData.cluster_centers_)

		cntRepCls = AprData.cluster_centers_.shape[0]

		centerDist = np.zeros((N, cntRepCls))
		
		self.__dist(centerDist, Features, AprData.cluster_centers_)
		
		minCenterIdxs = np.argmin(centerDist, axis=1)

		del centerDist

		nearestRepInRpFeaIdx = np.zeros((N), dtype=np.int32)
		for i in range(cntRepCls):
			originalIdxs = np.where(minCenterIdxs == i)[0]
			aprDataIdxs = np.where(AprData.labels_ == i)[0]

			originalTemp = np.take(Features, originalIdxs).ravel()
			aprDataTemp = np.take(RpData.cluster_centers_, aprDataIdxs).ravel()

			temp = np.zeros((originalTemp.shape[0], aprDataTemp.shape[0]))

			self.__dist(temp, originalTemp, aprDataTemp)

			tempMin = np.argmin(temp, axis=1)

			for j in range(temp.shape[0]):
				nearestRepInRpFeaIdx[originalIdxs[j]] = int(aprDataIdxs[tempMin[j]])
		
		del AprData

		neighSize = 10*Knn
		RpFeaW = np.zeros((N_representations, N_representations))

		self.__dist(RpFeaW, RpData.cluster_centers_, RpData.cluster_centers_)

		RpFeaKnnIdx = np.argsort(RpFeaW, axis=1, kind='mergesort')
		
		del RpFeaW

		RpFeaKnnIdx = RpFeaKnnIdx[:,:int(neighSize)]
		RpFeaKnnDist = np.zeros((N, RpFeaKnnIdx.shape[1]));

		for i in range(N_representations):
			originalIdxs = np.where(nearestRepInRpFeaIdx == i)[0]
			originalTemp = np.take(Features, originalIdxs).ravel()
			RpFeatTemp = np.take(RpData.cluster_centers_, RpFeaKnnIdx[i]).ravel()
			
			temp = np.zeros((originalTemp.shape[0], RpFeatTemp.shape[0]))

			self.__dist(temp, originalTemp, RpFeatTemp)
			
			for j in range(temp.shape[0]):
				RpFeaKnnDist[originalIdxs[j], :] = temp[j]
		
		del RpData
		
		RpFeaKnnIdxFull = RpFeaKnnIdx[nearestRepInRpFeaIdx,:]

		del RpFeaKnnIdx, nearestRepInRpFeaIdx

		knnDist = np.zeros((N, Knn));
		knnIdx = np.zeros((N, Knn));

		for i in range(Knn):
			idx = np.argmin(RpFeaKnnDist, axis=1)
			minV = np.amin(RpFeaKnnDist, axis=1)
			temp = ((idx) * N) + np.arange(N).T
			
			knnDist [:, i] = minV
			
			for j in range(len(temp)):
				
				knnIdx[j, i] = RpFeaKnnIdxFull[j, i]
				
				if(temp[j] < N):
					RpFeaKnnDist[j, i] = np.inf
		
		del RpFeaKnnDist, RpFeaKnnIdxFull

		knnMeanDiff = np.mean(knnDist)
		Gsdx = np.exp(-(np.power(knnDist,2))/2*np.power(knnMeanDiff,2))
		Gsdx[Gsdx == 0] = np.finfo(float).eps

		del knnDist

		Gidx = matlib.repmat(np.arange(N), Knn, 1).T

		B = self.__sparse(Gidx.ravel(), knnIdx.ravel(), Gsdx.ravel(), N, N_representations).toarray()
		
		del Gsdx, Gidx, knnIdx

		#Stage 3
		dx = np.sum(B, 1)
		np.where(dx == 0, 1e-10, dx)
		dx = 1/dx
		Dx = np.zeros((N, N))
		np.fill_diagonal(Dx, dx)

		del dx

		Er = B.T @ Dx @ B

		d = np.sum(Er.real, 1);
		d = 1/np.sqrt(d)
		D = np.zeros((N_representations, N_representations))
		np.fill_diagonal(D, d)
		Dr = D @ Er @ D
		where_are_NaNs = np.isnan(Dr)
		Dr[where_are_NaNs] = 0
		Dr = (Dr + Dr.T)/2

		aval, avec = np.linalg.eig(Dr)

		del d, Dr, Er

		idx = np.argsort(aval, kind='mergesort')[::-1]
		Ncut_avec = D @ avec[:,idx[:N_clusters]]
		where_are_NaNs = np.isnan(Ncut_avec)
		Ncut_avec[where_are_NaNs] = 0

		res = Dx @ B @ Ncut_avec

		del Dx, idx, D, B, Ncut_avec, aval, avec

		norm = (np.sqrt(np.sum(res*res,1)) + 1e-10)
		for i in range(res.shape[0]):
			res[i,:] = res[i,:]/norm[i]

		return KMeans(n_clusters=N_clusters, max_iter=500).fit(res.real).labels_

'''
TESTE
'''

# import matplotlib as mpl
# import matplotlib.pyplot as plt

# from sklearn.datasets import make_moons

# random_state = 21
# X_mn, y_mn = make_moons(5000, noise=.07, random_state=random_state)
# print(X_mn.shape)
# cmap = 'viridis'
# dot_size=50

# uspec=USPEC()

# print("Comecou")
# labels = uspec.predict(X_mn, 1000, 2)
# print("Terminou")

# fig, ax = plt.subplots(figsize=(9,7))
# ax.set_title('Data after spectral clustering from scratch', fontsize=18, fontweight='demi')
# ax.scatter(X_mn[:, 0], X_mn[:, 1],c=labels,s=dot_size, cmap=cmap)

# plt.show()