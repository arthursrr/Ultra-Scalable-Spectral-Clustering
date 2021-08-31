import time

import numpy as np
from numpy import matlib
import tensorflow as tf
from tqdm import tqdm

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
	
	def __dist_tf(self, A, B):
		return tf.sqrt(-2 * 
		tf.matmul(A, B, transpose_b=True) + 
		tf.reduce_sum(tf.square(B), axis=1) +
		tf.expand_dims(tf.reduce_sum(tf.square(A), axis=1), axis=1))
	
	def __sparse_matmul(self, A, B):
		if A.get_shape().as_list()[1] != B.get_shape().as_list()[0]:
			return None
		
		dense_shape = [A.get_shape().as_list()[0], B.get_shape().as_list()[1]]

		Aux_A = tf.concat([tf.cast(A.indices, dtype=A.values.dtype), tf.expand_dims(A.values, axis=1)], axis=1)
		Aux_B = tf.sparse.transpose(B)
		Aux_B = tf.concat([tf.cast(Aux_B.indices, dtype=Aux_B.values.dtype), tf.expand_dims(Aux_B.values, axis=1)], axis=1)
		new_idx = []
		new_values = []

		a = tf.cast(Aux_A[:, 0], dtype=tf.int64)
		b = tf.cast(Aux_B[:, 0], dtype=tf.int64)
		if tf.rank(a) == 1:
			a = tf.expand_dims(a, axis=0)
		if tf.rank(b) == 1:
			b = tf.expand_dims(b, axis=0)

		intersec_r = tf.cast(tf.sets.intersection(a, b).values, dtype=tf.float32)

		for i in tqdm(intersec_r):
			r_idx_A = tf.gather(Aux_A, tf.squeeze(tf.where(Aux_A[:, 0] == i)))
			# if r_idx_A.get_shape()[0] == 0:
			# 	continue
			if tf.rank(r_idx_A) == 1:
				r_idx_A = tf.expand_dims(r_idx_A, axis=0)
					
			for j in intersec_r:
				r_idx_B = tf.gather(Aux_B, tf.squeeze(tf.where(Aux_B[:, 0] == j))) 
				# if r_idx_B.get_shape()[0] == 0:
				# 	continue
				if tf.rank(r_idx_B) == 1:
					r_idx_B = tf.expand_dims(r_idx_B, axis=0)

				a = tf.cast(r_idx_A[:, 1], dtype=tf.int64)
				b = tf.cast(r_idx_B[:, 1], dtype=tf.int64)
				if tf.rank(a) == 1:
					a = tf.expand_dims(a, axis=0)
				if tf.rank(b) == 1:
					b = tf.expand_dims(b, axis=0)

				intersec = tf.cast(tf.sets.intersection(a, b).values, dtype=tf.float32)
				if intersec.get_shape()[0] == 0:
					continue
		
				soma = tf.constant([], dtype=A.values.dtype)
				for k in intersec:
					temp_A = tf.gather(r_idx_A, tf.squeeze(tf.where(r_idx_A[:, 1] == k)))
					temp_B = tf.gather(r_idx_B, tf.squeeze(tf.where(r_idx_B[:, 1] == k)))
					if tf.rank(temp_A) == 1:
						temp_A = tf.expand_dims(temp_A, axis=0)
					if tf.rank(temp_B) == 1:
						temp_B = tf.expand_dims(temp_B, axis=0)
					soma = tf.concat([soma, temp_A[:,-1]*temp_B[:,-1]], axis=0)
				
				soma = tf.reduce_sum(soma)	

				new_idx.append([i, j])
				new_values.append(soma)

		return tf.sparse.SparseTensor(tf.cast(new_idx, dtype=tf.int64), new_values, dense_shape)

	def __sparse_diagonal_matmul(self, A, B):
		if A.get_shape().as_list()[1] != B.get_shape().as_list()[0]:
			return None
		
		dense_shape = [A.get_shape().as_list()[0], B.get_shape().as_list()[0]]

		Aux_A = tf.concat([tf.cast(A.indices, dtype=A.values.dtype), tf.expand_dims(A.values, axis=1)], axis=1)
		
		for i in tqdm(range(dense_shape[1])):
			r_idx_A = tf.gather(Aux_A, tf.squeeze(tf.where(Aux_A[:, 1] == i))).numpy()
			r_idx_A[:,2] = r_idx_A[:,2]*B[i]
			Aux_A = tf.tensor_scatter_nd_update(Aux_A, tf.where(Aux_A[:, 1] == i), r_idx_A)

		return tf.sparse.SparseTensor(A.indices, Aux_A[:,2], dense_shape)



	def predict(self, Features, N_representations, N_clusters, Knn=10, cntTimes=10):
		#Stage 1
		print("Stage 1")
		RpData = self.__getRepresentativesByHybridSelection(Features, N_representations)
		
		print("Stage 2")
		#Stage 2
		N = Features.shape[0]

		cntRepCls = int(np.floor(np.sqrt(RpData.cluster_centers_.shape[0])))
		AprData = KMeans(n_clusters=cntRepCls).fit(RpData.cluster_centers_)

		cntRepCls = AprData.cluster_centers_.shape[0]
		
		Features = tf.constant(Features, dtype=tf.float32)
		Apr_centers = tf.constant(AprData.cluster_centers_, dtype=tf.float32)
		
		minCenterIdxs = tf.math.argmin(self.__dist_tf(Features, Apr_centers), axis=1)

		del Apr_centers
		
		Apr_labels = tf.constant(AprData.labels_)
		RpData_centers = tf.constant(RpData.cluster_centers_, dtype=tf.float32)

		nearestRepInRpFeaIdx = tf.zeros([N], dtype=tf.int64)
		for i in range(cntRepCls):
			originalIdxs = tf.where(minCenterIdxs == i)
			aprDataIdxs = tf.where(Apr_labels == i)

			originalTemp = tf.gather_nd(Features, originalIdxs)

			aprDataTemp = tf.gather_nd(RpData_centers, aprDataIdxs)

			temp = self.__dist_tf(originalTemp, aprDataTemp)

			tempMin = tf.math.argmin(temp, axis=1) 

			update = tf.squeeze(tf.gather(aprDataIdxs, tempMin)) 

			nearestRepInRpFeaIdx = tf.tensor_scatter_nd_update(nearestRepInRpFeaIdx, originalIdxs, update)
		
		del AprData, Apr_labels, update, tempMin, aprDataIdxs, aprDataTemp, minCenterIdxs

		neighSize = 10*Knn

		RpFeaW = self.__dist_tf(RpData_centers, RpData_centers)

		RpFeaKnnIdx = tf.argsort(RpFeaW, axis=1)
		
		del RpFeaW

		RpFeaKnnIdx = RpFeaKnnIdx[:,:int(neighSize)]
		
		RpFeaKnnDist = tf.zeros([N, RpFeaKnnIdx.get_shape().as_list()[1]]);

		for i in range(N_representations):
			originalIdxs = tf.where(nearestRepInRpFeaIdx == i)
			originalTemp = tf.gather_nd(Features, originalIdxs)
			RpFeatTemp = tf.gather(RpData_centers, RpFeaKnnIdx[i])

			temp = self.__dist_tf(originalTemp, RpFeatTemp)

			RpFeaKnnDist = tf.tensor_scatter_nd_update(RpFeaKnnDist, originalIdxs, temp)
		
		del RpData, originalIdxs, originalTemp, RpFeatTemp, temp, Features, RpData_centers
		
		RpFeaKnnIdxFull = tf.gather(RpFeaKnnIdx, nearestRepInRpFeaIdx)

		del RpFeaKnnIdx, nearestRepInRpFeaIdx

		knnDist = tf.zeros([N, Knn])
		knnIdx = tf.zeros([N, Knn], dtype=tf.int32)

		for i in range(Knn):
			idx = tf.math.argmin(RpFeaKnnDist, axis=1)
			minV = tf.reduce_min(RpFeaKnnDist, axis=1)
			idx_R = tf.range(0, limit=N, dtype=tf.int64)
			temp = (idx * N) + idx_R
			col = tf.expand_dims(tf.zeros([N], dtype=tf.int64)+i, axis=1)
			idx_R = tf.concat([tf.expand_dims(idx_R, axis=1), col], axis=1)

			knnDist = tf.tensor_scatter_nd_update(knnDist, idx_R, minV)

			knnIdx = tf.tensor_scatter_nd_update(knnIdx, idx_R, RpFeaKnnIdxFull[:,i])
			
			temp = tf.where(temp<N, np.inf, RpFeaKnnDist[:,i])
			RpFeaKnnDist = tf.tensor_scatter_nd_update(RpFeaKnnDist, idx_R, temp)
		
		del RpFeaKnnDist, RpFeaKnnIdxFull, idx, minV, idx_R, temp, col

		knnMeanDiff = tf.reduce_mean(knnDist)
		Gsdx = tf.math.exp(-(tf.square(knnDist)/2*tf.square(knnMeanDiff)))
		Gsdx = tf.where(Gsdx == 0.0, np.finfo(np.float32).eps, Gsdx)  

		del knnDist, knnMeanDiff

		Gidx = tf.expand_dims(tf.reshape(tf.constant(matlib.repmat(np.arange(N), Knn, 1).T), shape=[N*Knn]), axis=1)
		Gidx = tf.concat([Gidx, tf.expand_dims(tf.reshape(knnIdx,  shape=[N*Knn]), axis=1)], axis=1)
		
		B = tf.sparse.SparseTensor(tf.cast(Gidx, dtype=tf.int64), tf.reshape(Gsdx,  shape=[N*Knn]), [N, N_representations])
		B = tf.sparse.reorder(B)

		del Gsdx, Gidx, knnIdx

		print("Stage 3")
		#Stage 3
		dx = tf.sparse.reduce_sum(B, axis=1)
		dx = 1/dx
		dx = tf.where(tf.math.is_nan(dx), 0, dx)
		idx = tf.expand_dims(tf.range(0, limit=N, dtype=tf.int64), axis=1) 
		Dx = tf.sparse.SparseTensor(tf.concat([idx, idx], axis=1), dx, [N, N])

		del idx

		#Er = B.T @ Dx @ B
		#Er.shape => [N_representation, N_representation] => 1000x1000
		Er = tf.sparse.sparse_dense_matmul(self.__sparse_diagonal_matmul(tf.sparse.transpose(B), dx), tf.sparse.to_dense(B))
		#Er = tf.sparse.sparse_dense_matmul(tf.sparse.sparse_dense_matmul(tf.sparse.to_dense(tf.sparse.transpose(B)), Dx), B)
		


		d = tf.math.reduce_sum(Er, axis=1)
		d = 1/tf.math.sqrt(d)
		idx = tf.expand_dims(tf.range(0, limit=N_representations, dtype=tf.int64), axis=1) 
		D = tf.sparse.SparseTensor(tf.concat([idx, idx], axis=1), d, [N_representations, N_representations])
		
		del d, idx
		
		# Dr = self.__sparse_matmul(self.__sparse_matmul(D, Er), D)
		Dr = tf.sparse.sparse_dense_matmul(tf.sparse.sparse_dense_matmul(D, Er), D)
		# Dr = tf.sparse.to_dense(Dr)
		Dr = (Dr + tf.transpose(Dr))/2

		#Dr.shape => [N_representation, N_representation] => 1000x1000
		aval, avec = tf.linalg.eig(Dr)

		del Dr, Er

		idx = tf.argsort(tf.math.real(aval), direction='DESCENDING').numpy()[:N_clusters]
		
		avec = tf.gather(tf.math.real(avec), idx, axis=1)

		Ncut_avec =	tf.sparse.sparse_dense_matmul(D, avec)

		# res =  tf.sparse.sparse_dense_matmul(self.__sparse_matmul(Dx, B), Ncut_avec)
		res =  tf.matmul(tf.sparse.sparse_dense_matmul(Dx, tf.sparse.to_dense(B)), Ncut_avec)
		
		del Dx, idx, D, B, Ncut_avec, aval, avec

		norm = (tf.math.sqrt(tf.math.reduce_sum(res*res, axis=1)) + 1e-10)
		norm = tf.expand_dims(norm, axis=1)
		norm = tf.concat([norm, norm], axis=1)
		res = res/norm
		del norm

		res = res.numpy()

		return KMeans(n_clusters=N_clusters).fit(res).labels_

'''
TESTE
'''

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons

random_state = 21
X_mn, y_mn = make_moons(100000, noise=.07, random_state=random_state)
print(X_mn.shape)
cmap = 'viridis'
dot_size=50

uspec=USPEC()

print("Comecou")
labels = uspec.predict(X_mn, 1000, 2)
print("Terminou")

fig, ax = plt.subplots(figsize=(9,7))
ax.set_title('Data after spectral clustering from scratch', fontsize=18, fontweight='demi')
ax.scatter(X_mn[:, 0], X_mn[:, 1],c=labels,s=dot_size, cmap=cmap)

plt.show()