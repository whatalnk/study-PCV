# -*- coding: utf-8 -*-
from PIL import Image
from numpy import *

def pca(X):
  num_data, dim = X.shape # 行, 列
  
  mean_X = X.mean(axis=0) # 列方向
  X = X - mean_X

  if dim > num_data:
    M = dot(X, X.T) # covariance matrix; dot: 内積, T: 転置
    e, EV = linalg.eigh(M) # eigenvalues and eigenvectors; linalg:線形代数の関数群; eigh: エルミート行列の固有値と固有ベクトル
    tmp = dot(X.T, EV).T
    V = tmp[::-1]
    S = sqrt(e)[::-1]
    for i in range(V.shape[1]):
      V[:, 1] /= S
  else:
    U, S, V = linalg.svd(X) # 密行列の特異値分解
    V = V[:num_data]

  return V, S, mean_X
