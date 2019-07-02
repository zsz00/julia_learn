# coding=utf-8
import numpy as np
from scipy.spatial.distance import pdist

# d
x = np.load("/data/yongzhang/cluster/test_2/512.fea.npy")
print("x", x.shape)
dist2 = pdist(np.vstack([x,x]),metric='euclidean')

print('dist2', dist2.shape)


