# https://github.com/JuliaStats/Clustering.jl   julia聚类  (K-Means, 层次聚类)
using Clustering
using Makie  # plot
using PyCall
using Distances 
np = pyimport("numpy")
scipy = pyimport("scipy.sparse")
load_npz = scipy."load_npz"

x = features = np.load(raw"D:\download\features\512.fea.npy")

r = euclidean(x, x)
print(size(r))
# from scipy.sparse import csr_matrix,save_npz,load_npz
# D = load_npz(raw"C:\zsz\ML\code\DL\face_cluster\test\data_1\0429_global_mat.npz")  # 稀疏的距离矩阵
# println(size(D))
# D = rand(100, 100);
# D += D'; # symmetric distance matrix (optional)
result = hclust(r, linkage=:single);  # 单链接聚类
println(size(result.merges), result.heights, result.merges)
aa = cutree(result; h=6)
println(aa)

