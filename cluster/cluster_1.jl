# https://github.com/JuliaStats/Clustering.jl   julia聚类  (包含K-Means, 层次聚类)
using Clustering
using NearestNeighbors
using Makie  # plot
using PyCall
using Distances 
np = pyimport("numpy")
scipy = pyimport("scipy.sparse")
load_npz = scipy."load_npz"


# 求距离矩阵  哎
function distance_1()
    x = randn(100000, 512)   
    mat = pairwise(CorrDist(), x, dims=1)  # CorrDist  Euclidean    # >10w时 Segmentation fault
    println(size(mat))
end

# 求距离矩阵
function get_mat()
    # x = features = np.load(raw"D:\download\features\512.fea.npy")
    x = np.load("/data/yongzhang/cluster/test_2/512.fea.npy")
    # dists = euclidean(x, x)
    println("size(x):", size(x))
    x = x'
    dists = pairwise(Euclidean(), x, x);   # 求L2距离/欧式距离.  和faiss的计算结果不同
    println("size(dists):", size(dists))
    sort!(dists, dims=2)   # rev=true
    dists = dists[1:end,1:1000]
    out_file = joinpath("/data/yongzhang/cluster/test_2/out_2", "mat.npy")
    np.save(out_file, dists)    
end

# 求距离矩阵
function get_mat_2()
    # x = np.load("/data/yongzhang/cluster/data_3/clean/feat/0302_feat_1.npy")
    x = np.load("/data/yongzhang/cluster/test_2/512.fea.npy")
    println("size(x):", size(x), " ", typeof(x))
    X = transpose(x)  # 矩阵转置, 也可以用 x'
    X = convert(Array, X)
    println("size(x):", size(X), " ", typeof(X))
    knearest = 1000
    kdtree = KDTree(X)
    println("knn...")
    idxs, dists = knn(kdtree, X, knearest, true)  # 单线程的
    println("idxs, dists:",size(idxs), size(dists))
    println(dists[1])

end


function cluster_1()
    # x = features = np.load(raw"D:\download\features\512.fea.npy")
    # r = euclidean(x, x)
    # print(size(r))
    # from scipy.sparse import csr_matrix,save_npz,load_npz
    # D = load_npz(raw"C:\zsz\ML\code\DL\face_cluster\test\data_1\0429_global_mat.npz")  
    # 加载稀疏的距离矩阵
    # D = load_npz("/data/yongzhang/cluster/data_3/clean/out_disk_2_jxx_1/disk_2_jxx_local_mat.npz")
    # D = np.load("/data/yongzhang/cluster/data_3/clean/out_disk_2_jxx_4/disk_2_jxx_mat.npy")
    D = np.load("/data/yongzhang/cluster/cluster_v1/cluster_test/mat.npy")  # 19G*2 mem
    # println("=======:", D.shape)   # (320066, 320066)
    mat = D  # D.data   # ok   
    # mat = mat[1:1000,1:end]
    println("=======:", size(mat))    # (2249544,) 
    # D = rand(100, 100);
    # D += D'; # symmetric distance matrix (optional)
    result = hclust(mat, linkage=:single, uplo=:U);  # 单链接 层次聚类
    # Distance matrix should be square. mat必须是n*n的对称矩阵. 或者 AbstractArray{T,2}
    println(size(result.merges), result.heights, result.merges)
    aa = cutree(result; h=6)
    println(aa)

end


# get_mat()
# get_mat_2()
# cluster_1()
@time distance_1()

#=



=#
