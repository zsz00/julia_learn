# https://github.com/JuliaStats/Clustering.jl   julia聚类  (包含K-Means, 层次聚类)
using Clustering
using NearestNeighbors
using Makie  # plot
using PyCall
using Distances 
np = pyimport("numpy")
scipy = pyimport("scipy.sparse")
load_npz = scipy."load_npz"

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

end


# get_mat()
get_mat_2()
