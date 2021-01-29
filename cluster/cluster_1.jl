# https://github.com/JuliaStats/Clustering.jl   julia聚类 
# using Clustering
using NearestNeighbors
# using Makie  # plot
using PyCall
using Distances 
using RDatasets
using NPZ
np = pyimport("numpy")
# scipy = pyimport("scipy.sparse")
# load_npz = scipy."load_npz"


# 求距离矩阵 
function distance_1()
    # x = randn(40000, 512)   
    x = dataset("cluster", "plantTraits")   # 数据集中有missing, 返回的是DataFrame格式的数据
    x = convert(Matrix, x[:, 2:end])
    mat = pairwise(CorrDist(), x, dims=1)  # CorrDist  Euclidean    # >10w时 Segmentation fault
    println(size(mat))
end

# 求距离矩阵
function distance_2()
    # dist + sort
    # x = np.load("/data5/yongzhang/cluster/test_2/valse19.npy")
    # npzwrite("x.npy", x)
    x = npzread("x.npy")
    println("size(x):", size(x))
    x = x'
    # dists = euclidean(x, x)
    dists = pairwise(Euclidean(), x, x);   # 求L2距离/欧式距离.  和faiss的计算结果不同. 挺快的. 单进程
    println("size(dists):", size(dists))
    sort!(dists, dims=2)   # rev=true   # 排序很慢,还爆内存. 求ann/knn. 需要高级的ann算法
    dists = dists[1:end,1:1000]
    npzwrite("dist.npy", dists)
    # 555.789902 seconds (12.87 M allocations: 19.691 GiB, 0.11% gc time)
end

# 求距离矩阵
function distance_3()
    # knn, kdtree
    x = npzread("x.npy")
    println("size(x):", size(x), " ", typeof(x))
    X = transpose(x)  # 矩阵转置, 也可以用 x'
    X = convert(Array, X)
    println("size(x):", size(X), " ", typeof(X))
    k = 100
    gallery = X
    query = X
    println("knn...")
    brutetree = BruteTree(data)  # 暴力搜索树
    # kdtree = KDTree(gallery, leafsize=4)   # 同index.add(gallery) 
    idxs, dists = knn(kdtree, query, k, true)  # 单线程的, 很慢.  # query查询. (70184,)
    dists = vcat((hcat(i...) for i in dists)...)  # 转换 shape
    println("idxs: $(size(idxs)), dists: $(size(dists))")
    println(dists[1])
    npzwrite("dist_kdtree.npy", dists)
    # 1081.681778 seconds (21.04 M allocations: 1.623 GiB, 0.05% gc time)
end

# 求距离矩阵
function distance_4()
    # Rayuela.jl ,   PQ. 不成熟,不会用
    x = npzread("x.npy")
    println("size(x):", size(x), " ", typeof(x))
    X = transpose(x)  # 矩阵转置, 也可以用 x'
    X = convert(Array, X)
    println("size(x):", size(X), " ", typeof(X))
    k = 100
    gallery = X
    query = X
    println("knn...")
    # ...
    dists = vcat((hcat(i...) for i in dists)...)  # 转换 shape
    println("idxs: $(size(idxs)), dists: $(size(dists))")
    println(dists[1])
    npzwrite("dist_kdtree.npy", dists)
    #  
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


# cluster_1()
# @time distance_2()
@time distance_3()


#=
2020.1

=#
