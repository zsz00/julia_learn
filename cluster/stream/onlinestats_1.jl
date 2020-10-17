using OnlineStats
using CSV, Plots
using RDatasets
using Clustering
using Distances
using NPZ
using Dates
using ProgressMeter


function test_1()
    # op = Series(Mean(), Variance(), P2Quantile(), Extrema())
    op = Series(Mean(), Variance())

    a = fit!(op, 1.0)
    print(a)
    
    b = fit!(a, [1,2,3])   # randn(5)   a, b is operation 
    print(b)

    c = merge!(a, b)
    d = value(c)
    print(c, d)

end


function test_2()

    rows = CSV.Rows(open("cluster/stream/data/iris.txt"); reusebuffer = true)
    itr = (row.variety => parse(Float64, row.sepal_length) for row in rows)
    println(itr)
    
    o = GroupBy(String, Hist(4:0.25:8))
    b = fit!(o, itr)
    println(b)
    plot(o, layout=(3,1))
    
end


function test_3()
    # cluster offline
    # plant = dataset("cluster", "plantTraits")   # plantTraits数据集, 有missing, 返回的是DataFrame格式的数据
    iris = dataset("datasets", "iris")  # iris花的数据
    x = convert(Matrix, iris[:, 1:4])
    println(size(x))
    x = x'
    # dists = euclidean(x, x)
    dists = pairwise(Euclidean(), x, x);   # 求L2距离/欧式距离. 和faiss的计算结果不同. 挺快的. 单进程
    println(size(dists))
    result = hclust(dists, linkage=:average, uplo=:U)   # 层次聚类(最小距离)  average single
    # println(result)
    # Distance matrix should be square. mat必须是n*n的对称矩阵. 或者 AbstractArray{T,2}
    println("result:")
    println(size(result.merges), result.heights, result.merges)
    aa = cutree(result; h=1)
    println(aa)
    println(size(aa), " id:", length(Set(aa)))

end


function test_4()
    # cluster online
    # plant = dataset("cluster", "plantTraits")   # plantTraits数据集, 有missing, 返回的是DataFrame格式的数据
    iris = dataset("datasets", "iris")  # iris花的数据
    x = convert(Matrix, iris[:, 1:4])
    println(size(x))
    x = x'
    dists = euclidean(x, x)
    # dists = pairwise(Euclidean(), x, x);   # 求L2距离/欧式距离. 和faiss的计算结果不同. 挺快的. 单进程
    println(size(dists))
    result = hclust(dists, linkage=:average, uplo=:U)  # 层次聚类(最小距离)  average single
    # println(result)
    # Distance matrix should be square. mat必须是n*n的对称矩阵. 或者 AbstractArray{T,2}
    println("result:")
    println(size(result.merges), result.heights, result.merges)
    aa = cutree(result; h=0.6)
    println(aa)
    println(size(aa), " id:", length(Set(aa)))

end


# mutable struct Cluster
#     id::String
#     members::Array{String,1}

#     size = length(members)
#     add([id])
# end


function cluster_hac()
    """
    固定阈值层次聚类

    """
    t0 = Dates.now()
    if Sys.iswindows()
        feats = npzread(raw"C:\zsz\ML\code\DL\face_cluster\face_cluster\tmp2\data\valse19.npy")
    else
        # feats = npzread("/data5/yongzhang/cluster/data/cluster_data/valse/valse_feat.npy")
        feats = npzread("/data5/yongzhang/cluster/data/cluster_data/ms1m/ms1m_part1_test_feat.npy")
    end
    size_1 = size(feats)[1]
    t1 = Dates.now()
    println("used: ", (t1 - t0).value/1000, "s, ", size_1)

    feats_1 = []  # repo
    count = 0
    threshold = 0.60   # 0.6
    nodes = Dict()  
    clusters = Dict()
    @showprogress for i in range(1, stop=size_1)    # n*(n-1)/2.   @showprogress 
        push!(feats_1, feats[i,1:end])  # append   push
        feats_3 = vcat((hcat(i...) for i in feats_1)...)  # 转换 shape
        # feats_3 = vcat(feats_1...)   # 转换 shape
        feats_2 = feats[i, 1:end]
        feats_2 = reshape(feats_2, (1,384))
        cos = feats_2 * feats_3'  # cos相似度
        # println(join([i, size(feats_2), size(feats_3), size(cos)], ", "))
        # cos = feats[i, 1:end]' * feat_1   # cos相似度.  
        # dist = euclidean(feats_3', feats_2)   # L2距离
        # println(join([i, dist], ", "))

        # init 
        nodes[i] = i  
        clusters[i] = [i] 

        # idx_1 = findall(x-> x > threshold, cos)
        idx_1 = findall(cos .> threshold)
        idx_1 = Tuple.(idx_1)
        # println(cos)
        # println(idx_1)

        for (_, j) in idx_1
            union_2!(i, j, nodes, clusters)  # 并查集, 合并
        end

    end
    labels = values(nodes)

    t2 = Dates.now()
    println("labels: ", size(labels), length(Set(labels)))
    println("used: ", (t2 - t1).value/1000, "s, ", size_1)

end


function union_2!(i, j, nodes, clusters)
    id_1 = nodes[i]
    id_2 = nodes[j]
    if id_1 == id_2  # 查
        return
    else
        # 更新 set
        if length(clusters[id_1]) >= length(clusters[id_2])  # 更新小的cluster到大的中
            id_max, id_min = id_1, id_2
        else
            id_max, id_min = id_2, id_1
        end

        for idx_ in clusters[id_min]  # 把id_2的转为id_1
            nodes[idx_] = id_max
        end
        append!(clusters[id_max], pop!(clusters, id_min))  # 合并
    end

end





# test_3()
cluster_hac()

