# online 流式 聚类处理. OnlineStats
using OnlineStatsBase
using OnlineStats
using CSV, Plots
using RDatasets
using Clustering
using Distances
using NPZ
using Dates
using ProgressMeter
using ThreadsX


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
    # https://joshday.github.io/OnlineStats.jl/latest/bigdata/
    rows = CSV.Rows(open("cluster/stream/data/iris.txt"); reusebuffer=true)
    # rows = CSV.Rows(open("data/iris.txt"); reusebuffer = true)
    # rows = dataset("datasets", "iris")  # iris花的数据
    itr = (row.variety => parse(Float64, row.sepal_length) for row in rows)
    println(itr)
    
    o = GroupBy(String, Hist(4:0.25:8))
    # o = GroupBy(String, CountMap())
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
    # cluster online   2020.10.18
    # plant = dataset("cluster", "plantTraits")   # plantTraits数据集, 有missing, 返回的是DataFrame格式的数据
    # iris = dataset("datasets", "iris")  # iris花的数据
    # x = convert(Matrix, iris[:, 1:4])
    # println(size(x))
    # # x = transpose(x)
    # println(typeof(x))
    println("test_4()")
    t0 = Dates.now()
    if Sys.iswindows()
        feats = npzread(raw"C:\zsz\ML\code\DL\face_cluster\face_cluster\tmp2\data\valse19.npy")
    else
        # feats = npzread("/data5/yongzhang/cluster/data/cluster_data/valse/valse_feat.npy")
        feats = npzread("/data/zhangyong/data/longhu_1/sorted_2/feats.npy")
    end
    
    feats = convert(Matrix, feats[1:1000, 1:end])
    size_1 = size(feats)[1]
    t1 = Dates.now()
    println("used: ", (t1 - t0).value/1000, "s, ", size_1)
    # println(size(feats))

    op = ClusterOp()
    b = nothing
    @showprogress for i in 1:size_1
        x1 = feats[i,1:end]
        # println(typeof(x1))
        b = fit!(op, x1)
        # b = ThreadsX.reduce(op, x1)  # 报错
        # println(op.nodes)
    end
    t2 = Dates.now()
    println(size_1, ", id:", length(Set(values(op.nodes))))
    # println(op.nodes)
    println("used: ", (t2 - t1).value/1000, "s")

end

# 自定义类型, 结构体
mutable struct ClusterOp <: OnlineStat{Vector{Float32}}
    top_k::Int
    th::Float32
    num::Int64
    nodes::Dict
    clusters::Dict
    index::Array
    ClusterOp() = new(100, 0.6, 0, Dict(), Dict(), [])  # init
    # 调用外部api, 麻烦. 需要个julia api.  index=milvus_api_1.IndexMilvus(dim=384, repo="repo1")
end

# 重构 fit方法, 实现op功能
function OnlineStatsBase._fit!(o::ClusterOp, y)   # y::Array
    o.num += 1
    # feat_1 = transform(feature)   # 反序列化
    feat_1 = y

    # cos
    push!(o.index, feat_1)
    # dists, idxs = o.index.search(feat_1, o.top_k)
    feats_gallary = o.index
    feats_gallary = vcat((hcat(i...) for i in feats_gallary)...)  # 转换 shape

    feats_query = reshape(feat_1, (1,384))
    cos = feats_query * feats_gallary'  # cos相似度. 全取,没有top_k

    # init
    o.nodes[o.num] = o.num
    o.clusters[o.num] = [o.num]

    idx_1 = findall(cos .> o.th)
    idx_y = Tuple.(idx_1)

    for (_, j) in idx_y  # 遍历每个连接
        union_2!(o.num, j, o.nodes, o.clusters)
    end
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


function cluster_hac_1()
    """
    增量的层次聚类. 模拟流式

    """
    println("cluster_hac()")
    t0 = Dates.now()
    if Sys.iswindows()
        feats = npzread(raw"C:\zsz\ML\code\DL\face_cluster\face_cluster\tmp2\data\valse19.npy")
    else
        # feats = npzread("/data5/yongzhang/cluster/data/cluster_data/valse/valse_feat.npy")
        feats = npzread("/data/zhangyong/data/longhu_1/sorted_2/feats.npy")
    end
    feats = feats[1:10000, 1:end]
    size_1 = size(feats)[1]
    t1 = Dates.now()
    println("used: ", (t1 - t0).value/1000, "s, ", size_1)

    feats_1 = []  # repo
    count = 0
    threshold = 0.6   # 0.6
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

        idx_1 = findall(cos .> threshold)
        idx_1 = Tuple.(idx_1)
        # println(cos)
        # println(idx_1)

        for (_, j) in idx_1
            union_2!(i, j, nodes, clusters)  # 并查集, 合并. 基于dict的
        end

    end
    labels = values(nodes)

    t2 = Dates.now()
    println("labels: ", length(labels), " id:", length(Set(labels)))
    println("used: ", (t2 - t1).value/1000, "s, ", size_1)

end


# test_2()
test_4()
# cluster_hac_1()



#=
2020.10
JULIA_NUM_THREADS=4

cluster_hac_1()
used: 6.714s, 10000
labels: 10000 id:577
used: 1698.207s, 10000

test_4()
used: 6.607s, 10000
10000, id:577
used: 1685.058s

结论: 这两个增量的实现, 效果和速度 没啥区别

调用的api的:
10000, id:577
used: 564.013s


问题:  2020.10.22
0. 文档不够, 功能不够.
1. 并行 b=ThreadsX.reduce(op, x1) 失败
2. 没有可视化: graph可视化, matrix可视化
3. 没有flink高级, 没有graph优化
4. 没有souce和sink接口. 
5. 没有资源调度, 任务调度. 

OnlineStats 没有执行图, 没有 lazy执行, 跟flink原理不一样??
怎么动态增量的绘图画曲线?  

=#
