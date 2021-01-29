# online 流式 聚类处理. OnlineStats
using OnlineStatsBase
using OnlineStats
using CSV, Plots
using RDatasets
using NPZ, JLD2
using Dates
using ProgressMeter
using ThreadsX
using Strs
include("milvus_api.jl")   # milvus_api_2


function test_4()
    # cluster online   2020.10.18
    println("test_4()")
    # 加载数据
    t0 = Dates.now()
    if Sys.iswindows()
        feats = npzread(raw"C:\zsz\ML\code\DL\face_cluster\face_cluster\tmp2\data\valse19.npy")
    else
        # feats = npzread("/data5/yongzhang/cluster/data/cluster_data/valse/valse_feat.npy")
        feats = npzread("/data/zhangyong/data/longhu_1/sorted_2/feats.npy")
    end
    
    feats = convert(Matrix, feats[1:195000, 1:end])
    size_1 = size(feats)[1]
    t1 = Dates.now()
    println("used: ", (t1 - t0).value/1000, "s, ", size_1)
    # println(size(feats))

    # 聚类op
    op = ClusterOp()
    b = nothing
    @showprogress for i in 1:size_1
        x1 = feats[i,1:end]
        # println(typeof(x1))
        b = fit!(op, x1)   # x1需要支持更复杂的对象,包括质量和时空.
        # ThreadsX.reduce(op, x1)  # 报错, 不能并行
        # println(op.nodes)
    end
    t2 = Dates.now()
    
    # 获取结果
    labels = values(op.nodes)
    println(f"img_sum:\(size_1), id_sum:\(length(Set(labels)))")
    println(f"used: \%.1f((t2 - t1).value/1000)s")
    @save "nodes_1.jld2" labels

end

# 自定义类型, 结构体
mutable struct ClusterOp <: OnlineStat{Vector{Float32}}
    top_k::Int32
    th::Float32
    batch_size::Int32
    num::Int64
    nodes::Dict     # {"n_id0": {"n_id":"pid", "blur":0.3}}  只存代表点
    clusters::Dict  # {"c_id0":{"c_id":"c_id0", "members": [], "c_size":5}  只存代表点
    collection_name::String   # init index
    vectors::Array  # 把一批的feat存到状态里. 为batch加的
    ids::Array    # 把一批的id存到状态里
    ClusterOp() = new(100, 0.5, 1000, 0, Dict(), Dict(), creat_collection("repo_test_3", 384), [], [])  # init
    # 调用外部api.  index=milvus_api_1.IndexMilvus(dim=384, repo="repo1")
end


# 重构 fit方法, 实现op功能
function OnlineStatsBase._fit!(o::ClusterOp, y)   # y::Array
    o.num += 1
    # feat_1 = transform(feature)   # 反序列化
    feat_1 = y

    # init
    o.nodes[o.num] = o.num 
    o.clusters[o.num] = [o.num]

    # 调用api
    # vectors = [feat_1]
    # ids = [string(o.num)]
    push!(o.vectors, feat_1)   # 把历史feat存到状态里
    push!(o.ids, string(o.num))
    # push!(o.ids, o.num)
    # batch/window. 批处理. 是不是可以加个window op.
    if o.num % o.batch_size == 0
        # add_obj(o.collection_name, o.vectors, o.ids)   # add  慢
        println(f"======:\(o.num), \(length(o.ids)), \(size(o.vectors))")
        insert_obj(o.collection_name, o.vectors, o.ids)   # add  慢
        rank_result = search_obj(o.collection_name, o.vectors, o.top_k)   # search rank
        dists, idxs = prcoess_results_3(rank_result, o.top_k)
        o.vectors = []
        o.ids = []

        batch = o.num ÷ o.batch_size - 1
        for i in 1:o.batch_size
            idx_1 = findall(dists[i,:] .> o.th)   # 返回的idx
            idx_y = idxs[i,:][idx_1]   # Tuple.(idx_1)

            for j in idx_y  # 遍历每个连接

                union_2!(batch*o.batch_size+i, j, o.nodes, o.clusters)
            end
        end
    end
end


function cluster_hac()
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
    println("labels: ", length(labels), " id:", length(Set(labels)))
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


test_4()
# cluster_hac()



#=
2020.10, 2020.11
JULIA_NUM_THREADS=4
---------------------------------
基于onlinestats_1_2.jl

可以做一些事, 虽然不够完善.
 

TODO:
0. 加同镜,跨镜 多时空阶段聚类
1. 加代表点, 代表点更新
2. 质量 加权动态阈值

要应用Window

通过blur和cos 做代表点更新


=#

