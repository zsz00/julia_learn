# julia聚类, 层次聚类
using CSV, Plots
using RDatasets
using Clustering
using Distances
using NPZ
using Dates
using ProgressMeter
using NearestNeighbors


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


function cluster_hac_2()
    """
    增量的层次聚类. 模拟流式
    改进: rank, knn
    """
    println("cluster_hac_2()")
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


#=
function knn1(index, feats_1, feats, batch, start, knn=5, knn_th=0.5)
    # 外挂到 milvus/fse
    gdists_1, gidx_1 = index.search(feats_1, knn)  # 从所有底库里查询top5
    gidx_2, gdists_2 = hac_1.rank2(feats_1, feats_1, knn, ngpus='')  # 从当前这批数据中查op5

    gidx_2 = gidx_2 + start
    gdists = np.concatenate([gdists_1, gdists_2], axis=1)  # bs*10
    gidx = np.concatenate([gidx_1, gidx_2], axis=1)
    top_k_idx = np.argsort(gdists, axis=1)[:, ::-1][:, 0: knn]  # top_k  [0:knn]
    gdists = np.take_along_axis(gdists, top_k_idx, axis=1)  # 不快
    gidx = np.take_along_axis(gidx, top_k_idx, axis=1)
    center = gdists > knn_th
    for m in range(batch)
        idx = gidx[m][center[m]]
        tmp_feats = feats[idx]  # 从底库取
        feats_1[m] = np.mean(tmp_feats, axis=0, keepdims=True)
    end
    feats_1 = normalize(feats_1)  # 归一化
    return feats_1

end
feats_1 = knn_1(index, feats_1, feats, batch, start, knn_th=knn_th)

function knn2(index, feats_1, feats, batch, start, knn=5, knn_th=0.5)
    # 外挂到 milvus/fse
    gdists_1, gidx_1 = index.search(feats_1, knn)  # 从所有底库里查询top5
    gidx_2, gdists_2 = hac_1.rank2(feats_1, feats_1, knn, ngpus='')  # 从当前这批数据中查op5

    gidx_2 = gidx_2 + start
    gdists = np.concatenate([gdists_1, gdists_2], axis=1)  # bs*10
    gidx = np.concatenate([gidx_1, gidx_2], axis=1)
    top_k_idx = np.argsort(gdists, axis=1)[:, ::-1][:, 0: knn]  # top_k  [0:knn]
    gdists = np.take_along_axis(gdists, top_k_idx, axis=1)  # 不快
    gidx = np.take_along_axis(gidx, top_k_idx, axis=1)
    center = gdists > knn_th
    for m in range(batch)
        idx = gidx[m][center[m]]
        tmp_feats = feats[idx]  # 从底库取
        feats_1[m] = np.mean(tmp_feats, axis=0, keepdims=True)
    end
    feats_1 = normalize(feats_1)  # 归一化
    return feats_1

end
=#

# 求距离矩阵
function distance_3()
    # knn, kdtree
    feats = npzread("/data/zhangyong/data/longhu_1/sorted_2/feats.npy")
    feats = feats[1:100, 1:end]
    size_1 = size(feats)[1]
    x = feats   # npzread("x.npy")
    println("size(x):", size(x), " ", typeof(x))
    X = transpose(x)  # 矩阵转置, 也可以用 x'. 必须. 垃圾
    X = convert(Array, X)
    println("size(x):", size(X), " ", typeof(X))
    k = 10
    gallery = X
    query = X
    brutetree = BruteTree(gallery)  # 暴力搜索树
    # kdtree = KDTree(gallery, leafsize=4)   # 同index.add(gallery) 
    idxs, dists = knn(brutetree, query, k, true)  # 单线程的, 很慢.  # query查询.  
    dists = vcat((hcat(i...) for i in dists)...)  # 转换 shape
    idxs = vcat((hcat(i...) for i in idxs)...)  # 转换 shape
    println("idxs: $(size(idxs)), dists: $(size(dists))")
    println(dists[1:10, end:1])
    # npzwrite("dist_kdtree.npy", dists)
    # 1081.681778 seconds (21.04 M allocations: 1.623 GiB, 0.05% gc time)
end



distance_3()
# cluster_hac_2()


#=
2020.10, 2020.11



=#
