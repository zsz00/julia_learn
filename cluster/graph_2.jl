using LightGraphs, MetaGraphs
using NPZ
using Dates
using ProgressMeter
using LinearAlgebra
using Statistics
using RDatasets
using Clustering
using Distances


function cluster_2()
    t0 = Dates.now()
    G = Graph()
    mg = MetaGraph(G)
    # feats = npzread(raw"C:\zsz\ML\code\DL\face_cluster\face_cluster\tmp2\data\valse19.npy")
    feats = npzread("/data5/yongzhang/cluster/data/cluster_data/valse/valse_feat.npy")

    size_1 = size(feats)[1]
    t1 = Dates.now()
    println("used: ", (t1 - t0).value/1000, " s, ", size_1)

    p = Progress(size_1)
    Threads.@threads for i in range(1, stop=size_1)    # n*(n-1)/2.   @showprogress
        add_vertex!(mg)
        set_props!(mg, i, Dict(:feat=>feats[i,1:end]))  # feats可以不存储在图里,可以存储在外边,可以用节点号索引.
        next!(p)  # 进度条
        # nodes_1 = vertices(mg)
        # println("nodes:", collect(nodes_1))  # 打印每个node的key/id.

        for key in range(1, stop=i-1)   # nodes 是一个嵌套字典结构
            feat_1 = feats[key, 1:end]  # get_prop(mg, key, :feat)   # 在图中查找节点数据. 按字典方式查.hash查找. 竟然不能并行
            cos = feats[i, 1:end]' * feat_1   # 一个一个的算, 没有用矩阵运算. 可以换成矩阵运算.
            # println(join([i, key, cos], ", "))
            if cos > 0.5
                add_edge!(mg, i, key)
                set_prop!(mg, Edge(i, key), :weight, cos)
                # println("add_edge:", i, key)
                # println("ddd", get_prop(mg, Edge(i, key), :weight))
            end
        end
        # println("nv(mg):", nv(mg))  # mg的节点数量
        # gg, var = get_gg(mg, i)   # 找到包含当前node的子图
        # println("i:", i, " var:", var)

        # 递归处理 var大的簇. 动态阈值.  怎么递归,循环

    end
    t2 = Dates.now()
    println("used: ", (t2 - t1).value/1000, " s, ", size_1)
    g_list = connected_components(mg)
    # println("g_list:", size(g_list)[1])    # 子图的数量
end


function cluster_1(feats)
    t0 = Dates.now()
    G = Graph()
    mg = MetaGraph(G)

    size_1 = size(feats)[1]
    p = Progress(size_1)
    for i in range(1, stop=size_1)    # n*(n-1)/2.   @showprogress  Threads.@threads 
        add_vertex!(mg)
        set_props!(mg, i, Dict(:feat=>feats[i,1:end]))  # feats可以不存储在图里,可以存储在外边,可以用节点号索引.
        next!(p)  # 进度条
        # nodes_1 = vertices(mg)
        # println("nodes:", collect(nodes_1))  # 打印每个node的key/id.

        for key in range(i+1, stop=size_1)   # nodes 是一个嵌套字典结构
            feat_1 = feats[key, 1:end]  # get_prop(mg, key, :feat)   # 在图中查找节点数据. 按字典方式查.hash查找. 竟然不能并行
            # cos = feats[i, 1:end]' * feat_1   # cos相似度. 一个一个的算, 没有用矩阵运算. 
            dist = euclidean(feats[i, 1:end]', feat_1)   # L2距离
            println(join([i, key, dist], ", "))
            # if cos > 0.5
            if dist < 1
                add_edge!(mg, i, key)
                set_prop!(mg, Edge(i, key), :weight, cos)
                # println("add_edge:", i, key)
                # println("ddd", get_prop(mg, Edge(i, key), :weight))
            end
        end
        # println("nv(mg):", nv(mg))  # mg的节点数量
        # gg, var = get_gg(mg, i)   # 找到包含当前node的子图
        # println("i:", i, " var:", var)

        # 递归处理 var大的簇. 动态阈值.  怎么递归,循环
    end
    t2 = Dates.now()
    println("used: ", (t2 - t0).value/1000, " s, ", size_1)
    g_list = connected_components(mg)
    println("g_list:", size(g_list)[1])    # 子图的数量
    println(g_list)
end


# cluster_1_2()


function test_1()
    iris = dataset("datasets", "iris")  # iris花的数据
    x = convert(Matrix, iris[:, 1:4])
    println(size(x))
    x = x

    cluster_1(x)
end

test_1()
