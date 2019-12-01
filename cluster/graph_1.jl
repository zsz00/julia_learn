using LightGraphs, MetaGraphs
using NPZ
using Dates
using ProgressMeter
using LinearAlgebra
using Statistics


function cluster_1()
    """
    动态的 层次聚类. 算方差
    used: 14845.068 s, 70184.  4h on 10.42.64.84
    g_list:(3358,)
    
    """
    t0 = Dates.now()
    G = Graph()
    mg = MetaGraph(G)
    # feats = npzread(raw"C:\zsz\ML\code\DL\face_cluster\face_cluster\tmp2\data\valse19.npy")
    feats = npzread("/data5/yongzhang/cluster/data/cluster_data/valse/valse_feat.npy")

    size_1 = size(feats)[1]
    t1 = Dates.now()
    println("used: ", (t1 - t0).value/1000, " s, ", size_1)
    @showprogress for i in range(1, stop=size_1)    # n*(n-1)/2.   @showprogress
        add_vertex!(mg)
        set_props!(mg, i, Dict(:feat=>feats[i,1:end]))
        # println(props(mg, i))
        # nodes_1 = vertices(mg)
        for key in range(1, stop=i-1)   # nodes 是一个嵌套字典结构
            feat_1 = get_prop(mg, key, :feat)   # nodes_1[key]["feat"]
            cos = feats[i, 1:end]' * feat_1
            # println(join([i, key, cos], ", "))
            if cos > 0.5
                add_edge!(mg, i, key)
                set_prop!(mg, Edge(i, key), :weight, cos)
                # println("ddd", get_prop(mg, Edge(i, key), :weight))
            end
        end
    end
    t2 = Dates.now()
    println("used: ", (t2 - t1).value/1000, " s, ", size_1)

    g_list = connected_components(mg)
    println("g_list:", size(g_list))
end


function cluster_2()
    t0 = Dates.now()
    G = Graph()
    mg = MetaGraph(G)
    # feats = npzread(raw"C:\zsz\ML\code\DL\face_cluster\face_cluster\tmp2\data\valse19.npy")
    feats = npzread("/data5/yongzhang/cluster/data/cluster_data/valse/valse_feat.npy")

    size_1 = size(feats)[1]
    t1 = Dates.now()
    println("used: ", (t1 - t0).value/1000, " s, ", size_1)
    @showprogress for i in range(1, stop=size_1)    # n*(n-1)/2.   @showprogress
        add_vertex!(mg)   #  
        set_props!(mg, i, Dict(:feat=>feats[i,1:end]))  # feats可以不存储在图里,可以存储在外边,可以用节点号索引.
        # nodes_1 = vertices(mg)
        # println("nodes:", collect(nodes_1))  # 打印每个node的key/id.

        for key in range(1, stop=i-1)   # nodes 是一个嵌套字典结构
            feat_1 = get_prop(mg, key, :feat)   # 在图中查找节点数据. 按字典方式查.hash查找.
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
        gg, var = get_gg(mg, i)   # 找到包含当前node的子图
        # println("i:", i, " var:", var)

        # 递归处理 var大的簇. 动态阈值.  怎么递归,循环

    end
    t2 = Dates.now()
    println("used: ", (t2 - t1).value/1000, " s, ", size_1)

end

function cluster_3()
    t0 = Dates.now()
    G = Graph()
    mg = MetaGraph(G)
    # feats = npzread(raw"C:\zsz\ML\code\DL\face_cluster\face_cluster\tmp2\data\valse19.npy")
    feats = npzread("/data5/yongzhang/cluster/data/cluster_data/valse/valse_feat.npy")

    size_1 = size(feats)[1]
    t1 = Dates.now()
    println("used: ", (t1 - t0).value/1000, " s, ", size_1)
    @showprogress for i in range(1, stop=size_1)    # n*(n-1)/2.   @showprogress
        add_vertex!(mg)   #  
        set_props!(mg, i, Dict(:feat=>feats[i,1:end]))  # feats可以不存储在图里,可以存储在外边,可以用节点号索引.
        # nodes_1 = vertices(mg)
        # println("nodes:", collect(nodes_1))  # 打印每个node的key/id.


        
        for key in range(1, stop=i-1)   # nodes 是一个嵌套字典结构
            feat_1 = get_prop(mg, key, :feat)   # 在图中查找节点数据. 按字典方式查.hash查找.
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
        gg, var = get_gg(mg, i)   # 找到包含当前node的子图
        # println("i:", i, " var:", var)

        # 递归处理 var大的簇. 动态阈值.  怎么递归,循环

    end
    t2 = Dates.now()
    println("used: ", (t2 - t1).value/1000, " s, ", size_1)

end


function get_gg(mg, i)
    g_list = connected_components(mg)
    # println("g_list:", size(g_list)[1])    # 子图的数量
    gg = 0
    # println("mg:", mg)
    for g in g_list    # 找到包含当前node的子图
        sg, vmap = induced_subgraph(mg, g)   # 子图sg 里的节点编号是变了,变成子集的新节点了(1,2..).
        # println("g_1: ", join([g, collect(vertices(sg)), i, vmap], ", "))
        if i in vmap  # has_vertex
            gg = sg
            # println("gg: ", gg, i)
            break
        end
    end

    # 求这个子图的 var. 利用所有边长.
    # println("n_v:", nv(gg))  # 子图的节点数
    sim = []
    # println("ddd:", size(collect(edges(gg))))   # 子图的边数量
    for e in collect(edges(gg))
        cos = get_prop(gg, e, :weight)
        # println("e:", e)
        # println("cos:", cos)
        push!(sim, cos)
    end
    # println("sim:", sim)
    if sim == []
        sim_mean = 1
    else
        sim_mean = mean(sim)
    end
    var = 1 - sim_mean

    return gg, var
end


cluster_2()



"""
used: 558.827 s

图数据/图数据库 是存储在字典数据结构中的, 像mongodb. 读写,查询速度会快吗??

按对象类型各自单独存储, 节点数据存储在节点文件里, 边数据存储在边文件里, 索引数据存储在索引文件里. 
这样每个文件里, 数据就是同类型的, 可以高效存储和查询. 
节点数据: 有两种存储方式
1. 以字典方式存储在图里, 查询时候是按key-value方式查询, 字典/hash查询很快. 
2. 以结构数据存储在图外, 查询时按索引位置查找. 

感觉在图里,也是全走的索引.

每个img跟所有img算cos, 找出cos>th的点,连边. 
每个img跟所有img算cos, 找出top_k点. 


"""




