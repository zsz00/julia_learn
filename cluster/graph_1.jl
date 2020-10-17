# import Pkg; Pkg.add("NPZ")
using LightGraphs, MetaGraphs
using NPZ
using Dates
using ProgressMeter
using LinearAlgebra
using Statistics
# using ProfileView


function cluster_1()
    """
    增量的 层次聚类. 算方差
    used: 14845.068 s, 70184.  4h on 10.42.64.84
    g_list:(3358,)    太慢
    
    """
    t0 = Dates.now()
    G = Graph()
    mg = MetaGraph(G)
    # feats = npzread(raw"C:\zsz\ML\code\DL\face_cluster\face_cluster\tmp2\data\valse19.npy")
    # feats = npzread("/data5/yongzhang/cluster/data/cluster_data/valse/valse_feat.npy")
    feats = npzread(raw"C:\zsz\ML\code\DL\face_cluster\test\biaozhu\longhu_20w\feats.npy")

    size_1 = size(feats)[1]
    t1 = Dates.now()
    println("used: ", (t1 - t0).value/1000, "s, ", size_1)
    @showprogress for i in range(1, stop=size_1)    # n*(n-1)/2.   @showprogress
        add_vertex!(mg)
        set_props!(mg, i, Dict(:feat=>feats[i,1:end]))
        # println(props(mg, i))
        # nodes_1 = vertices(mg)
        for key in range(1, stop=i-1)   # nodes是一个嵌套字典结构
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
    println("used: ", (t2 - t1).value/1000, "s, ", size_1)

    g_list = connected_components(mg)
    println("g_list:", size(g_list))
end

function cluster_1_2()
    """
    增量的 层次聚类. 算方差
    used: 14845s, 70184.  4h on 10.42.64.84
    g_list:(3358,)
    2:05:04   1. 换用SimpleGraph()   2. 不用feat_1=get_prop(mg, key, :feat) 

    """
    t0 = Dates.now()
    G = SimpleGraph()  # Graph()
    mg = G  # MetaGraph(G)
    # feats = npzread(raw"C:\zsz\ML\code\DL\face_cluster\face_cluster\tmp2\data\valse19.npy")
    # feats = npzread("/data5/yongzhang/cluster/data/cluster_data/valse/valse_feat.npy")  # [1:20000,1:end]  # 7w*384 2s  比np.load()的慢
    feats = npzread(raw"C:\zsz\ML\code\DL\face_cluster\test\biaozhu\longhu_20w\feats.npy")  # [1:10000,1:end]

    size_1 = size(feats)[1]
    t1 = Dates.now()
    println("used: ", (t1 - t0).value/1000, "s, ", size_1)
    p = Progress(size_1)
    println("nthreads:", Threads.nthreads())
    for i=1:size_1    # n*(n-1)/2.    并行崩溃   Threads.@threads 单线程还更快
        add_vertex!(mg)
        next!(p)
        for j=1:i-1   # @inbounds
            feat_1 = feats[j, 1:end]  # get_prop(mg, j, :feat)   # nodes_1[j]["feat"]
            # cos = sum(feats[i, 1:end] .* feat_1)   # 慢
            cos = feats[i, 1:end]' * feat_1   # 慢
            # println(join([i, j, cos], ", "))
            if cos > 0.5
                add_edge!(mg, i, j)   # 并行崩溃, 不是原子操作 
            end
        end
    end
    t2 = Dates.now()
    println("used: ", (t2 - t1).value/1000, "s, ", size_1)

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
    p = Progress(size_1)
    Threads.@threads for i in range(1, stop=size_1)    # n*(n-1)/2.   @showprogress
        add_vertex!(mg) 
        set_props!(mg, i, Dict(:feat=>feats[i,1:end]))  # feats可以不存储在图里,可以存储在外边,可以用节点号索引.
        # sleep(1)
        next!(p)
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


function cluster_3()
    """
    批量的 层次聚类. 很快
    used: 26.141 s, 70184  batch:1000
    used: 23.641 s, 70184  batch:2000
    g_list:3358
    
    used: 186.011 s, 256016 batch:2000  zzf  
    g_list:4618

    ms1m:
    used: 647.45 s, 584013   0:10:47
    th:0.74 id_sum:121726

    """
    t0 = Dates.now()
    G = SimpleGraph()
    mg = G   # MetaGraph(G)
    if Sys.iswindows()
        feats = npzread(raw"C:\zsz\ML\code\DL\face_cluster\face_cluster\tmp2\data\valse19.npy")
    else
        # feats = npzread("/data5/yongzhang/cluster/data/cluster_data/valse/valse_feat.npy")
        feats = npzread("/data5/yongzhang/cluster/data/cluster_data/ms1m/ms1m_part1_test_feat.npy")
    end
    size_1 = size(feats)[1]
    t1 = Dates.now()
    println("used: ", (t1 - t0).value/1000, " s, ", size_1)
    feats_1 = []
    th = 0.74
    batch = 2000
    @showprogress for i in range(1, stop=size_1, step=batch)    # n*(n-1)/2.   @showprogress
        if size_1-i<batch
            batch = size_1-i+1
        end
        add_vertices!(mg, batch)   #  
        # set_props!(mg, i, Dict(:feat=>feats[i,1:end]))  # feats可以不存储在图里,可以存储在外边,可以用节点号索引.
        # nodes_1 = vertices(mg)
        # println("nodes:", collect(nodes_1))  # 打印每个node的key/id.
        push!(feats_1, feats[i:i+batch-1,1:end])
        # feats_3 = vcat((hcat(i...) for i in feats_1)...)  # 转换 shape
        feats_3 = vcat(feats_1...)   # 转换 shape
        feats_2 = feats[i:i+batch-1, 1:end]
        # feats_3 = reshape(feats_3, (:,384))
        # println(size(feats_2), size(feats_3'))
        cos =  feats_2 * feats_3'
        # println("cos:",size(cos))
        idx_1 = findall(cos.>th)  # 很慢
        # println("idx_1: ", size(idx_1))
        for j in Tuple.(idx_1)
            # print("i:", i, " j:", j)
            # println(j[1]+i,  ", ", j[2])
            add_edge!(mg, j[1]+i-1, j[2])    # 怎么批量加edges ??? 自己写个循环吧
        end
        # println("nv(mg):", nv(mg))  # mg的节点数量
        # gg, var = get_gg(mg, i)   # 找到包含当前node的子图
        # println("i:", i, " var:", var)
    end
    t2 = Dates.now()
    println("used: ", (t2 - t1).value/1000, "s, ", size_1)
    g_list = connected_components(mg)
    println("th:", th, " id_sum:", size(g_list)[1])    # 子图的数量
end


function cluster_3_2()
    """
    批量的 动态阈值层次聚类
    很慢
    valse:70184, 
    2h

    used: 3:10:08, 70184   笔记本(主频高):2:33:08
    th:0.6/0.48/0.4, g_list:3336

    count: 46428129
    used: 11901.713 s, 70184   3:18:21
    th:0.6/0.48/0.45, g_list:3335     var在for外无拆图


    """
    t0 = Dates.now()
    G = SimpleGraph()
    mg = G   # MetaGraph(G)
    # dir_1 = "/data5/yongzhang/cluster/data/cluster_data/zhaji/zzf/results_2"
    # feats = npzread(joinpath(dir_1, "new_feat_2730.npy"))  #
    if Sys.iswindows()
        feats = npzread(raw"C:\zsz\ML\code\DL\face_cluster\face_cluster\tmp2\data\valse19.npy")
    else
        # feats = npzread("/data5/yongzhang/cluster/data/cluster_data/valse/valse_feat.npy")
        feats = npzread("/data5/yongzhang/cluster/data/cluster_data/ms1m/ms1m_part1_test_feat.npy")
    end

    size_1 = size(feats)[1]
    t1 = Dates.now()
    println("used: ", (t1 - t0).value/1000, "s, ", size_1)
    feats_1 = []
    batch = 1000
    count = 0
    th_max = 0.70   # 0.6
    th_min = 0.64  # 0.48
    step_th = 0.02
    var_th = 0.5  # 0.45
    @showprogress for i in range(1, stop=size_1, step=batch)    # n*(n-1)/2.   @showprogress  不能并行
        if size_1-i<batch
            batch = size_1-i
        end
        add_vertices!(mg, batch) 
        push!(feats_1, feats[i:i+batch-1,1:end])
        # feats_3 = vcat((hcat(i...) for i in feats_1)...)  # 转换 shape
        feats_3 = vcat(feats_1...)   # 转换 shape
        feats_2 = feats[i:i+batch-1, 1:end]
        # feats_3 = reshape(feats_3, (:,384))
        # println(size(feats_2), size(feats_3'))
        cos =  feats_2 * feats_3'
        # println("cos:",size(cos))
        for k=1:batch
            th = th_max     # 0.66
            # count = 0
            flag = 0
            while th >= th_min   # 合并图的方式.
                idx_1 = findall(cos[k,:].>th)
                # println("idx_1: ", size(idx_1), " ", idx_1)
                if size(idx_1)[1] == 0
                    th -= step_th
                    th = round(th, digits=3)
                    # println("...")
                    continue
                end
                th -= step_th
                th = round(th, digits=3)
                
                for j in Tuple.(idx_1)   # 怎么批量加edges,自己写个循环吧
                    # println("i:", i, " j:", j)
                    # println(i+k-1,  ", ", j[1])
                    add_edge!(mg, i+k-1, j[1])    # 加一个边
                    count += 1
                    # 算方差, 做判断
                    # gg, var_1 = get_gg(mg, i+k-1, feats)   # 找到包含当前node的子图. ,慢. 放到这层循环里要跑更多的var. 越到后边,越慢
                    # # var_1 = 0.4
                    # # println("i:", i+k, " th:", th, " var:", var_1)
                    # if var_1 > var_th
                    #     println("\n over var_1..... ", var_1)
                    #     rem_edge!(mg, i+k-1, j[1])
                    #     flag = 1
                    #     break
                    # end
                end

                gg, var_1 = get_gg(mg, i+k-1, feats)   # 找到包含当前node的子图
                if var_1 >= var_th
                    println("\n over var_1..... ", var_1, " ,", th, " i:",i, " k:",k)
                    # 拆边
                    # for j in Tuple.(idx_1)   # 怎么批量加edges,自己写个循环吧
                    #     rem_edge!(mg, i+k-1, j[1])
                    #     gg, var_1 = get_gg(mg, i+k-1, feats)
                    #     if var_1 <= var_th
                    #         println("\n over over var_1..... ", var_1)
                    #         flag = 1
                    #         break
                    #     end
                    # end
                    flag = 1
                    break
                end
                # if flag == 1
                #     break
                # end
            end
        end
    end
    t2 = Dates.now()
    println("count: ", count)
    println("used: ", (t2 - t1).value/1000, "s, ", size_1)
    g_list = connected_components(mg)
    t3 = Dates.now()
    println("used: ", (t3 - t2).value/1000, "s, ")
    println("th:", th_max, "id_sum:", size(g_list)[1])    # 子图的数量
end


function get_gg(mg, i, feats)
    g_list = connected_components(mg)    # 获取连通子图.  慢 ??   0.05s * 46460000 = 26day
    # println("g_list:", size(g_list)[1])    # 子图的数量
    gg = 0
    var_1 = 1
    # println("mg:", mg)
    Threads.@threads for g in g_list    # 找到包含当前node的子图
        # sg, vmap = induced_subgraph(mg, g)   # 子图sg 里的节点编号是变了,变成子集的新节点了(1,2..).
        # println("g_1: ", join([g, collect(vertices(sg)), i, vmap], ", "))
        if i in g  # has_vertex
            # println("g_1: ", join([g, i, "..."], ", "))
            # gg = sg
            # nodes = collect(vertices(sg))   # 求这个子图的 var. 利用所有边长.
            # var = np.sum(np.var(feats[nodes], axis=0))
            @inbounds var_1 = sum(var(feats[g,:], dims=1,corrected=false))   # 快
            # var_1 = 0.4
            # println("sg: ", join([g, i, var_1], ", "))
            break
        end
    end
    return gg, var_1
end


function cluster_4()
    """
    动态阈值层次聚类

    """

    if Sys.iswindows()
        feats = npzread(raw"C:\zsz\ML\code\DL\face_cluster\face_cluster\tmp2\data\valse19.npy")
    else
        # feats = npzread("/data5/yongzhang/cluster/data/cluster_data/valse/valse_feat.npy")
        feats = npzread("/data5/yongzhang/cluster/data/cluster_data/ms1m/ms1m_part1_test_feat.npy")
    end

    size_1 = size(feats)[1]
    t1 = Dates.now()
    println("used: ", (t1 - t0).value/1000, "s, ", size_1)
    feats_1 = []
    batch = 1000
    count = 0
    th_max = 0.70   # 0.6
    th_min = 0.64  # 0.48
    step_th = 0.02
    var_th = 0.5  # 0.45
    nodes = []  # dict()
    clusters = Dict()
    for i in range(1, stop=size_1)    # n*(n-1)/2.   @showprogress  不能并行
        push!(feats_1, feats[i,1:end])
        # feats_3 = vcat((hcat(i...) for i in feats_1)...)  # 转换 shape
        feats_3 = vcat(feats_1...)   # 转换 shape
        feats_2 = feats[i, 1:end]
        cos =  feats_2 * feats_3'

        push!(nodes,i)
        clusters[i] = Cluster(i)
        clusters[i].add([i])
        th = th_max
        while th >= th_min
            idx = np.where(cos > th)  # 相似性搜索.  或者去rank
        end
        
    end
    t2 = Dates.now()
    println("count: ", count)
    println("used: ", (t2 - t1).value/1000, "s, ", size_1)
    g_list = connected_components(mg)
    t3 = Dates.now()
    println("used: ", (t3 - t2).value/1000, "s, ")
    println("th:", th_max, "id_sum:", size(g_list)[1])    # 子图的数量
end



# @time test_1()   
cluster_1_2()
# cluster_3()
# cluster_3_2()
# @profview cluster_1_2()
# ProfileView.svgwrite("profile_results.svg")



"""
export JULIA_NUM_THREADS=4
used: 558.827s  10min

图数据/图数据库 是存储在字典数据结构中的, 像mongodb. 读写,查询速度会快吗??

按对象类型各自单独存储, 节点数据存储在节点文件里, 边数据存储在边文件里, 索引数据存储在索引文件里. 
这样每个文件里, 数据就是同类型的, 可以高效存储和查询. 
节点数据: 有两种存储方式
1. 以字典方式存储在图里, 查询时候是按key-value方式查询, 字典查询很快. 
2. 以结构数据存储在图外, 查询时按索引位置查找. 

感觉在图里,也是全走的索引.

每个img跟所有img算cos, 找出cos>th的点,连边. 
每个img跟所有img算cos, 找出top_k点. 


"""




