# online HAC base on Transducers.   2021.1.16
using NPZ, JLD2
using Transducers
using Transducers: R_, start, next, complete, inner, xform, wrap, unwrap, wrapping
using Strs
include("milvus_api.jl")


struct HAC <: Transducer
    top_k::Int32  # rank top k
    th::Float32   # 聚类阈值
    batch_size::Int32   
end

HAC() = HAC(100, 0.5, 1000)  # 初始化结构体

function Transducers.start(rf::R_{HAC}, result)  
    hac = xform(rf)
    num = 0
    nodes = Dict()     # {"n_id0": {"n_id":"pid", "blur":0.3}}  只存代表点
    clusters = Dict()    # {"c_id0":{"c_id":"c_id0", "members": [], "c_size":5}  只存代表点
    collection_name = creat_collection("repo_test_3", 384)   # init index
    vectors = []  # 把一批的feat存到状态里. 为batch加的
    ids = []      # 把一批的id存到状态里

    private_state = (hac, num, nodes, clusters, collection_name, vectors, ids)   # 初始化私有状态

    result = wrap(rf, private_state, start(inner(rf), result))
    return result
end


function Transducers.next(rf::R_{HAC}, result, input)
    wrapping(rf, result) do private_state, iresult
        (hac, num, nodes, clusters, collection_name, vectors, ids) = private_state
        
        top_k = hac.top_k
        th = hac.th
        batch_size = hac.batch_size
        # println(f"\(top_k), \(th), \(batch_size)")

        num += 1
        # feat_1 = transform(input)   # 反序列化
        feat_1 = input   # 特征

        # init
        nodes[num] = num 
        clusters[num] = [num]

        # 调用api
        # vectors = [feat_1]
        # ids = [string(num)]
        push!(vectors, feat_1)   # 把一批的feat存到状态里. 为batch加的
        push!(ids, string(num))
        # push!(ids, num)
        # batch/window. 批处理. 是不是可以加个window op.
        if num % batch_size == 0
            # add_obj(collection_name, vectors, ids)   # add  慢
            println(f"======:\(num), \(length(ids)), \(size(vectors))")
            
            # insert_obj(collection_name, vectors, ids)   # add  慢
            # rank_result = search_obj(collection_name, vectors, top_k)   # search rank
            # dists, idxs = prcoess_results_3(rank_result, top_k)
            
            insert_obj_batch(collection_name, vectors, ids)
            dists, idxs = search_obj_batch(collection_name, vectors, top_k)

            batch = num ÷ batch_size - 1
            for i in 1:batch_size
                idx_1 = findall(dists[i,:] .> th)   # 返回的idx
                idx_y = idxs[i,:][idx_1]   # Tuple.(idx_1)

                for j in idx_y  # 遍历每个连接
                    union_2!(batch*batch_size+i, j, nodes, clusters)
                end
            end
            vectors = []
            ids = []
        end

        iinput = (hac, num, nodes)  # 就是输出. 状态
        iresult = next(inner(rf), iresult, iinput)
        return (hac, num, nodes, clusters, collection_name, vectors, ids), iresult
    end
end


function Transducers.complete(rf::R_{HAC}, result)
    _private_state, inner_result = unwrap(rf, result)
    result = complete(inner(rf), inner_result)
    return result
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


function test_1()
    # cluster online   2021.1.16
    println("test_1()")
    # 加载数据
    t0 = Dates.now()
    feats = npzread("/mnt/zy_data/data/longhu_1/feats.npy")
    feats_org = feats[1:195000, 1:end]
    feats = [feats_org[i,1:end] for i in 1:size(feats_org)[1]]
    size_1 = size(feats)[1]
    t1 = Dates.now()
    println(f"used: \((t1 - t0).value/1000) s, \(size_1), \(size(feats)), \(size(feats_org))")

    # stream pipeline
    # aa = collect(HAC(), feats)  # 获取每次的结果
    op_1 = HAC()
    aa = Transducers.foldl(right, op_1, Transducers.withprogress(feats; interval=1e-2))  
    # right: 获取最后的结果.  withprogress 用来显示进度(没起作用)
    
    hac, num, nodes = aa
    # println(f"\(num), \(nodes)")
    
    t2 = Dates.now()
    
    # 获取结果
    labels = values(nodes)
    println(f"img_sum:\(size_1), id_sum:\(length(Set(labels)))")
    println(f"used: \%.1f((t2 - t1).value/1000)s")
    # @save "nodes_1.jld2" labels

end


test_1()


#=
# online HAC base on Transducers.   2021.1.16  跑通了, 结果对齐了. 
batch_size 越大,milvus的速度越快,cpu消耗越小.  
最基本的层次聚类, 没有加其他的. 

base    milvus用的CPU 
th=0.5  img_sum:195000, id_sum:3721
used: 577.5s=9.6min

hac    milvus用的CPU  bs=100.  julia cpu:30%
img_sum: 195000, id_sum: 3723
used: 592.717s=10min

hac    milvus用的CPU  bs=1000   julia cpu:65%
img_sum:195000, id_sum:3721
used: 511.0s
img_sum:195000, id_sum:3721
used: 326.3s




=#

