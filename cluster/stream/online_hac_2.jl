# online hac based on Transducers.  2020.11.18
using Transducers
using Transducers: R_, start, next, complete, inner, xform, wrap, unwrap, wrapping
using NPZ, JLD2
using ProgressMeter
include("milvus_api.jl")



# 定义一个类, 并且init
struct Hac_1 <: Transducer
    top_k::Int64
    th::Float64
    batch_size::Int64
end

Hac_1() = Hac_1(10, 0.5, 10)


# 定义states, init states
function Transducers.start(rf::R_{Hac_1}, result)
    num = 0
    nodes = Dict()
    clusters = Dict()
    collection_name = creat_collection("repo_test_3", 384)  # init index
    vectors = []
    ids = []

    private_states = (num, nodes, clusters, collection_name, vectors, ids)
    return wrap(rf, private_states, start(inner(rf), result))
end


function Transducers.next(rf::R_{Hac_1}, result, input)
    wrapping(rf, result) do states, iresult
        (num, nodes, clusters, collection_name, vectors, ids) = states
        num += 1
        feat_1 = input
        # init
        nodes[num] = num
        clusters[num] = [num]

        # 调用api
        push!(vectors, feat_1)
        push!(ids, string(num))
        # push!(ids, num)
        hac = xform(rf)
        batch_size = hac.batch_size
        # batch/window. 批处理. 是不是可以加个window op 
        if num % batch_size == 0
            println(length(vectors), length(ids))
            insert_obj(collection_name, vectors, ids)   # add  慢
            rank_result = search_obj(collection_name, vectors, hac.top_k)   # search rank
            dists, idxs = prcoess_results_3(rank_result, hac.top_k)
            vectors = []
            ids = []

            batch = num ÷ batch_size - 1
            for i in 1:batch_size
                idx_1 = findall(dists[i,:] .> hac.th)
                idx_y = idxs[i,:][idx_1]   # Tuple.(idx_1)

                for j in idx_y  # 遍历每个连接
                    union_2!(batch*batch_size+i, j, nodes, clusters)
                end
            end
        end
        iinput = input
        iresult = next(inner(rf), iresult, iinput)
        return (num, nodes, clusters, collection_name, vectors, ids), iresult
    end
end


function Transducers.complete(rf::R_{Hac_1}, result)
    _private_state, inner_result = unwrap(rf, result)
    return complete(inner(rf), inner_result)
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


function test_4()
    # cluster online   2020.10.18
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
    # Hac_1() = Hac_1(100, 0.5, 100)
    
    b = nothing
    @showprogress for i in 1:size_1
        x1 = feats[i,1:end]
        # println(typeof(x1))
        # b = fit!(op, x1)   # x1 要是更复杂的对象,包括质量和时空.
        b = collect(Hac_1(), x1)
        println(b)
    end
    t2 = Dates.now()
    
    labels = values(op.nodes)
    println(size_1, ", id:", length(Set(labels)))
    # println(op.nodes)
    println("used: ", (t2 - t1).value/1000, "s")

    # @save "nodes_1.jld2" labels

end


test_4()



#=
online cluster base on Transducers. 2020.11.18
实现了基本框架, 但没跑通


=#
