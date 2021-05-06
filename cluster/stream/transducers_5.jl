# online HAC base on Transducers.   2021.1.16
using NPZ, JLD2, FileIO
using Transducers
using Transducers: R_, start, next, complete, inner, xform, wrap, unwrap, wrapping
using Strs, JSON3, Base64
using NearestNeighbors, Distances
using LinearAlgebra, Statistics
using PyCall
include("milvus_api.jl")



mutable struct Node <: Any
    n_id::Int32  # node id
    c_id::Int32  # cluster id
    obj_id::String  # object id
    blur::Float32 
    feature::Array 
    timestamp::Int64   # timestamp
    device_id::String  # device id
    img_url::String    # image url
    yaw::Float32 
    pitch::Float32 
    mask::Int32  # mask
    glass::Int32 #glass
    hat::Int32   #hat

end


struct HAC <: Transducer
    top_k::Int32  # rank top k
    th::Float32   # 聚类阈值
    batch_size::Int32   
end

HAC() = HAC(100, 0.5, 100)  # 初始化结构体

function Transducers.start(rf::R_{HAC}, result)  
    hac = xform(rf)
    num = 0
    nodes = Dict()     # {"n_id0": {"n_id":"pid", "blur":0.3}}  只存代表点
    clusters = Dict(0=>[])    # {"c_id0":{"c_id":"c_id0", "members": [], "c_size":5}  只存代表点
    collection_name = creat_collection("repo_test_3", 384)   # init index
    vectors = []  # 把一批的feat存到状态里. 为batch加的
    size_keynotes = 0      # 代表点数量

    private_state = (hac, num, nodes, clusters, collection_name, vectors, size_keynotes)   # 初始化私有状态
    # 状态里存储了 img_url, feature 比较大. 

    result = wrap(rf, private_state, start(inner(rf), result))
    return result
end


function Transducers.next(rf::R_{HAC}, result, input)
    wrapping(rf, result) do private_state, iresult
        (hac, num, nodes, clusters, collection_name, vectors, size_keynotes) = private_state
        
        top_k = hac.top_k
        th = hac.th
        batch_size = hac.batch_size

        num += 1
        node = Node(num, num, "", 1.0, [], 0, "", "", 0, 0, 1, 1, 1)   # init
        node = prase_json(input, node)   # 反序列化
        feat_1 = node.feature   # 特征

        # init. 存了所有点
        nodes[num] = node 
        clusters[num] = [num]

        push!(vectors, feat_1)   # 把一批的feat存到状态里. 为batch加的

        # batch/window. 批处理, 矩阵操作. 是不是可以加个window op.
        if num % batch_size == 0
            keynodes_feats = []
            keynodes_ids = []
            del_keynodes_ids = []
            println(f"======:\(num), \(size(vectors))")
            # insert_obj(collection_name, vectors, ids)   # add  慢
            
            # query  knn
            feats_1 = knn_feat(collection_name, vectors, num-batch_size)  # knn
            feats_2 = matix2array(feats_1)
            # feats_2 = vectors  # 不用knn
            # search top_k 
            dists_1, idxs_1 = rank_2(feats_2, top_k, num-batch_size)    # 在本批查询
            rank_result = search_obj(collection_name, feats_2, top_k)   # search rank
            dists_2, idxs_2 = prcoess_results_3(rank_result, top_k)
            
            dists = size(dists_2)[1] == 0 ? dists_1 : hcat(dists_1, dists_2) 
            idxs = size(idxs_2)[1] == 0 ? idxs_1 : hcat(idxs_1, idxs_2)
            # println(f"===:\(num), \(size(dists)), \(size(idxs))")

            batch = num ÷ batch_size - 1
            for i in 1:batch_size
                idx_1 = findall(dists[i,:] .> th)   # 返回的idx
                dists_y = dists[i,:][idx_1]  # cos 
                idx_y = idxs[i,:][idx_1]   # index.  Tuple.(idx_1)
                num_1 = batch*batch_size+i
                node_1 = nodes[num_1]

                quality_1 = -40<node_1.yaw<40  && -20<node_1.pitch<20 && node_1.mask<2
                # 质量差的丢掉, 放到废片簇0里
                if quality_1 && node_1.blur < 0.1
                    if num_1 in keys(clusters)
                        # println(f"\(num_1), \(size(clusters[0]))")
                        node_1.c_id = 0
                        append!(clusters[0], pop!(clusters, num_1))
                        continue
                    end
                end

                quality_2_1 = quality_1 && node_1.blur >= 0.1
                cos_1 = length(dists_y) > 1.0 ? dists_y[2] : 0.0
                if quality_2_1 && cos_1 < 0.95    # add
                    push!(keynodes_feats, feats_2[i])   # 代表点
                    push!(keynodes_ids, string(node_1.n_id))   # 代表点
                    size_keynotes += 1
                elseif quality_2_1 && cos_1 >= 0.95  # update
                    push!(keynodes_feats, feats_2[i])   # 代表点
                    push!(keynodes_ids, string(node_1.n_id))    # 代表点
                    push!(del_keynodes_ids, string(idx_y[2]))   # 要被删除的id
                end

                for j in 1: length(idx_y)  # 遍历每个连接
                    # node_1 = nodes[batch*batch_size+i]
                    idx_j = idx_y[j]
                    id_1 = node_1.c_id
                    node_2 = nodes[idx_j]
                    cos_1 = dists_y[j]  # 相似度
                    # println(f"batch:\(batch),i:\(i),j:\(j), idx_j:\(idx_j),\(cos_1), th:\(th)")
                    id_2 = node_2.c_id
                    if id_1 != 0 && id_2 != 0
                        union_2!(id_1, id_2, nodes, clusters)
                    end
                end
            end
            if length(keynodes_ids) > 0
                # println(size(keynodes_ids))
                insert_obj(collection_name, keynodes_feats, keynodes_ids)   # add  慢
            end
            if length(del_keynodes_ids) > 0
                delete_obj(collection_name, del_keynodes_ids)
            end
            vectors = []
        end

        iinput = (hac, num, nodes, size_keynotes)  # 就是输出. 状态
        iresult = next(inner(rf), iresult, iinput)
        return (hac, num, nodes, clusters, collection_name, vectors, size_keynotes), iresult
    end
end


function Transducers.complete(rf::R_{HAC}, result)
    _private_state, inner_result = unwrap(rf, result)
    result = complete(inner(rf), inner_result)
    return result
end


function union_2!(id_1, id_2, nodes, clusters)
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
            nodes[idx_].c_id = id_max
        end
        append!(clusters[id_max], pop!(clusters, id_min))  # 合并
    end

end


function prase_json(json_data, node)
    data = JSON3.read(json_data)  # string to dict

    node.blur = data["RawMessage"]["vseResult"]["RecFaces"][1]["Qualities"]["Blur"]
    node.obj_id = data["RawMessage"]["vseResult"]["RecFaces"][1]["Metadata"]["AdditionalInfos"]["FeatureID"]
    feature = data["RawMessage"]["vseResult"]["RecFaces"][1]["Features"]  # base64
    feature = base64decode(feature)
    feature = reinterpret(Float32, feature)
    node.feature = feature

    node.img_url = data["RawMessage"]["vseResult"]["RecFaces"][1]["Img"]["Img"]["URI"]

    node.yaw = data["RawMessage"]["vseResult"]["RecFaces"][1]["Qualities"]["Yaw"]
    node.pitch = data["RawMessage"]["vseResult"]["RecFaces"][1]["Qualities"]["Pitch"]
    
    attributes = data["RawMessage"]["vseResult"]["RecFaces"][1]["Attributes"]
    face_att_key_dict = Dict(1=>"age", 3=>"glass", 4=>"hat", 6=>"mask", 16=>"gender", 19=>"other", 5=>"helmet",
                                 28=>"肤色", 29=>"人脸表情", 30=>"人脸颜值")
    att_dict = Dict()
    for att in attributes
        att_id = att["AttributeId"]
        att_name = face_att_key_dict[att_id]
        att_value = att["ValueId"]
        att_confidence = att["Confidence"]
        att_dict[att_name] = Dict("value"=> att_value, "confidence"=> att_confidence)
    end

    node.mask = Int32(att_dict["mask"]["value"])
    node.glass = Int32(att_dict["glass"]["value"])
    node.hat = Int32(att_dict["hat"]["value"])
    
    node.timestamp = data["RawMessage"]["vseResult"]["RecFaces"][1]["Metadata"]["Timestamp"]
    node.device_id = data["RawMessage"]["vseResult"]["RecFaces"][1]["Metadata"]["AdditionalInfos"]["UniqueSensorId"]

    # println(node)

    return node
end

function rank_1(feats, n)
    feats = vcat((hcat(i...) for i in feats)...)
    size_1 = size(feats)[1]
    dists = feats * feats'   # cos相似度
    
    idxs = zeros(Int32, (size_1, size_1))
    for i in 1:size_1
        idxs[i, :] = collect(n+1:n+size_1)
    end

    return dists, idxs
end

function rank_2(feats, top_k, n)
    # knn, top_k
    feats = vcat((hcat(i...) for i in feats)...)
    size_1 = size(feats)[1]
    x = feats
    X = transpose(x)  # 矩阵转置, 也可以用 x'. 必须. 垃圾
    X = convert(Array, X)
    # println("size(x):", size(X), " ", typeof(X))
    top_k = top_k == 100 ? top_k-1 : top_k
    gallery = X
    query = X
    brutetree = BruteTree(gallery, Euclidean())  # 暴力搜索树, 只支持Euclidean() 不支持 CosineDist(), 但是可以转换
    # kdtree = KDTree(gallery, leafsize=4)   # 同index.add(gallery) 
    idxs, dists = knn(brutetree, query, top_k, true)  # 单线程的, 很慢.  # query查询.  
    dists = vcat((hcat(i...) for i in dists)...)  # 转换 shape
    idxs = vcat((hcat(i...) for i in idxs)...)  # 转换 shape
    # 后处理
    dists = 1 .- dists ./ 2
    idxs = idxs .+ n
    return dists, idxs
end

function knn_feat(collection_name, feats, n)
    # knn feat merge
    # feats = mean(top_5 && cos>0.5)(feats)
    top_k = 5
    knn_th = 0.5
    vectors = feats   # 100*384
    # println(f"---vectors[1]:\(vectors[1])")

    dists_1, idxs_1 = rank_2(vectors, top_k, n)  # 在本批查询
    rank_result = search_obj(collection_name, vectors, 5)   # search rank.  
    dists_2, idxs_2 = prcoess_results_3(rank_result, 5)  
    
    dists = size(dists_2)[1] == 0 ? dists_1 : hcat(dists_1, dists_2)  # 合并  100*10
    idxs = size(idxs_2)[1] == 0 ? idxs_1 : hcat(idxs_1, idxs_2)

    feats_1 = vcat((hcat(i...) for i in vectors)...)   # [[1,2,4],[2,4,5]]-> [1 2 4; 2 4 5]
    feats_2 = zeros(size(feats_1))
    size_1 = size(dists)
    for i in 1:size_1[1]
        dist = dists[i, :]
        idx = idxs[i, :]
        idx_1 = partialsortperm(dist, 1:top_k, rev=true)  # top_5
        dist_sort = dist[idx_1]
        # println(f"top_5 dist: \(n+i), \(dist), \(dist_sort), \(idx_1)")
        idx_2 = idx_1[dist_sort .> knn_th]   # cos>0.5
        idx_org = idx[idx_2]
        
        idx_org_1 = idx_org[idx_org.>n] .- n  # 本批次
        idx_org_2 = idx_org[idx_org.<=n]  # 历史库里的
        # println(f"\(n+i), \(idx),\(idx_2), \(idx_org), \(idx_org_1), \(idx_org_2)")
        
        feat_1 = feats_1[idx_org_1, :]

        if length(idx_org_2) > 0
            feat_2 = get_feat(collection_name, idx_org_2)  # 从历史库里取出feat
            if length(feat_2) > 0
                tmp_feats = vcat(feat_1, feat_2)
            else
                tmp_feats = feat_1
            end
        else
            tmp_feats = feat_1
            feat_2 = []
        end
        # println(size(feat_1), ",", size(feat_2), ",", size(tmp_feats), ",", size(mean(tmp_feats, dims=1)))
        
        feats_2[i, 1:end] = normalize(mean(tmp_feats, dims=1))

    end

    # feats_2 = LinearAlgebra.normalize(feats_2)  # 归一化.  有问题

    return feats_2
end

function max_k(data, k)
    # top_k, 取idx
    size_1 = size(data)
    idxs = zeros(size_1)
    for i in 1:size_1[1]
        a = data[i, :]
        idx = partialsortperm(a, 1:k, rev=true)
        a_sort = a[idx]
        data[i, :] = a_sort
        idxs[i, :] = idx
    end
    return data, idxs
end

function matix2array(b)
    c = []
    for i in 1:size(b)[1]
        c_1 = Array{Float32, 1}(b[i,:])
        push!(c, c_1)
    end
    return c
end


function test_1(input_path, out_path)
    # cluster online   2021.1.16
    println("test_1()")
    # 加载数据
    t0 = Dates.now()
    size_1 = 0
    input_json = open(input_path)  # input_languang_5_2
    t1 = Dates.now()
    # println(f"used: \((t1 - t0).value/1000) s, \(size_1), \(size(feats)), \(size(feats_org))")

    # stream pipeline
    op_hac = HAC() 
    # aa = Transducers.foldxt(right, op_hac, eachline(input_json))    # foldxt 并行用不了  foldl
    aa = Transducers.foldxt(right, eachline(input_json) |> op_hac |> collect)
    
    hac, num, nodes, size_keynotes = aa
    
    # 获取结果
    labels = [node.c_id for node in values(nodes)]
    t2 = Dates.now()
    id_sum = length(Set(labels))
    println(f"img_sum:\(length(labels)), id_sum:\(id_sum), keynotes_sum:\(size_keynotes), \%.1f(size_keynotes/id_sum)img/id")
    println(f"used: \%.1f((t2 - t1).value/1000)s")
    
    # 结果保存和评估
    f_out = open(out_path, "w")
    for node in values(nodes)
        ss = f"\(node.obj_id),\(node.c_id)\n"
        write(f_out, ss)
    end
    # @save f"nodes_1.jld2" labels
    file_name = basename(out_path)
    eval_1(file_name)   # 评估
end


function eval_1(file_name)
    pushfirst!(PyVector(pyimport("sys")."path"), "")

    py"""
    import os
    import numpy as np
    import pandas as pd
    from utils import eval_1
 
    dir_1 = "/data2/zhangyong/data/pk/pk_13/output_1"
    cluster_path = os.path.join(dir_1, "out_1", $file_name)  # out_1_21.csv
    labels_pred_df = pd.read_csv(cluster_path, names=["obj_id", "person_id"])
    
    # merged_all_out_1_1_1_21  merged_all_out_1_1_1_9
    gt_path = os.path.join(dir_1, "merged_all_out_1_1_1_21-small_1.pkl")
    gt_sorted_df = pd.read_pickle(gt_path)

    labels_true, labels_pred = eval_1.align_obj(gt_sorted_df, labels_pred_df)

    print(cluster_path)
    p_waste_id = 0
    metric, info = eval_1.eval(labels_true, labels_pred, p_waste_id, is_show=False)
    print(info)
    """
end


input_path = "/data2/zhangyong/data/pk/pk_13/input/input_languang_5_2.json"
out_path = "/data2/zhangyong/data/pk/pk_13/output_1/out_1/out_1_31.csv"
test_1(input_path, out_path)


#=
export JULIA_NUM_THREADS=4

# online HAC base on Transducers.   2021.1.16  跑通了, 结果对齐了. 
batch_size 越大,milvus的速度越快,cpu消耗越小.  

----------------------------------------
TODO:
0. 加多维信息   OK
0. 加同镜,跨镜 多时空阶段聚类
1. 加代表点[OK], 代表点更新  OK
2. 质量加权动态阈值
3. 加knn feat.  OK
4. 接入kafka数据源, 超内存的数据源, 流式的dataloader. OK
5. 加窗口

加速: 
1. fse/milvus
2. 并行

=#

