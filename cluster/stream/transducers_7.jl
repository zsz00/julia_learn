# online cluster base on Transducers.   2021.1.30
using NPZ, JLD2, FileIO
using Transducers
using Transducers: R_, start, next, complete, inner, xform, wrap, unwrap, wrapping
using Strs, JSON3, Base64
using NearestNeighbors, Distances
using LinearAlgebra, Statistics
using PyCall
using ThreadsX
using Folds, FoldsThreads
using BangBang  # for `push!!`
using SimilaritySearch
using Strs
include("milvus_api.jl")
include("keyby.jl")
# include("faiss_api.jl")


mutable struct Node <: Any
    n_id::String  # node id
    c_id::String  # cluster id
    obj_id::String  # object id
    blur::Float32 
    feature::Array 
    timestamp::Float64   # timestamp
    device_id::String  # device id
    img_url::String    # image url
    yaw::Float32 
    pitch::Float32 
    mask::Int32  # mask
    glass::Int32 #glass
    hat::Int32   #hat

end

mutable struct Cluster <: Any
    c_id::String  # cluster id
    c_size::Int32 # cluster size
    c_key_size::Int32 # cluster keypoints size
    c_members::Array  # cluster members
    t_ctime::Int32 # cluster create Timestamp 
    c_utime::Int32 # cluster upate Timestamp 
end

mutable struct Track <: Any
    t_id::Int32  # tarcking id
    t_size::Int32 # tarcking size
    t_key_size::Int32 # tarcking keypoints size
    t_members::Array  # tarcking members
    t_ctime::Int32 # tarcking create Timestamp 
    t_utime::Int32 # tarcking upate Timestamp
end

# --------------------------------------------------------------
# 同镜 
struct Spacetime1_Cluster <: Transducer 
end

function Transducers.start(rf::R_{Spacetime1_Cluster}, result)  
    top_k = 100  # rank top k
    th = 0.45   # 聚类阈值
    batch_size = 100  
    num = 0
    nodes = Dict()     # 节点信息.  最好只存代表点
    clusters = Dict("0"=>Cluster("0", 0, 0, [], 0, 0))    # 同镜中簇信息 
    vectors = []  # 把一批的feat存到状态里. 为batch加的
    ids = []
    size_keynotes = 0      # 代表点数量

    private_state = (top_k, th, batch_size, num, nodes, clusters, vectors, ids, size_keynotes)  # 初始化私有状态
    result = wrap(rf, private_state, start(inner(rf), result))
    return result
end

function Transducers.next(rf::R_{Spacetime1_Cluster}, result, input)
    wrapping(rf, result) do private_state, iresult
        (top_k, th, batch_size, num, nodes, clusters, vectors, ids, size_keynotes) = private_state

        # num += 1
        device_id = input[1]  # input是Pair{String,Node}
        node = input[2]
        # println(f"device_id:\(device_id), \ninput: \(input)") 
        num = node.n_id
        feat_1 = node.feature   # 特征
        track = Cluster(num, 1, 1, [num], 0, 0)

        # init. 存了所有点
        nodes[num] = node 
        clusters[num] = track

        # # 到达定时器时清空
        if length(ids)>0 && node.timestamp - nodes[ids[end]].timestamp > 60000
            vectors = []
        end

        quality_1 = -40<node.yaw<40  && -20<node.pitch<20 && node.mask<2
        # 质量差的丢掉, 放到废片簇0里
        if quality_1 && node.blur < 0.1  # 0.15
            if num in keys(clusters)  # 注意:此处不能用c_id_1 
                append!(clusters["0"].c_members, pop!(clusters, num).c_members)  # 按道理只一个member
                node.c_id = "0"   # 只一个member的c_id. 完全的应该是改全部的members
                # continue
            end
        end

        push!(vectors, feat_1)   # 把一批的feat存到状态里. 为batch加的
        push!(ids, num)
        # println(f"======:device_id:\(device_id[1:8]), n_id:\(num[1:8]), \(size(vectors))")

        # 不加batch. 是不是可以加个window op. key by(track_id)
        # query 
        gallery = vcat((hcat(i...) for i in vectors)...)
        query = vcat((hcat(i...) for i in [feat_1])...)
        # search top_k 
        dists_1, idxs_1 = rank_3(gallery, query, ids, top_k)    # 用的NN,慢, 改为faiss的
        # println(f"===:\(num), \(size(dists)), \(size(idxs))")

        idx_1 = findall(dists_1 .> th)   # 返回的idx
        dists_y = dists_1[idx_1]  # cos 
        idx_y = idxs_1[idx_1]   # index. 
        node_1 = nodes[num]

        # 聚类
        for j in 1: length(idx_y)  # 遍历每个连接
            id_1 = node_1.c_id

            idx_j = idx_y[j]
            node_2 = nodes[idx_j]
            id_2 = node_2.c_id
            cos_1 = dists_y[j]  # 相似度
            # println(f"batch:\(batch),i:\(i),j:\(j), idx_j:\(idx_j),\(cos_1), th:\(th)")
            if id_1 != "0" && id_2 != "0" && id_1 in keys(clusters) && id_2 in keys(clusters)
                union_2!(id_1, id_2, nodes, clusters)
            end
        end

        output = node_1  # , nodes, device_id  # 就是输出. 状态
        iresult = next(inner(rf), iresult, output)
        private_state = (top_k, th, batch_size, num, nodes, clusters, vectors, ids, size_keynotes)
        return private_state, iresult
    end  # do end
end

function Transducers.complete(rf::R_{Spacetime1_Cluster}, result)
    _private_state, inner_result = unwrap(rf, result)
    result = complete(inner(rf), inner_result)
    return result
end

# --------------------------------------------------------------
# 全局
struct HAC <: Transducer
    top_k::Int32  # rank top k
    th::Float32   # 聚类阈值
    batch_size::Int32   
end

HAC() = HAC(100, 0.5, 1000)  # 初始化结构体

function Transducers.start(rf::R_{HAC}, result)  
    hac = xform(rf)
    num = 0
    nodes = Dict()     # 节点信息.  最好只存代表点
    clusters = Dict("0"=>Cluster("0", 0, 0, [], 0, 0))    # 簇信息 
    tracks = Dict()    # 跟踪信息
    collection_name = creat_collection("repo_test_3", 384)   # init index
    vectors = []  # 把一批的feat存到状态里. 为batch加的
    ids = []
    size_keynotes = 0      # 代表点数量

    private_state = (hac, num, nodes, clusters, collection_name, vectors, ids, size_keynotes)   # 初始化私有状态
    # 状态里存储了 img_url, feature 比较大. 

    result = wrap(rf, private_state, start(inner(rf), result))
    return result
end

function Transducers.next(rf::R_{HAC}, result, input)
    wrapping(rf, result) do private_state, iresult
        (hac, num, nodes, clusters, collection_name, vectors, ids, size_keynotes) = private_state  # 本op的state

        top_k = hac.top_k
        th = hac.th
        batch_size = hac.batch_size
        num += 1

        # input是PrivateState对象, 可获得 前op的state和result
        # top_k_st1, th_st1, batch_size_st1, num_st1, nodes_st1, clusters_st1, vectors_st1, ids_st1, size_keynotes_st1 = input.state  # state
        # node_st1 = input.result  # input.result    input
        node_st1 = input   # # 无同镜
        # println(f"node:\(node_st1)")
        node = node_st1
        feat_1 = node.feature   # 特征
        n_id = node.n_id
        c_id = node.c_id
        # cluster = clusters_st1[c_id]
        cluster = Cluster(c_id, 1, 1, [n_id], 0, 0)    # 无同镜

        # init. 存了所有点
        nodes[n_id] = node 
        clusters[c_id] = cluster
 
        push!(vectors, feat_1)   # 把一批的feat存到状态里. 为batch加的
        push!(ids, n_id)

        # batch/window. 批处理, 矩阵操作. 是不是可以加个window op.
        if num % batch_size == 0
            keynodes_feats = []
            keynodes_ids = []
            del_keynodes_ids = []
            println(f"======:\(num), \(size(vectors))")

            # query knn
            # query 
            gallery = vectors  # vcat((hcat(i...) for i in vectors)...)  # Vectors -> Matrix
            query = gallery
            # ids = vcat((hcat(i...) for i in ids)...)
            feats_1 = knn_feat(collection_name, gallery, query, num-batch_size)  # knn
            feats_2 = matix2Vectors(feats_1)   # knn feats
            # feats_2 = vectors  # 不用knn
            # search top_k 
            # dists_1, idxs_1 = rank_2(feats_1, top_k, num-batch_size)    # 在本批查询, NN
            # dists_1, idxs_1 = rank_1(feats_1, top_k, num-batch_size)    # 在本批查询, faiss
            # dists_1, idxs_1 = rank_3(gallery, query, ids, top_k)
            dists_1, idxs_1 = rank_4(feats_2, feats_2, top_k, num-batch_size)  # 在本批查询
            # println(feats_2)
            rank_result = search_obj(collection_name, feats_2, top_k)  # search rank in milvus/fse 
            dists_2, idxs_2 = prcoess_results_3(rank_result, top_k)
            # dists_2, idxs_2 = search_obj_batch(collection_name, feats_2, top_k)
            
            dists = size(dists_2)[1] == 0 ? dists_1 : hcat(dists_1, dists_2) 
            idxs = size(idxs_2)[1] == 0 ? idxs_1 : hcat(idxs_1, idxs_2)
            # println(f"===:\(num), \(size(dists)), \(size(idxs))")

            batch = num ÷ batch_size - 1
            for i in 1:batch_size
                idx_1 = findall(dists[i,:] .> th)   # 返回的idx
                dists_y = dists[i,:][idx_1]  # cos 
                idx_y = idxs[i,:][idx_1]   # index.  Tuple.(idx_1)
                num_1 = batch*batch_size+i
                n_id_1 = ids[num_1]   # 获取真obj_id
                node_1 = nodes[n_id_1]
                c_id_1 = node_1.c_id  # 会传递到nodes吗?会. 并且 未来的也会被下面的union_2 改变c_id

                quality_1 = -40<node_1.yaw<40  && -20<node_1.pitch<20 && node_1.mask<2
                # 质量差的丢掉, 放到废片簇0里
                if quality_1 && node_1.blur < 0.1  # 0.15
                    if n_id_1 in keys(clusters)  # 注意:此处不能用c_id_1 
                        append!(clusters["0"].c_members, pop!(clusters, n_id_1).c_members)  # 按道理只一个member
                        node_1.c_id = "0"   # 只一个member的c_id. 完全的应该是改全部的members
                        continue
                    end
                end

                quality_2_1 = quality_1 && node_1.blur >= 0.1
                cos_1 = length(dists_y) > 1 ? dists_y[2] : 0.0
                if quality_2_1 && cos_1 < 0.95    # add  0.95
                    push!(keynodes_feats, feats_2[i])   # 代表点
                    push!(keynodes_ids, string(num_1))   # 代表点  node_1.n_id
                    size_keynotes += 1
                elseif quality_2_1 && cos_1 >= 0.95  # update
                    push!(keynodes_feats, feats_2[i])   # 代表点
                    push!(keynodes_ids, string(num_1))    # 代表点
                    push!(del_keynodes_ids, string(idx_y[2]))   # 要被删除的id
                end

                # println(f"batch:\(batch),i:\(i), keynodes_feats:\(size(keynodes_feats)), keynodes_ids:\(size(keynodes_ids))")

                for j in 1: length(idx_y)  # 遍历每个连接
                    id_1 = nodes[n_id_1].c_id 

                    idx_j = idx_y[j]
                    n_id_2 = ids[idx_j]   # 也有低质量的, 需要控制下
                    node_2 = nodes[n_id_2]
                    id_2 = node_2.c_id
                    cos_1 = dists_y[j]  # 相似度
                    if !(id_1 in keys(clusters))
                        println(f"id_1: batch:\(batch),i:\(i),j:\(j), id_1:\(id_1), \(node_1.blur),\(node_1.n_id)")
                        continue
                    end
                    if !(id_2 in keys(clusters))
                        println(f"id_2: batch:\(batch),i:\(i),j:\(j), id_2:\(id_2), \(node_2.blur), \(size(ids)), \(length(clusters))")
                        continue
                    end

                    if id_1 != "0" && id_2 != "0"
                        union_2!(id_1, id_2, nodes, clusters)
                    end
                end
            end
            if length(keynodes_ids) > 0
                # println(f"\(collection_name), keynodes_feats:\(size(keynodes_feats)), keynodes_ids:\(size(keynodes_ids))")
                insert_obj(collection_name, keynodes_feats, keynodes_ids)   # add  慢
            end
            if length(del_keynodes_ids) > 0
                delete_obj(collection_name, del_keynodes_ids)
            end
            vectors = []
            # ids = []
        end

        iinput = (hac, num, nodes, size_keynotes)  # 就是输出. 状态
        iresult = next(inner(rf), iresult, iinput)
        return (hac, num, nodes, clusters, collection_name, vectors, ids, size_keynotes), iresult
    end   # do end
end

function Transducers.complete(rf::R_{HAC}, result)
    _private_state, inner_result = unwrap(rf, result)
    result = complete(inner(rf), inner_result)
    return result
end

# --------------------------------------------------------------
# 辅助函数
function union_2!(id_1, id_2, nodes, clusters)
    if id_1 == id_2  # 查
        return
    else
        # 更新 set
        if length(clusters[id_1].c_members) >= length(clusters[id_2].c_members)  # 更新小的cluster到大的中
            id_max, id_min = id_1, id_2
        else
            id_max, id_min = id_2, id_1
        end

        for idx_ in clusters[id_min].c_members  # 把id_2的转为id_1
            if idx_ in keys(nodes)
                nodes[idx_].c_id = id_max
            else
                println(f"union_2: \(idx_) no in nodes")
            end
        end
        append!(clusters[id_max].c_members, pop!(clusters, id_min).c_members)  # 合并
    end

end

function prase_json(json_data)
    data = JSON3.read(json_data)  # string to dict
    node = Node("0", "0", "", 1.0, [], 0, "", "", 0, 0, 1, 1, 1)   # init node
    
    node.blur = data["RawMessage"]["vseResult"]["RecFaces"][1]["Qualities"]["Blur"]
    feature_id = data["RawMessage"]["vseResult"]["RecFaces"][1]["Metadata"]["AdditionalInfos"]["FeatureID"]
    feature = data["RawMessage"]["vseResult"]["RecFaces"][1]["Features"]  # base64
    feature = base64decode(feature)
    feature = reinterpret(Float32, feature)
    node.feature = feature
    node.obj_id = feature_id
    node.n_id = feature_id
    node.c_id = feature_id
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
    
    node.timestamp = data["RawMessage"]["vseResult"]["RecFaces"][1]["Metadata"]["Timestamp"]/(10^13)  # 13位,毫秒.
    node.device_id = data["RawMessage"]["vseResult"]["RecFaces"][1]["Metadata"]["AdditionalInfos"]["UniqueSensorId"]

    # println(node)

    return node
end

function rank_1(feats, top_k, n)
    # base faiss_python
    query = np.array(feats)
    gallery = query
    feat_dim = query.shape[1]
    index = create_index(feat_dim, "")
    dists, idxs = rank(index, query, gallery, topk=top_k)
    idxs = idx .+ n
    return dists, idxs
end

function rank_2(feats, top_k, n)
    # base NearestNeighbors.jl, top_k
    # feats = vcat((hcat(i...) for i in feats)...)
    X = transpose(feats)  # 矩阵转置, 也可以用 x'. 必须. 垃圾
    X = convert(Array, X)
    # println("size(x):", size(X), " ", typeof(X))
    
    gallery = X
    query = X
    # top_k = top_k == 100 ? top_k-1 : top_k
    top_k = top_k >=size(gallery)[2] ? size(gallery)[2] : top_k
    brutetree = BruteTree(gallery, Euclidean())  # 暴力搜索树, 只支持Euclidean()不支持CosineDist(),但是可以转换. 没有增量add方式
    # kdtree = KDTree(gallery, leafsize=4)   # 同index.add(gallery) 
    idxs, dists = knn(brutetree, query, top_k, true)  # 单线程的, 很慢.  # query top_k  
    dists = vcat((hcat(i...) for i in dists)...)  # 转换 shape
    idxs = vcat((hcat(i...) for i in idxs)...)  # 转换 shape
    # 后处理
    dists = 1 .- dists ./ 2
    idxs = idxs .+ n
    return dists, idxs
end

function rank_3(gallery, query, ids, top_k)
    # knn, top_k. 基于NN的. 内存式,小批量适用. 支持id
    gallery = convert(Array, transpose(gallery))  # 矩阵转置, 也可以用 x'. 必须. 垃圾
    # println("size(gallery):", size(gallery), " ", typeof(gallery))
    top_k = top_k >=size(gallery)[2] ? size(gallery)[2] : top_k
    
    query = convert(Array, transpose(query))  # 矩阵转置, 也可以用 x'. 必须. 垃圾
    # println("size(query):", size(query), " ", typeof(query))

    brutetree = BruteTree(gallery, Euclidean())  # 暴力搜索树, 只支持Euclidean()不支持CosineDist(),但是可以转换. 没有增量add方式
    # kdtree = KDTree(gallery, leafsize=4)   # 同index.add(gallery) 
    idxs, dists = knn(brutetree, query, top_k, true)  # 单线程的, 很慢.  # query top_k  
    dists = vcat((hcat(i...) for i in dists)...)  # 转换 shape
    idxs = vcat((hcat(i...) for i in idxs)...)  # 转换 shape
    # println(f"\(size(idxs)), \(size(idxs)), \(size(ids)), \(ids)")
    idxs = ids[idxs]
    # idxs = vcat((hcat(i...) for i in idxs)...)  # 转换 shape
    # println(f"\(size(idxs)), \(size(idxs)), \(size(ids)), \(ids)")
    # 后处理
    dists = 1 .- dists ./ 2
    idxs = idxs
    return dists, idxs
end

function rank_4(gallery, query, top_k, n)
    # base SimilaritySearch.jl
    # feats = matix2Vectors(feats)
    println(size(query), typeof(query))

    top_k_1 = size(gallery)[1]

    t0 = Dates.now()
    index = ExhaustiveSearch(CosineDistance(), gallery)   # gallary是Vectors,不支持增量add
    out = [search(index, q, KnnResult(top_k_1)) for q in query]
    # println(length(out), out)
    dists, idxs = prcoess_ss(out, top_k_1)  # 解析

    dists = dists[:,1:top_k]
    idxs = idxs[:,1:top_k]
    println(size(dists), typeof(dists))

    # 后处理
    idxs = idxs .+ n
    return dists, idxs
end

function prcoess_ss(results, topk)
    # println(length(results))
    size = length(results)
    dists = zeros(Float32, (size, topk))
    idxs = zeros(Int32, (size, topk))
    for (i, p) in enumerate(results)
        for (j, pp) in enumerate(p)
            # println(f"\(i), \(j), \(pp.id), \(pp.dist)")
            dists[i, j] = pp.dist
            idxs[i, j] = pp.id
        end
    end
    dists = reverse(dists, dims=2)
    idxs = reverse(idxs, dims=2)
    return dists, idxs
end

function knn_feat(collection_name, gallery, query, n)
    # knn feat merge
    # knn_feats = mean(top_5 && cos>0.5)(feats)
    top_k = 5
    knn_th = 0.5

    # dists_1, idxs_1 = rank_1(query, top_k, n)  # 在本批查询   faiss
    # dists_1, idxs_1 = rank_2(query, top_k, n)  # 在本批查询
    # dists_1, idxs_1 = rank_3(gallery, query, ids, top_k)  # 在本批查询
    dists_1, idxs_1 = rank_4(query, query, top_k, n)  # 在本批查询

    query_1 = query  # matix2Vectors(query)
    rank_result = search_obj(collection_name, query_1, 5)   # search top5 in milvus/fse 
    dists_2, idxs_2 = prcoess_results_3(rank_result, 5)  
    
    dists = size(dists_2)[1] == 0 ? dists_1 : hcat(dists_1, dists_2)  # 合并  100*10
    idxs = size(idxs_2)[1] == 0 ? idxs_1 : hcat(idxs_1, idxs_2)

    feats_1 = vcat((hcat(i...) for i in query)...)   # Vectors -> Matrix
    feats_2 = zeros(Float32, size(feats_1))
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

    # feats_2 = LinearAlgebra.normalize(feats_2)  # 统一归一化. 有问题

    return feats_2
end

function knn_feat_2(index, gallery, query, n)
    # knn feat merge
    # knn_feats = mean(top_5 && cos>0.5)(feats)
    top_k = 5
    knn_th = 0.5
    # vectors = query   # 100*384
    # println(f"---vectors[1]:\(vectors[1])")

    # dists_1, idxs_1 = rank_1(query, top_k, n)  # 在本批查询   faiss
    dists_1, idxs_1 = rank_2(query, top_k, n)  # 在本批查询
    # dists_1, idxs_1 = rank_3(gallery, query, ids, top_k)  # 在本批查询
    # query_1 = matix2array(query)
    query_1 = query
    gallery = []
    dists_2, idxs_2 = rank(index, query, gallery; topk=5)
    # rank_result = search_obj(collection_name, query_1, 5)   # search top5 in milvus/fse 
    # dists_2, idxs_2 = prcoess_results_3(rank_result, 5)  
    
    dists = size(dists_2)[1] == 0 ? dists_1 : hcat(dists_1, dists_2)  # 合并  100*10
    idxs = size(idxs_2)[1] == 0 ? idxs_1 : hcat(idxs_1, idxs_2)

    feats_1 = query  # vcat((hcat(i...) for i in vectors)...)   # [[1,2,4],[2,4,5]]-> [1 2 4; 2 4 5]
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

    # feats_2 = LinearAlgebra.normalize(feats_2)  # 统一归一化. 有问题

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

function matix2Vectors(b)
    c = []
    for i in 1:size(b)[1]
        c_1 = Array{Float32, 1}(b[i,:])
        push!(c, c_1)
    end
    return c
end

# --------------------------------------------------------------
# 主函数
function test_1(input_path, out_path)
    # cluster online   2021.1.16
    println("test_1()")
    println("nthreads:", Threads.nthreads())
    # load data
    input_json = open(input_path)
    t1 = Dates.now()
    # stream pipeline
    op_st_1 = Spacetime1_Cluster()  # 同镜, on a camera
    op_hac = HAC()   # 全局, on all camera
    
    aa = Transducers.foldl(right, eachline(input_json) |> Map(prase_json) |> op_hac|> collect)
    # aa = Transducers.foldl(right, eachline(input_json) |> Map(prase_json) |> KeyBy((x -> x.device_id), op_st_1)|> op_hac |> collect )
    # KeyBy((x -> x.device_id), op_st_1) |>  |> op_hac    foldl foldxt
    # Folds.reduce((right, eachline(input_json) |> Map(prase_json) |> collect), NondeterministicEx()) 

    hac, num, nodes, size_keynotes = aa

    # 获取结果
    labels = [node.c_id for node in values(nodes)]
    t2 = Dates.now()
    id_sum = length(Set(labels))
    println(f"img_sum:\(length(labels)), id_sum:\(id_sum), keynotes_sum:\(size_keynotes), \%.1f(size_keynotes/id_sum)img/id")
    used_time = (t2 - t1).value/1000
    println(f"used: \%.1f(used_time)s=\%.1f(used_time/60)min")
    
    # 结果保存和评估
    f_out = open(out_path, "w")
    for node in values(nodes)
        ss = f"\(node.obj_id),\(node.c_id)\n"
        write(f_out, ss)
    end
    file_name = basename(out_path)
    eval_1(file_name)   # 评估
end


function eval_1(file_name)
    # pushfirst!(PyVector(pyimport("sys")."path"), "")
    # pushfirst!(PyVector(pyimport("sys")."path"), "../..")

    py"""
    import os, sys
    import numpy as np
    import pandas as pd
    sys.path.insert(0, "")
    sys.path.insert(0, "..")
    from utils import eval_cluster


    dir_1 = "/mnt/zy_data/data/pk/pk_13/output_1"
    cluster_path = os.path.join(dir_1, "out_1", $file_name)  # out_1_21.csv
    labels_pred_df = pd.read_csv(cluster_path, names=["obj_id", "person_id"])

    gt_path = os.path.join(dir_1, "merged_all_out_1_1_1_21-small_1.pkl")  # 21  9
    gt_sorted_df = pd.read_pickle(gt_path)

    labels_true, labels_pred, _ = eval_cluster.align_gt(gt_sorted_df, labels_pred_df)
    metric, info = eval_cluster.eval(labels_true, labels_pred, is_show=True)
    metric["img_count"] = len(labels_pred) * 1.0
    metric["cluster_count"] = len(set(labels_pred)) * 1.0
    metric["drop"] = (len(labels_pred_df) - len(labels_true)) / len(labels_pred_df)
    print(f'drop:{metric["drop"]:.4f}')

    """
end


function main()
    # input_path = "/data2/zhangyong/data/pk/pk_13/input/input_languang_5_2.json"   # input_languang_5_2
    input_path = "/mnt/zy_data/data/languang/input_languang_5_2.json"   # input_new.json
    out_path = "/mnt/zy_data/data/pk/pk_13/output_1/out_1/out_tmp_4.csv"
    test_1(input_path, out_path)
    # eval_1(basename(out_path))   # 评估
end


function test_2()
    println("nthreads:", Threads.nthreads())
    xs = randn(1000_000_000)
    # aa = foldl(+, Map(sin), xs)
    bb = foldxt(+, Map(sin), xs)
    print(bb)
end


@time main()



#=
export JULIA_NUM_THREADS=4
julia stream/transducers_7.jl
----------------------------------------
TODO:
0. 加多维信息   OK
0. 加同镜,跨镜,多时空阶段聚类 [keyby]  *****
1. 加代表点[OK], 代表点更新  OK
2. 质量加权动态阈值, 加权到knn里
3. 加knn feat.  OK
4. 接入kafka数据源, 超内存的数据源, 流式的dataloader. OK. 
5. 加窗口
6. 并行flods

加速: 
1. fse/milvus gpu 
2. 并行
3. 多op 级联
----------------------------------------
input_data |> spacetime1_cluster(json) |> spacetime2_cluster(nodes, clsuters) |> global_hac(nodes, clsuters) |> output_data

op1 = Spacetime1_Cluster    # PartitionBy(time)
eachline(input_json) |> Map(prase_json) |> GroupBy((x -> x.device_id), op1) |> op2 |> ouput
走通group by, 走通输出

----------------------------------------
问题: 
1. 没有可add的rank. 慢, 改为faiss[难]
2. foldxt 并行不起作用, 因为不支持eachline

milvus add with ids ,  OK
有问题(c_id 找不到), 结果不能回归. 解决了此bug. ok
还没 和同镜 联调, 联调OK

联调: 跑通. 很慢.  
milvus很慢, gpu使用率非常低. cpu利用率也很低[40%]. 


6.4w
used: 2138.8s=35.6min   foldl
used: 2113.4s=35.2min   foldxt  
used: 313.9s=5.2min  foldl  0.21机器,milvus也在这里
used: 336.5s=5.6min  foldxt 0.21机器,milvus也在这里, nthreads=40, cpu,gpu没变. 



=#

