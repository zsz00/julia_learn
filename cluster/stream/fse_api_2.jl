# fse api. 2.调用python的   2021.1.22
using HTTP, JSON3, NPZ
using Dates, BenchmarkTools
using ProgressMeter
using Strs
using PyCall


function creat_collection(collection_name, dim)
    pushfirst!(PyVector(pyimport("sys")."path"), "")

    py"""
    import os
    import numpy as np
    import pandas as pd
    from utils import fse_api_2

    index = fse_api_2.IndexFseKnn($dim, bgpu=True, repo=$collection_name)
    index.delete_repo()
    index.create_repo(is_knn='false')

    """
    index = py"index"
    println(index)
    return index
    
end


function delete_collection(collection_name)
    # commen_api(component, method, body)
    # devices = commen_api("devices", "GET", "")   # 获取到设备信息
    delete_coll = commen_api("collections/$collection_name", "DELETE")  # 删除collection  

    println(delete_coll)

end


function insert_obj(index, vectors, ids)
    println("insert_obj")

    index.add(vectors, ids)  # 全加入底库

end

function flush_coll(collection_name)

    body_dict = 
        Dict("flush" =>  Dict("collection_names" => [collection_name]))
    body = JSON3.write(body_dict)
    commen_api("/system/task", "PUT", body)

end


function search_obj(index, vectors, top_k)
    # println("search_obj")
    
    rank_result = index.search(vectors, top_k,)
    
    return rank_result
end


function prcoess_results_3(results, topk)

    size = length(results["data"]["result"])   # results["nq"]
    result = results["data"]["result"]

    dists = zeros(Float32, (size, topk))
    idxs = zeros(Int32, (size, topk))

    for i in 1:size
        for j in 1:topk
            try
                data = result[i][j]
                if data["id"] == "-1"
                    dists[i, j] = -1
                    idxs[i, j] = -1
                else
                    dists[i, j] = parse(Float32, data["distance"])
                    idxs[i, j] = parse(Int32, data["id"])
                end
            catch
                dists[i, j] = -1
                idxs[i, j] = -1
            end
        end
    end

    return dists, idxs

end


function test_1()
    collection_name = "test_coll_1"
    # delete_collection(collection_name)
    index = creat_collection(collection_name, 2)  # 384
    # delete_collection(collection_name)

    vectors = [[1.0, 2.0], [2.2, 3.2], [3.1, 4.1]]
    ids = ["1","2","3"]  # string list
    # 需要先把类型转为 py的类型

    vectors = PyObject(vectors)
    ids = PyObject(ids)
    insert_obj(index, vectors, ids)

    top_k = 3
    query_vectors = [[2.2, 3.2], [1.1, 2.2]]
    query_vectors = PyObject(query_vectors)
    rank_result = search_obj(index, query_vectors, top_k)
    println(rank_result)
    # dists, idxs = prcoess_results_3(rank_result, 5)
    # println(dists)
    # println(idxs)
end


function test_2()
    println("test_2()")
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
    collection_name = "test_coll_2"
    creat_collection(collection_name, 384)  # 384
    # get_coll_info = commen_api("collections/$collection_name", "GET", "")  # 获取指定collections的信息
    # println(get_coll_info)
    top_k = 10
    @showprogress for i in 1:size_1
        vectors = [feats[i,1:end]]
        qurey_vectors = [feats[i,1:end]]
        # println(size(qurey_vectors))
        # ids = [string(i)]
        ids = [i]

        insert_obj(collection_name, vectors, ids)   # add
        rank_result = search_obj(collection_name, qurey_vectors, top_k)  # rank
        
        dists, idxs = prcoess_results_3(rank_result, 5)  # 解析rank结果
        if i == 10
            println(rank_result)
            println(dists)
            println(idxs)
        end

    end

end


test_1()
# test_2()


#=
有两种方式:1.julia写RESTful API调用, 2.调用python的
1. 创建, 删除 coll
2. add
3. search
4. remove

https://github.com/milvus-io/milvus/tree/0.11.0/core/src/server/web_impl

v0.11  变化很大
很多API变化了. creat, insert, search都变了

v0.10 时, ids 是 [string]
v0.11 时, ids 是 [int]
test_2()   100000: add+search  13min

=#


