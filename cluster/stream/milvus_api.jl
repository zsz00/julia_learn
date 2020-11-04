# milvus api.   2020.11.4
using HTTP, JSON3
using NPZ
using Dates, BenchmarkTools
using ProgressMeter


function commen_api(component, method, body="", show=false)
    api_url = "http://192.168.100.221:19121/$component"   # 19530  19121
    headers = Dict("accept"=>"application/json")  # , "Content-Type" => "application/json"

    response = HTTP.request(method, api_url, headers=headers, body=body)   # 必须是body, 不能是data,json
    # response = HTTP.request(method, api_url, headers=headers)
    # a ? b : c
    status = response.status  #  == 200 ? "OK" : "requests get failed."   # status  status_code
    # println(status)

    data_text = String(response.body)   # text
    if data_text == "" 
        data_text = "{}"
    end
    # println("data_text: ", data_text)

    data = JSON3.read(data_text)  # string to dict
    # println(data)
    return data
end


function milvus_api()
    # commen_api(component, method, body)
    # devices = commen_api("devices", "GET", "")   # 获取到设备信息
    get_colls = commen_api("collections", "GET", "")  # 获取到所有collections的信息
    println(get_colls)

end


function creat_collection(collection_name, dim)
    println("creat_collection")

    body_dict = 
        Dict("collection_name" => collection_name,
              "dimension" => dim,
            #   "index_file_size" => 10000,
              "metric_type" => "IP")

    # body = JSON.json(body_dict)   # body_dict
    body = JSON3.write(body_dict)
    
    creat_coll = commen_api("collections", "POST", body)  # 创建collection
    println(creat_coll)

    get_colls = commen_api("collections", "GET", "")  # 获取到所有collections的信息
    get_coll_info = commen_api("collections/$collection_name", "GET", "")  # 获取指定collections的信息
    println(get_coll_info)

    return collection_name

end


function delete_collection(collection_name)
    # commen_api(component, method, body)
    # devices = commen_api("devices", "GET", "")   # 获取到设备信息
    delete_coll = commen_api("collections/$collection_name", "DELETE")  # 删除collection  

    println(delete_coll)

end


function add_obj(collection_name, vectors, ids)
    # println("add_obj")
    body_dict = 
        Dict( # "partition_tag" => "test_collection5",
              "vectors" => vectors,
              "ids" => ids
            )

    body = JSON3.write(body_dict)
    # println(body)
    
    creat_coll = commen_api("collections/$collection_name/vectors", "POST", body)  # 创建collection
    # println(creat_coll)

end


function search_obj(collection_name, vectors)
    # println("search_obj")
    body_dict = Dict("search" => Dict(
                    "topk" => 100,
                    # "partition_tags" => [string],
                    # "file_ids" => [string],
                    "vectors" => vectors,
                    "params" => Dict("nprobe" => 16))
                )

    body = JSON3.write(body_dict)
    # println(body)
    
    rank_result = commen_api("collections/$collection_name/vectors", "PUT", body)  # 创建collection
    # println(rank_result)
    return rank_result
end


function prcoess_results_3(results, topk, size)
    # size = len(results)
    dists = np.zeros((size, topk), "float")
    idxs = np.zeros((size, topk), "int32")

    for i in range(size)
        for j in range(topk)
            try
                data = results[i][j]
                if data.id == -1
                    dists[i, j] = -1
                    idxs[i, j] = -1
                else
                    dists[i, j] = data.distance
                    idxs[i, j] = data.id
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
    delete_collection(collection_name)
    creat_collection(collection_name, 384)
    # delete_collection(collection_name)

    # vectors = [[1.0, 2.0], [2.2, 3.2], [3.1, 4.1]]
    # ids = ["1","2","3"]  # string list
    # add_obj(collection_name, vectors, ids)

    # query_vectors = [[2.2, 3.2]]
    # search_obj(collection_name, query_vectors)

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

    feats = convert(Matrix, feats[1:100000, 1:end])
    size_1 = size(feats)[1]
    t1 = Dates.now()
    println("used: ", (t1 - t0).value/1000, "s, ", size_1)
    # println(size(feats))
    collection_name = "test_coll_1"
    get_coll_info = commen_api("collections/$collection_name", "GET", "")  # 获取指定collections的信息
    println(get_coll_info)

    @showprogress for i in 1:size_1
        vectors = [feats[i,1:end]]
        qurey_vectors = [[feat] for feat in feats[i:i+1,:]]
        println(size(qurey_vectors))
        ids = [string(i), string(i+1)]

        add_obj(collection_name, vectors, ids)   # add
        rank_result = search_obj(collection_name, qurey_vectors)  # rank
        if i == 5
            println(rank_result)
        end
        # prcoess_results_3(rank_result, topk, size)  # 解析rank结果

    end

end


# test_1()
test_2()


#=
有两种方式:1.julia写RESTful API调用, 2.调用python的
1. 创建, 删除 coll
2. add
3. search
4. remove

https://github.com/milvus-io/milvus/tree/0.11.0/core/src/server/web_impl


test_2()   100000: add+search  13min

=#
