# milvus api. v2.0     2021.11.22
using HTTP, JSON3
using NPZ
using Dates, BenchmarkTools
using ProgressMeter


function commen_api(component, method, body="", show=false)
    # api_url = "http://192.168.100.221:19121/$component"   # 19530  19121
    api_url = "http://192.168.3.199:19121/$component"
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
    println("creat_collection()")
    try
        delete_collection(collection_name)
        println("delete coll ok")
    catch e
        println("delete error.  ")
    end

    body_dict = 
        Dict("collection_name" => collection_name,
            "fields" => [
                Dict(
                    "name" => "id",
                    "type" => "Int32"
                    # "params" => Dict("dim" => dim)
                ),
                Dict(
                    "name" => "feat",
                    "type" => "VECTOR_FLOAT",
                    "params" => Dict("dim" => dim)
                )
            ],
            "segment_row_limit" => 10000,
            "auto_id" => true
        )

    # body = JSON.json(body_dict)   # body_dict
    body = JSON3.write(body_dict)
    println(body)
    try
        creat_coll = commen_api("createCollection", "POST", body)  # 创建collection
        println(creat_coll)
    catch e
        println("create coll error. ", e)
    end
    body_dict = Dict("collection_name" => collection_name)
    body = JSON3.write(body_dict)
    # get_colls = commen_api("collections", "GET", "")  # 获取到所有collections的信息
    get_coll_info = commen_api("describeCollection", "POST", body)  # 获取指定collections的信息
    println(get_coll_info)

    return collection_name

end


function delete_collection(collection_name)
    # commen_api(component, method, body)
    # devices = commen_api("devices", "GET", "")   # 获取到设备信息
    delete_coll = commen_api("collections/$collection_name", "DELETE")  # 删除collection  

    println(delete_coll)

end


function insert_obj_batch(collection_name, vectors, ids)
    vectors_batch = []
    

end


function insert_obj(collection_name, vectors, ids)
    # println("insert_obj")
    entities = []
    for i in 1:length(vectors)
        entity = Dict(
                # "__id" => ids[i],
                "id" => ids[i],
                "feat" => vectors[i]
                )
        push!(entities, entity)
        
    end

    body_dict = Dict( # "partition_tag" => "part",
        "entities" => entities,
        "ids" => ids
    )

    body = JSON3.write(body_dict)
    
    insert_obj = commen_api("collections/$collection_name/entities", "POST", body)  # insert_obj, 不是实时commit的
    # println(insert_obj)
    flush_coll(collection_name)  # 频繁flush会很慢. 所以要batch的add

end

function flush_coll(collection_name)

    body_dict = 
        Dict("flush" =>  Dict("collection_names" => [collection_name]))
    body = JSON3.write(body_dict)
    commen_api("/system/task", "PUT", body)

end

function search_obj(collection_name, vectors, top_k)
    # println("search_obj")

    body_dict = Dict(
                    "query"=> Dict(
                        "bool"=> Dict(
                            "must" => [
                                Dict("vector"=>
                                    Dict(
                                        "feat"=> Dict(
                                            "params"=> Dict("nprobe"=> 64),
                                            "topk"=> top_k,
                                            "metric_type"=> "IP",
                                            "values"=> vectors
                                        )
                                    )
                                )
                            ]
                        ),
                        "fields" => ["feat"]
                    )
                )

    body = JSON3.write(body_dict)
    # println(body)
    
    rank_result = commen_api("collections/$collection_name/entities", "GET", body)  # 创建collection
    # println(rank_result)
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
    creat_collection(collection_name, 2)  # 384
    delete_collection(collection_name)

    # vectors = [[1.0, 2.0], [2.2, 3.2], [3.1, 4.1]]
    # ids = ["1","2","3"]  # string list
    # # insert_obj(collection_name, vectors, ids)

    # top_k = 10
    # query_vectors = [[2.2, 3.2], [1.1, 2.2]]
    # rank_result = search_obj(collection_name, query_vectors, top_k)
    # # println(rank_result)
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
        feats = npzread("/mnt/zy_data/data/longhu_1/sorted_2/feats.npy")
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


# test_1()
test_2()


#=
有两种方式:1.julia写RESTful API调用, 2.调用python的
1. 创建, 删除 coll
2. add
3. search
4. remove



=#
