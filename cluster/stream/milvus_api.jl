# milvus api. v0.10.x   2020.11.4
using HTTP, JSON3
using Dates, BenchmarkTools
using ProgressMeter
using Strs, NPZ


function commen_api(component, method, body="", show=false)
    # api_url = "tcp://192.168.3.199:19530/$component"   # 19530  19121 
    api_url = "http://192.168.3.199:19121/$component"
    headers = Dict("accept"=>"application/json")  # , "Content-Type" => "application/json"

    response = HTTP.request(method, api_url, headers=headers, body=body)
    status = response.status  #  == 200 ? "OK" : "requests get failed."
    if show println(status) end

    data_text = String(response.body)   # text
    if data_text == "" 
        data_text = "{}"
    end
    data = JSON3.read(data_text)  # string to dict
    if show println("data: ", data) end
    return data
end


function milvus_api()
    # devices = commen_api("devices", "GET", "")   # 获取到设备信息
    get_colls = commen_api("collections", "GET", "")  # 获取到所有collections的信息
    println(get_colls)

end


function creat_collection(collection_name, dim)
    println("creat_collection()")
    try
        delete_collection(collection_name)
        println("delete collection ok")
    catch
        println("collection not exist, delete error")
    end

    body_dict = 
        Dict("collection_name" => collection_name,
              "dimension" => dim,
            #   "index_file_size" => 10000,
              "metric_type" => "IP")

    body = JSON3.write(body_dict)
    try
        creat_coll = commen_api("collections", "POST", body)  # 创建collection
        println("creat_coll ok")
    catch 
        println("create coll error")
    end
    
    # get_colls = commen_api("collections", "GET", "")  # 获取到所有collections的信息
    
    get_coll_info = commen_api("collections/$collection_name", "GET", "")  # 获取指定collections的信息
    println("get_coll_info ok")

    return collection_name

end


function delete_collection(collection_name)
    # commen_api(component, method, body)
    # devices = commen_api("devices", "GET", "")   # 获取到设备信息
    delete_coll = commen_api("collections/$collection_name", "DELETE")  # 删除collection  

    println("delete_coll:", delete_coll)

end


function insert_obj_batch(collection_name, vectors, ids)
    vectors_batch = []
    

end

function insert_obj(collection_name, vectors, ids)
    # println("insert_obj")  # ids只能是数字的字符串
    body_dict = Dict( # "partition_tag" => "test_collection5",
              "vectors" => vectors,
              "ids" => ids   # ids只能是数字的字符串
            )

    body = JSON3.write(body_dict)
    # println(body)
    insert_obj = commen_api("collections/$collection_name/vectors", "POST", body)  # insert_obj, 不是实时commit的
    # println(f"\(ids), \(insert_obj)")
    flush_coll(collection_name)  # 频繁flush会很慢. 所以要batch的add

end

function flush_coll(collection_name)

    body_dict = 
        Dict("flush" =>  Dict("collection_names" => [collection_name]))
    body = JSON3.write(body_dict)
    commen_api("/system/task", "PUT", body)

end

function delete_obj(collection_name, ids)
    body_dict = Dict("delete" => Dict("ids" => ids))
    body = JSON3.write(body_dict)
    delete_obj = commen_api("collections/$collection_name/vectors", "PUT", body)
    # println(f"delete: \(ids), \(delete_obj)")
end

function search_obj(collection_name, vectors, top_k)
    # println("search_obj")
    body_dict = Dict("search" => Dict(
                    "topk" => top_k,
                    # "partition_tags" => [string],
                    # "file_ids" => [string],
                    "vectors" => vectors, 
                    "params" => Dict("nprobe" => 16))
                )

    body = JSON3.write(body_dict)
    
    rank_result = commen_api("collections/$collection_name/vectors", "PUT", body)  # 创建collection
    # println(rank_result)
    return rank_result
end

function get_feat(collection_name, ids)
    # println("search_obj")
    ids_list = join(ids, ",")
    rank_result = commen_api("collections/$collection_name/vectors?ids=$ids_list", "GET", "")
    # println(rank_result)
    feats = []
    for vector in rank_result["vectors"]
        id = vector["id"]
        feat = Array(vector["vector"])
        if length(feat) == 0
            continue
        end
        push!(feats, feat)
        # println(f"-----:\(ids), \(id)")
    end
    feats = vcat((hcat(i...) for i in feats)...)

    return feats
end

function prcoess_results_3(results, topk)
    # println(results)
    size = results["num"]
    result = results["result"]

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

function prcoess_results_2(results, topk)
    # println(results)
    size = results["num"]
    result = results["result"]

    dists = zeros(Float32, (size, topk))
    idxs = Array{String,2}(undef, size, Int64(topk))

    for i in 1:size
        for j in 1:topk
            try
                data = result[i][j]
                if data["id"] == "-1"
                    dists[i, j] = -1
                    idxs[i, j] = data["id"]
                else
                    dists[i, j] = parse(Float32, data["distance"])
                    idxs[i, j] = data["id"]
                end
            catch
                dists[i, j] = -1
                idxs[i, j] = "-1"
            end
        end
    end

    return dists, idxs

end


function test_1()
    collection_name = "test_coll_1"
    creat_collection(collection_name, 2)

    vectors = [[1.0, 2.0], [2.2, 3.2], [3.1, 4.1]]
    # ids = ["aesa6ut","bdg5r","crdf3w"]  # string list
    ids = ["1", "2", "3"]
    insert_obj(collection_name, vectors, ids)

    top_k = 10
    query_vectors = [[2.2, 3.2], [1.1, 2.2]]
    rank_result = search_obj(collection_name, query_vectors, top_k)
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
        feats = npzread("/data/zhangyong/data/longhu_1/sorted_2/feats.npy")
    end

    feats = convert(Matrix, feats[1:10000, 1:end])
    size_1 = size(feats)[1]
    t1 = Dates.now()
    println("used: ", (t1 - t0).value/1000, "s, ", size_1)
    # println(size(feats))
    collection_name = "test2"
    creat_collection(collection_name, 384)  # 384
    # get_coll_info = commen_api("collections/$collection_name", "GET", "")  # 获取指定collections的信息
    # println(get_coll_info)
    top_k = 1000
    @showprogress for i in 1:size_1
        vectors = [feats[i,1:end]]
        qurey_vectors = [feats[i,1:end]]
        # println(size(qurey_vectors))
        ids = [string(i)]

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
# test_2()


#=
v0.10
有两种方式:1.julia写RESTful API调用, 2.调用python的
1. 创建, 删除 coll
2. add
3. search
4. remove

https://github.com/milvus-io/milvus/tree/0.10.5/core/src/server/web_impl


test_2()   100000: add+search  13min

为什么用不了GPU?

=#
