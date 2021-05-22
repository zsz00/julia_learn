# ann, similarity search. 2021.5.22
using NPZ, JLD2, FileIO
using Strs, JSON3
using NearestNeighbors, Distances
using SimilaritySearch
# include("faiss_api.jl")


function rank_1(feats, top_k, n)
    # base faiss_python. 未测试完成. 基于faiss_api.jl
    query = np.array(feats)
    gallery = query
    feat_dim = query.shape[1]
    index = create_index(feat_dim, "")
    dists, idxs = rank(index, query, gallery, topk=top_k)
    idxs = idx .+ n
    return dists, idxs
end

function rank_2(feats, top_k, n)
    # base NearestNeighbors.jl, 内存式,小批量适用. 不支持ids
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
    # knn, top_k. 基于NN的. 内存式,小批量适用. 支持ids
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

    index = ExhaustiveSearch(NormalizedCosineDistance(), gallery)   # gallery是Vectors,不支持增量add
    out = [search(index, q, KnnResult(top_k)) for q in query]

    # 后处理
    dists, idxs = prcoess_ss(out, top_k)  # 解析
    idxs = idxs .+ n

    return dists, idxs
end

function prcoess_ss(results, topk)
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
    dists = 1.0 .- dists
    return dists, idxs
end


function test_ss()
    # 基于 SimilaritySearch.jl, 但是结果不准确, 还要查出 n*m,再取topk.
    feats = npzread("/mnt/zy_data/data/longhu_1/feats.npy")
    # feats = convert(Matrix, feats[1:end, 1:end])
    feats = matix2Vectors(feats)

    size_1 = size(feats)
    println(size(feats[1]), typeof(feats[1]))

    t0 = Dates.now()
    query = feats[1:10]
    gallery = query
    println(size(query[1]), typeof(query[1]))
    topk = 3
    index = ExhaustiveSearch(NormalizedCosineDistance(), gallery)  # gallery是Vectors,不支持增量add
    out = [search(index, q, KnnResult(topk)) for q in query]
    println(length(out), out)
    dists, idxs = prcoess_ss(out, topk)  # 解析
    
    println(dists)
    # println(idxs)
    
    t1 = Dates.now()
    println("used: ", (t1 - t0).value/1000, "s, ", size_1)

    return out
end


# test_ss()
