ENV["JULIA_PYTHONCALL_EXE"] = "/home/zhangyong/miniconda3/bin/python"

using PythonCall

sys = pyimport("sys")
np = pyimport("numpy")
sys.path.insert(0, pystr("/home/zhangyong/codes/julia_learn/cluster/stream"))
faiss_api = pyimport("faiss_api")
# println(sys.path)


function test()
    dir_1 = "/mnt/zy_data/data/longhu_1/sorted_2/"
    feats = np.load(joinpath(dir_1, "feats.npy"))
    println(typeof(feats), feats.shape)
    # feats 需要从julia对象转换为py对象
    index = faiss_api.get_index(384, "")   # L2 index
    println(typeof(index))
    index.add(feats)
    dist, idx = index.search(feats[0:10], 5)  # size:(11,384),因为[0:10]是julia对象
    println(1 - dist / 2)

    query = feats[0:10]    # py对象
    query = pyconvert(Array{Float32, 2}, query)   # juia对象
    # query = ones(Float32, (10, 384))  # Array{Float32}()
    query = np.array(pyrowlist(query), dtype=np.float32)   # py对象
    println(query.shape)
    gallery = feats
    feat_dim = query.shape[1]
    index = faiss_api.get_index(feat_dim, "")
    idxs, dists = faiss_api.rank5(index, query, gallery, topk=5)  # py对象
    println(dists)
    dists = pyconvert(Array{Float32, 2}, dists)
    idxs = pyconvert(Array{Int64, 2}, idxs)

end


# test()


#=
问题:
不支持np.array()的超出总数的idx. 报错 
解决办法: query[pyslice(5,8)]   # idx就是python的 [5,8], 5<=x<8.  

=#


