# using Strs
using PythonCall
ENV["JULIA_PYTHONCALL_EXE"] = "CONDA"
sys = pyimport("sys")
np = pyimport("numpy")
# println(sys.path)


function create_index(feat_dim, gpus="")
    faiss = pyimport("faiss")
    # println("ngpus:", faiss.get_num_gpus())

    ENV["CUDA_VISIBLE_DEVICES"] = gpus
    if gpus == ""
        ngpus = 0
    else
        ngpus = length(split(ENV["CUDA_VISIBLE_DEVICES"], ','))
    end

    if ngpus == 0
        cpu_index = faiss.IndexFlatL2(feat_dim)
        gpu_index = cpu_index
    elseif ngpus == 1
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        flat_config.useFloat16 = True
        res = faiss.StandardGpuResources()
        # res.setTempMemory(6516192768)
        gpu_index = faiss.GpuIndexFlatL2(res, feat_dim, flat_config)  # use one gpu.  初始化很慢
    else
        # print('use all gpu')
        cpu_index = faiss.IndexFlatL2(feat_dim)
        gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)  # use all gpus
    end

    index = gpu_index
    return index
end


function rank(index, query, gallery; topk=1000, metric="cos")
    index.add(gallery)  # <8M
    topk = min(topk, pyconvert(Int, gallery.shape[0]))
    dists = np.zeros((query.shape[0], topk), "f")  # 很大.  n*k*4  1m->4G
    idxs = np.zeros((query.shape[0], topk), "int32")  # 很大.  n*k*4
    bs = pyconvert(Int, np.ceil(1e7 * 1.0 / topk))  # 1e6 /  topk=100w/1000  1000, 1w
    qbatch = pyconvert(Int, np.ceil(query.shape[0] * 1.0 / bs))
    println(qbatch)
    for i in 0:qbatch-1
        slice_1 = pyslice(i * bs,(i + 1) * bs)
        dist, idx = index.search(query[slice_1], topk)  # L2距离
        dists[slice_1] = dist
        idxs[slice_1] = idx
    end
    # println(idxs.shape, dists.shape)
    if metric == "cos"
        dists = 1 - dists / 2  # 转换为cos相似度
    else
        dists = dists  # l2 相似度
    end

    return dists, idxs
end


function test()
    dir_1 = "/mnt/zy_data"
    feats = np.load(joinpath(dir_1, "feats.npy"))
    println(feats.shape)

    # index = create_index(384, "")   # L2 index
    # index.add(feats)
    # dist, idx = index.search(feats[0:10], 5)
    # println(1 - dist / 2)

    query = feats[0:10]
    println(query.shape)
    gallery = feats
    feat_dim = query.shape[1]
    index = create_index(feat_dim, "")
    dists, idxs = rank(index, query, gallery, topk=5)
    println(dists)

end


# test()


#=
问题:
[1:8] 的含义不同. 到底当前用的哪个, 是不明确的
不支持np.array()的超出总数的idx. 报错 
解决办法: query[pyslice(5,8)]   # idx就是python的 [5,8], 5<=x<8.  

=#


