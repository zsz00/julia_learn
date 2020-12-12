# julia聚类, 层次聚类
using Dates
using ProgressMeter
using PyCall
np = pyimport("numpy")


function find(x, same)
    if x != same[x]
        same[x] = find(same[x], same)
    end
    return same[x]
end

function union!(x, y, same, rank)
    x = find(x, same)
    y = find(y, same)
    if x == y
        return same, rank
    end
    if rank[x] > rank[y]
        same[y] = x
    else
        same[x] = y
        if rank[x] == rank[y]
            rank[y] += 1
        end
    end
    return same, rank
end

function add(dic, k, v)  
    aa = k in keys(dic)    
    if aa == false
        dic[k] = []
    end
    push!(dic[k], v)
end


function cluster_jl(dists, idx, th)
    println(size(dists))  # (185091, 1000)
    dists = dists[:,1:end]
    # th = 0.6
    idx_1 = findall(dists.>th)  # 很慢
    println(size(idx_1), typeof(idx_1))  # Array{CartesianIndex{2},1} 
    ys = idx[idx_1] .+ 1
    idx_1 = Tuple.(idx_1)
    println(size(ys), " ", size(idx_1))
    # return 0
    size_1 = size(dists)[1]
    rank = zeros(Int, size_1)
    same = Array(range(1,size_1, step=1))
    Threads.@threads for i in range(1,size(idx_1)[1],step=1)  # pair数, 量大
        same, rank = union!(idx_1[i][1], ys[i], same, rank)   # 基于数组的
    end
    labels = [find(i, same) for i in range(1,size_1,step=1)]
    labels = labels .-1

    dic = Dict()
    for (i, l) in enumerate(labels)
        add(dic, l, i)
    end
    println("cluster:", th, " img:", size(labels), " id:", length(dic))
end

function cluster_jl_2(mat_csr, th)
    # 输入是 mat_csr, 稀疏矩阵.
    gallery_shape = mat_csr.shape[1] 
    println("gallery_shape;", gallery_shape)
    size_1 = gallery_shape
    rank = zeros(Int, size_1)
    same = Array(range(1,size_1, step=1))
    # Threads.@threads
    @showprogress for i in range(1, stop=gallery_shape)
        sim = mat_csr[i].data
        idx = mat_csr[i].indices
        # println("sim: ", sim, " idx: ", idx)
        if size(sim)[1] == 0
            continue
        end

        wh = np.where(sim .> th)  # 用的python numpy 会慢,类型转换.还是两语言问题. 比直接用python还慢. 
        xs = wh[1]
        ys = idx[xs.+1] .+1  # xs
        # println(xs, " ", ys)
        for j in range(1, stop=size(xs)[1])  # pair数, 量大.  要求query=gallery
            union!(i, ys[j], same, rank)   # 结合,合并
        end
    end
    labels = [find(i, same) for i in range(1,size_1,step=1)]
    labels = labels .-1

    dic = Dict()
    for (i, l) in enumerate(labels)
        add(dic, l, i)
    end
    println("cluster: th:", th, " img:", size(labels), " id:", length(dic))
    np.save("sty_labels_$(th).npy", labels)
end


function cluster_one()
    ENV["JULIA_NUM_THREADS"]=40
    println("load mat...")
    dists = np.load("/data/yongzhang/cluster/data_3/clean_2/out_disk_2_1/mat.npy")  # 19G*2 mem
    idx = np.load("/data/yongzhang/cluster/data_3/clean_2/out_disk_2_1/idx.npy")
    println("load mat over...")

    thresholds = Array(range(0.5,0.6, step=0.01))
    time1 = Dates.now()
    for (i, th) in enumerate(thresholds)
        cluster_jl(dists, idx, th)
    end
    println("used: ", (Dates.now()-time1).value/1000, " s")
end


function cluster_all()
    # x = features = np.load(raw"D:\download\features\512.fea.npy")
    # r = euclidean(x, x)
    # from scipy.sparse import csr_matrix,save_npz,load_npz
    # D = load_npz("/data/yongzhang/cluster/data_3/clean/out_disk_2_jxx_1/disk_2_jxx_local_mat.npz")
    # D = np.load("/data/yongzhang/cluster/data_3/clean/out_disk_2_jxx_4/disk_2_jxx_mat.npy")
    # dists = np.load(raw"D:\test\mat.npy")  # 19G*2 mem
    # idx = np.load(raw"D:\test\idx.npy")
    dists = np.load(raw"/data/yongzhang/cluster/data_3/clean/out_disk_2_jxx_7/mat.npy")  # 19G*2 mem
    idx = np.load(raw"/data/yongzhang/cluster/data_3/clean/out_disk_2_jxx_7/idx.npy")
    thresholds = Array(range(0.5,0.6, step=0.01))
    time1 = Dates.now()
    for (i, th) in enumerate(thresholds)
        cluster_jl(dists, idx, th)
    end
    println("used: ", (Dates.now()-time1).value/1000, " s")
end


function cluster_all_2()
    # 跑稀疏数据的
    # ENV["JULIA_NUM_THREADS"]=10
    scipy = pyimport("scipy.sparse")
    load_npz = scipy.load_npz
    println("load mat...")
    # mat_csr = load_npz("/data5/yongzhang/cluster/data/cluster_data/valse/out1/mat.npy.npz")  # 19G*2 mem
    mat_csr = load_npz("/data5/yongzhang/cluster/data/cluster_data/sty/mat.npy.npz")  # 19G*2 mem
    println("load mat over...")
    println("mat_csr: ", mat_csr.shape[1])   # getnnz(mat_csr), 

    thresholds = Array(range(0.5,0.5, step=0.1))
    time1 = Dates.now()
    for (i, th) in enumerate(thresholds)
        cluster_jl_2(mat_csr, th)
    end
    println("used: ", (Dates.now()-time1).value/1000, " s")
end


@time cluster_all_2()



#=
2020.2
(3742293,) (3742293,)
100%|##########| 3742293/3742293 [00:13<00:00, 287073.56it/s]
cluster: img: 185091 id: 59732
th: 0.6 id_sum: 59732 cluster over...

cluser_py:14 s
cluser_c: 1.6
cluser_jl: 6s    10s

cluser_jl: 6.5min 很慢  3753887 imgs  c: 22s
py: 13min

不能用多线程, 会Segmentation fault (core dumped)
48s
55s
=#

#=
export JULIA_NUM_THREADS=40
ENV["JULIA_NUM_THREADS"]=40
=#
