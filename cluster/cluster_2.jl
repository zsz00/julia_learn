# https://github.com/JuliaStats/Clustering.jl   julia聚类  (包含K-Means, 层次聚类)
# using Clustering
# using NearestNeighbors
# using Makie  # plot
using Dates
using PyCall
# using Distances 
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
    idx_1 = findall(dists.>th)  
    println(size(idx_1), typeof(idx_1))  # (6501809,)Array{CartesianIndex{2},1} 
    ys = idx[idx_1] .+ 1
    idx_1 = Tuple.(idx_1)
    println(size(ys), " ", size(idx_1))
    # return 0
    size_1 = size(dists)[1]
    rank = zeros(Int, size_1)
    same = Array(range(1,size_1, step=1))

    Threads.@threads for i in range(1,size(idx_1)[1],step=1)  # pair数, 量大
        same, rank = union!(idx_1[i][1], ys[i], same, rank)
    end
    labels = [find(i, same) for i in range(1,size_1,step=1)]
    labels = labels .-1

    dic = Dict()
    for (i, l) in enumerate(labels)
        add(dic, l, i)
    end
    println("cluster:", th, " img:", size(labels), " id:", length(dic))

end


function cluster_1()
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

@time cluster_1()

#=
(3742293,) (3742293,)
100%|##########| 3742293/3742293 [00:13<00:00, 287073.56it/s]
cluster: img: 185091 id: 59732
th: 0.6 id_sum: 59732 cluster over...
[67526     1     4     4     4     7     7     7     9     9     9    12
    12    14    14    14 65600    18    18    21    21    21    23    23
    23    25    30    30    28    39    30    30    30 55339    30    30
    30    64 23622    39]
cluser time:14 s
cluser_c: 1.6 s/th
cluser_jl: 6s    10s
=#

#=
def cluster():
    wh = np.where((dists > th))  # cos相似度  &(dists < 0.9)
    # 下边都是索引操作
    xs = wh[0]
    ys = idx[wh]
    size = idx.shape[0]  # img_sum, label_sum
    rank = [0] * size
    same = range(size)  # 注:在 py3 中这不是个list,是个迭代器. 要list(range(szie))

    # print "mat:", dists.shape, idx.shape
    # print "xs, ys:", xs.shape, ys.shape, xs.shape[0], "pairs"
    for i in tqdm(range(xs.shape[0])):  # pair数, 量大
        union(xs[i], ys[i], same, rank)  # 结合,合并
    labels = np.array([find(i, same) for i in range(size)])


def find(x, same):
    if x != same[x]:
        same[x] = find(same[x], same)
    return same[x]


def union(x, y, same, rank):
    x = find(x, same)
    y = find(y, same)
    if x == y:
        return
    if rank[x] > rank[y]:
        same[y] = x
    else:
        same[x] = y
        if rank[x] == rank[y]:
            rank[y] += 1

=#
