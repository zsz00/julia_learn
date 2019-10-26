import Dates
using PyCall
np = pyimport("numpy")


function  compute_iou(pred, label)
    s1 = Set(pred)
    s2 = Set(label)
    # println(union(s1, s2), " ", intersect(s1, s2))
    return 1. * length(intersect(s1, s2)) / length(union(s1, s2))
end

function  test_iou()
    a = [1   1  1]
    b = [1 2  1  1]
    aa = compute_iou(a, b)
    println(aa)
end


function nms(clusters, th=1.0)
    # nms
    println("nms...")
    time1 = Dates.now()
    suppressed = Set()
    if th < 1
        tot_size = length(clusters)
        println("len(clusters):", tot_size)
        for start_idx in range(1,tot_size)
            if issubset(start_idx, suppressed)
                continue
            end
            cluster = clusters[start_idx]
            for j in range(start_idx+1, tot_size)
                if issubset(j, suppressed)
                    continue
                end
                if compute_iou(cluster, clusters[j]) > th    # 慢
                    push!(suppressed, j)
                end
            end
            if start_idx % 100 == 1
                println(start_idx, " ", tot_size, " ", (Dates.now()-time1).value/(1000*start_idx))   # 0.42s/item
            end
        end
    else
        println("th=$(th), th>= 1, skip the nms")
    end
    println("used: ", (Dates.now()-time1).value/1000, " s")

end


clusters = np.load(raw"C:\zsz\ML\code\julia\julia_learn\cluster\clusters.npy")   # , allow_pickle=1
println("load mat over...")
nms(clusters, 0.9)

