#=
eval cluster 
# https://blog.csdn.net/SAM2un/article/details/85013340 
=# 
using PyCall
using ProgressMeter
using Printf 


function f_score(cluster, labels)
    TP = 0  # Threads.Atomic{Int64}(0)
    FP = 0  # Threads.Atomic{Int64}(0)
    TN = 0  # Threads.Atomic{Int64}(0)
    FN = 0  # Threads.Atomic{Int64}(0)

    labels_size = size(labels)[1]  # (70184, 1)
    for i in range(1, labels_size, step=1)   # Threads.@threads 
        for j in range(i+1, labels_size, step=1)
            same_label = (labels[i] == labels[j])
            same_cluster = (cluster[i] == cluster[j])
            if same_cluster
                if same_label
                    TP += 1
                else
                    FP += 1
                end
            else
                if same_label
                    FN += 1
                else
                    TN += 1
                end
            end
        end
    end

    # count_TP = count_TP.value
    # count_FP = count_FP.value
    # np.savetxt("fp_out/out_dengqili/j_$(threshold)_2.txt", fp, fmt="%d")
    count_pairs = TP + FP + FN + TN
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    acc = (TP+TN) / count_pairs
    fscore = 2 * precision * recall / (precision + recall)
    # println("fscore, precision, recall, count_pairs:", join([fscore, precision, recall, count_pairs],", "))
    @printf("fscore:%.4f acc/RI:%.6f precision:%.4f recall:%.4f count_pairs:%d\n", fscore, acc, precision, recall, count_pairs)
    return fscore, acc, precision, recall, count_pairs
end


dir_1 = "/data/yongzhang/cluster/test_2"
label_file = joinpath(dir_1, "labels.txt")
labels = readlines(label_file)
labels = [parse(Int,a) for a in labels]
cluster_file = joinpath(dir_1, "cluster_2.txt")
cluster = readlines(cluster_file)
cluster = [parse(Int,a) for a in cluster]


@time f_score(cluster, labels)



#=
cluster_3.txt:
fscore:0.9928 acc/RI:0.999956 precision:0.9907 recall:0.9950 count_pairs:2462861836   5.630300 seconds 


=#
