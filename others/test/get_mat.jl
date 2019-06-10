using PyCall
using ProgressMeter
# using HDF5
# using Clustering

np = pyimport("numpy")

function get_label_mat_2(mat, idx, labels)
    threshold = 0.6
    count_T = 0
    count_F = 0
    count_P = 0
    count_TP = 0
    count_FP = 0
    s_1 = ""
    fp = []
    labels_size = size(labels)   # (1862120, 1)
    println(size(mat))
    @showprogress for i in range(1,labels_size[1],step=1)
        if labels[i] in (-1,-2)
            continue
        end
        # println(i, " id:", labels[i])
        idx_1 = findall(x->(x==labels[i]) && (x !=-2), labels)  # 正样本
        idx_1_1 = Tuple.(idx_1)   # idx
        # println("idx_1:", size(idx_1), " ", idx_1_1, "\n", idx_1_1[1])
        idx_1_2=findall(x->x>(i,1), idx_1_1)
        idx_1_3 = idx_1_1[idx_1_2]
        # println("idx_1_3:", size(idx_1_3), " ", idx_1_3)

        # print("gt: i:%s, id:%s, T:%s, F:%s, %s" % (
        #     i, labels[i][0], idx_1.shape[0], idx_2.shape[0], idx_1.shape[0] + idx_2.shape[0]))
        count_T += length(idx_1_3)
        # count_F += idx_2.shape[0]
        
        sim = mat[i,1:end]
        idx_p_1 = findall(x-> x>threshold, sim)
        # println("idx_p_1:", size(idx_p_1), " ", idx_p_1)
        idx_p_2 = idx[i,1:end][idx_p_1]
        idx_p_2 = idx_p_2 .+ 1
        # println("idx_p_2:", size(idx_p_2), idx_p_2)
        idx_p_3=findall(x-> x>i, idx_p_2)
        idx_p_4 = idx_p_2[idx_p_3]   # p
        # println("P:", size(idx_p_4), " ", idx_p_4)

        count_2 = 0
        count_3 = 0
        for idx_5 in idx_p_4
            if (idx_5,1) in idx_1_3
                count_2 += 1
                # println("TP:", idx_5)
            else
                count_3 += 1
                # println("FP:", idx_5)
                push!(fp,[i-1, idx_5-1])
            end
        end
        # println(i, " id:", labels[i]," TP:", count_2, " FP:", count_3)
        count_TP += count_2
        count_FP += count_3

        # if i >10
        #     break
        # end

    end
    println("count_T:", count_T, " count_TP:",count_TP," count_FP:", count_FP)
    # h5write("fp_out/test_1.h5", "fp", fp)
    # HDF5.h5open("fp_out/test_1.h5", "w") do file
    #     write(file, "fp", fp)
    # end
    np.savetxt("fp_out/test_1.txt", fp, fmt="%d")

end


function main()
    # dir_2 = "/data/yongzhang/22/test_1"
    dir_2 = "/data/yongzhang/cluster/test_1"
    mat_file = joinpath(dir_2, "out_dengqili/out_5/mat.npy")  # top_k=1000
    idx_file = joinpath(dir_2, "out_dengqili/out_5/idx.npy")
    label_file = joinpath(dir_2, "deepglint.npy")

    mat_file_2 = joinpath(dir_2, "out_dengqili/out_5/mat_2.npy")  # top_k=1000
    idx_file_2 = joinpath(dir_2, "out_dengqili/out_5/idx_2.npy")
    
    mat = np.load(mat_file)   # 会比在py例加载慢很多, 内存使用多
    idx = np.load(idx_file)
    labels = np.load(label_file)
    # print("mat[1:1000].shape:", size(mat[1:1000,1:end]))
    # mat_2 = np.save(mat_file_2, mat[1:1000,1:end])
    # idx_2 = np.save(idx_file_2, idx[1:1000,1:end])
    # labels_2 = np.save(labels_file_2, labels[:1000])

    get_label_mat_2(mat, idx, labels)

end

@time main()

