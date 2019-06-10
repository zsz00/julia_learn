using PyCall
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
    println(typeof(labels))
    for i in range(1,labels_size[1],step=1)
        if labels[i] in (-1,-2)
            continue
        end
        println(i, " id:", labels[i])
        idx_1 = findall(x->(x==labels[i]) && (x !=-2), labels)  # 正样本
        idx_1_1 = Tuple.(idx_1)   # idx
        # idx_1_1[:,1]
        # labels[idx_1]  # data
        println("idx_1:", size(idx_1), " ", idx_1_1, "\n", idx_1_1[1])
        idx_1_2=findall(x->x>(i,1), idx_1_1)
        idx_1_3 = idx_1_1[idx_1_2]
        println("idx_1_3:", size(idx_1_3), " ", idx_1_3)

        # print("gt: i:%s, id:%s, T:%s, F:%s, %s" % (
        #     i, labels[i][0], idx_1.shape[0], idx_2.shape[0], idx_1.shape[0] + idx_2.shape[0]))
        # count_T += size(idx_1_3)
        # count_F += idx_2.shape[0]
        
        sim = mat[i]
        idx_p_1 = findall(x->x>threshold, sim)
        println("idx_p_1:", size(idx_p_1), "\n", idx_p_1)
        println("idx[i]:", size(idx[i]))
        idx_p_2 = idx[i][idx_p_1]
        idx_p_2=findall(x->x>(i,1), idx_p_2)
        idx_p_3 = idx_p_2[idx_p_2]   # p
        println("idx_p_3:", size(idx_p_3), " ", idx_p_3)

        if i >6
            break
        end


    end
    println(count_T, count_F)


end


function main()
    # dir_2 = "/data/yongzhang/22/test_1"
    dir_2 = "/data/yongzhang/cluster/test_1"
    mat_file = joinpath(dir_2, "out_dengqili/out_5/mat.npy")  # top_k=1000
    idx_file = joinpath(dir_2, "out_dengqili/out_5/idx.npy")
    label_file = joinpath(dir_2, "deepglint.npy")

    mat_file_2 = joinpath(dir_2, "out_dengqili/out_5/mat_3.npy")  # top_k=1000
    idx_file_2 = joinpath(dir_2, "out_dengqili/out_5/idx_3.npy")
    
    mat = np.load(mat_file)   # 会比在py例加载慢很多, 内存使用多
    idx = np.load(idx_file)
    # labels = np.load(label_file)
    print("mat[1:1000].shape:", size(mat[1:1000]))
    mat_2 = np.save(mat_file_2, mat[1:1000])
    idx_2 = np.save(idx_file_2, idx[1:1000])
    # labels_2 = np.save(labels_file_2, labels[:1000])

    # get_label_mat_2(mat, idx, labels)

end

main()
