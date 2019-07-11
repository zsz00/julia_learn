# 评估, 取FP.  2019.6.10
using PyCall
using ProgressMeter
# using HDF5


np = pyimport("numpy")

function get_scores(mat, idx, labels, threshold)
    count_T = Threads.Atomic{Int64}(0)
    count_F = 181560886682  # Threads.Atomic{Int64}(0)
    count_P = Threads.Atomic{Int64}(0)
    count_TP = Threads.Atomic{Int64}(0)
    count_FP = Threads.Atomic{Int64}(0)
    fp = []
    labels_size = size(labels)   # (1862120, 1)
    println("threshold:", threshold, " size(mat):", size(mat))
    Threads.@threads for i in range(1,labels_size[1],step=1)  # @showprogress  Threads.@threads
        if labels[i] in (-1,-2)
            continue
        end
        # if i % 1000==1
        #     println(i, " id:", labels[i])
        # end
        idx_1 = findall(x->(x==labels[i]) && (x !=-2), labels)  # 正样本
        idx_1_1 = Tuple.(idx_1)   # idx
        # println("idx_1:", size(idx_1), " ", idx_1_1, "\n", idx_1_1[1])
        idx_1_2=findall(x->x>(i,1), idx_1_1)
        idx_1_3 = idx_1_1[idx_1_2]
        # println("idx_1_3:", size(idx_1_3), " ")  # , idx_1_3
        # count_T += length(idx_1_3)
        Threads.atomic_add!(count_T, length(idx_1_3))
        
        sim = mat[i,1:end]
        idx_p_1 = findall(x-> x>threshold, sim)                  # P
        # println("idx_p_1:", size(idx_p_1), " ", idx_p_1)
        idx_p_2 = idx[i,1:end][idx_p_1]
        idx_p_2 = idx_p_2 .+ 1
        # println("idx_p_2:", size(idx_p_2), idx_p_2)
        idx_p_3=findall(x-> x>i, idx_p_2)
        idx_p_4 = idx_p_2[idx_p_3]   # p
        # count_P += length(idx_p_4)
        Threads.atomic_add!(count_P, length(idx_p_4))
        # println("P:", size(idx_p_4), " ")  # , idx_p_4

        count_2 = 0
        count_3 = 0
        for idx_5 in idx_p_4
            if (idx_5,1) in idx_1_3
                count_2 += 1
                # println("TP:", idx_5)
            else
                if labels[idx_5] != -2
                    count_3 += 1
                    idx_fp_1=findall(x-> x==idx_5, idx_p_2)
                    fp_sim = sim[idx_fp_1]
                    push!(fp,[i-1, idx_5-1])          # 并行有点问题  , fp_sim[1]
                end
            end
        end
        # println(i, " id:", labels[i]," TP:", count_2, " FP:", count_3)

        Threads.atomic_add!(count_TP, count_2)
        Threads.atomic_add!(count_FP, count_3)

    end

    count_T = count_T.value
    # count_F = count_F.value
    count_P = count_P.value
    count_TP = count_TP.value
    count_FP = count_FP.value
    
    tpr = count_TP/count_T
    fpr = count_FP/count_F
    println(threshold, " count_T:", count_T," count_F:", count_F," count_P:",count_P, " count_TP:",count_TP," count_FP:", count_FP)
    println("threshold:", threshold, " TPR:", tpr, " FPR:", fpr)

    # h5write("fp_out/test_1.h5", "fp", fp)
    out_dir = "/data/yongzhang/cluster/test_1"
    println("=========:", size(fp))
    np.savetxt(joinpath(out_dir, "fp_out/out_r124/j_$(threshold)_3.txt"), fp, fmt="%d")  # fmt=["%d","%d","%s"]

end


function main()
    # dir_2 = "/data/yongzhang/22/test_1"
    dir_2 = "/data/yongzhang/cluster/test_1"
    # mat_file = joinpath(dir_2, "out_dengqili/out_5/mat_2.npy")  # top_k=1000
    # idx_file = joinpath(dir_2, "out_dengqili/out_5/idx_2.npy")
    # mat_file = joinpath(dir_2, "out_dengqili/out_1/mat.npy")  # top_k=1000
    # idx_file = joinpath(dir_2, "out_dengqili/out_1/idx.npy")
    mat_file = joinpath(dir_2, "out_r124/out_1/mat.npy")  # top_k=1000
    idx_file = joinpath(dir_2, "out_r124/out_1/idx.npy")
    label_file = joinpath(dir_2, "deepglint.npy")

    mat = np.load(mat_file)   # 会比在py例加载慢很多, 内存使用多
    idx = np.load(idx_file)
    labels = np.load(label_file)

    threshold = 0.4
    get_scores(mat, idx, labels, threshold)

end

@time main()


#=
export JULIA_NUM_THREADS=4
julia get_mat.jl 
多线程的 count统计不准确, 输出的顺序不一样, 但是结果是一样的.
mem:14G, 正常的
out_dengqili:
0.6 count_T:11162233 count_TP:1789258 count_FP:1435476    40min  cpu:100%
2715.293647 seconds (31.18 M allocations: 19.764 GiB, 0.13% gc time)

0.7 count_T:11162233 count_TP:768453 count_FP:600909  cpu:100%
2535.467317 seconds (25.72 M allocations: 19.031 GiB, 0.07% gc time)

0.8 count_T:11131742 count_TP:117715 count_FP:89695  10min  cpu:400%
664.664287 seconds (17.95 M allocations: 18.334 GiB, 0.12% gc time)
-------
0.6 count_T:11162233 count_TP:6201927 count_FP:201134
2636.597931 seconds (24.37 M allocations: 19.264 GiB, 0.05% gc time)

0.7 count_T:11162233 count_TP:2667991 count_FP:56229
2522.424604 seconds (22.71 M allocations: 18.831 GiB, 0.05% gc time)

0.8 count_T:11162233 count_TP:407334 count_FP:6136
2593.702428 seconds (21.58 M allocations: 18.576 GiB, 0.03% gc time)

------------------
out_NothingLC:
0.6 count_T:11162233 count_TP:5408071 count_FP:160652   40min  cpu:100%
2651.369333 seconds (24.00 M allocations: 19.158 GiB, 0.06% gc time)

0.7 count_T:11162233 count_TP:1889634 count_FP:35770    40min  cpu:100%
2441.002142 seconds (22.37 M allocations: 18.739 GiB, 0.04% gc time)

0.8 count_T:11162233 count_TP:225559 count_FP:3322      40min  cpu:100%
2679.599290 seconds (21.48 M allocations: 18.557 GiB, 0.04% gc time)


为什么count_FP 差异这么大??  是相似度矩阵的问题??

-----------------------------
out_dengqili:
0.4 count_T:11154520 count_TP:10372720 count_FP:55734  -1:29318    26416
835.157405 seconds (21.03 M allocations: 19.401 GiB, 0.12% gc time)

0.5 count_T:11161655 count_TP:8929817 count_FP:6075   -1:201
1406.407383 seconds (19.68 M allocations: 19.194 GiB, 0.09% gc time)

0.6 count_T:11145043 count_TP:6197680 count_FP:990    -1:2
687.092225 seconds (19.09 M allocations: 18.897 GiB, 0.11% gc time)

0.7 count_T:11161957 count_TP:2667863 count_FP:47     -1:1
1343.848945 seconds (18.21 M allocations: 18.537 GiB, 0.07% gc time)

0.8 count_T:Base.Threads.Atomic{Int64}(11162233) count_P:Base.Threads.Atomic{Int64}(413470) count_TP:406830 count_FP:10
353.802312 seconds (17.37 M allocations: 18.301 GiB, 0.20% gc time)


0.6 count_T:11162233 count_P:6403061 count_TP:6201927 count_FP:990
0.5556
=#
