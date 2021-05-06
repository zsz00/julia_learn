# 评估, 取FP.  2019.6.10
# 参考: https://github.com/diegozea/ROC.jl
using PyCall
using ProgressMeter
using RecipesBase  # 画图
# using HDF5
using Plots

np = pyimport("numpy")

function get_scores(mat, idx, labels, threshold) 
    count_T = Threads.Atomic{Int64}(0)
    count_F = 330145575217  # Threads.Atomic{Int64}(0)  # 330145575217  # 332576170328
    count_P = Threads.Atomic{Int64}(0)
    count_TP = Threads.Atomic{Int64}(0)
    count_FP = Threads.Atomic{Int64}(0)
    fp = []
    labels_size = size(labels)   # (1862120, 1)
    # println("threshold:", threshold, " size(mat):", size(mat), " typeof(labels):", typeof(labels), " size(labels):",size(labels))
    Threads.@threads for i in range(1,labels_size[1],step=1)  # @showprogress  Threads.@threads
        if labels[i] in (-1,-2)
            continue
        end
        # println(i, " id:", labels[i], typeof(labels[i]))
        idx_1 = findall(x->(x==labels[i]) && (x !=-2), labels)  # 正样本  T
        idx_1_1 = Tuple.(idx_1)   # idx
        # println("idx_1:", size(idx_1), " ", idx_1_1[1])  # , idx_1_1, "\n",
        idx_1_2=findall(x->x>(i,1), idx_1_1)
        idx_1_3 = idx_1_1[idx_1_2]
        # println("idx_1_3:", size(idx_1_3), " ")  # , idx_1_3
        # count_T += length(idx_1_3)
        Threads.atomic_add!(count_T, length(idx_1_3))

        # idx_2 = findall(x->(x!=labels[i]) && (x !=-2), labels)   # 负样本  F  慢,只用算一次.
        # idx_2_1 = Tuple.(idx_2)
        # idx_2_2=findall(x->x>(i,1), idx_2_1)
        # idx_2_3 = idx_2_1[idx_2_2]
        # Threads.atomic_add!(count_F, length(idx_2))

        sim = mat[i,1:end]
        idx_p_1 = findall(x-> x>threshold, sim)
        # println("idx_p_1:", size(idx_p_1), " ", idx_p_1)
        idx_p_2 = idx[i,1:end][idx_p_1]
        idx_p_2 = idx_p_2 .+ 1
        # println("idx_p_2:", size(idx_p_2), idx_p_2)
        idx_p_3=findall(x-> x>i, idx_p_2)
        idx_p_4 = idx_p_2[idx_p_3]                                 # P
        # count_P += length(idx_p_4)
        Threads.atomic_add!(count_P, length(idx_p_4))
        # println("P:", size(idx_p_4), " ")  # , idx_p_4

        count_2 = 0
        count_3 = 0
        for idx_5 in idx_p_4
            # if (idx_5,1) in idx_1_3                         # TP
            if (idx_5,1) in idx_1_3
                count_2 += 1
                # println("TP:", idx_5)
            else
                if labels[idx_5] != -2                      # FP
                    count_3 += 1
                    # if labels[idx_5] == -1               
                    #     println("=====================:-1 ", i-1, " ", idx_5-1)
                    # end
                    # println("FP:", idx_5)
                    # push!(fp,[i-1, idx_5-1])  # 并行有点问题
                end
            end
        end
        # println(i, " id:", labels[i]," TP:", count_2, " FP:", count_3)
        Threads.atomic_add!(count_TP, count_2)
        Threads.atomic_add!(count_FP, count_3)
        # if i>10
        #     break
        # end
    end

    count_T = count_T.value
    # count_F = count_F.value
    count_P = count_P.value
    count_TP = count_TP.value
    count_FP = count_FP.value
    
    tpr = count_TP/count_T
    fpr = count_FP/count_F
    println(threshold, " count_T:", count_T, " count_F:", count_F," count_P:", count_P," count_TP:",count_TP," count_FP:", count_FP)
    # println("threshold:", threshold, " TPR:", tpr, " FPR:", fpr)

    #=
    tar=TPR = TP/T  far=FPR = FP/F
    =#
    # np.savetxt("fp_out/out_dengqili/j_$(threshold)_2.txt", fp, fmt="%d")
    return tpr,fpr, count_FP
end

struct ROCData{T <: Real}
	thresholds::Vector{T}
	# P::Int
	# N::Int
	# TP::Vector{Int}
	# # TN::Vector{Int}
	# FP::Vector{Int}
	# FN::Vector{Int}
	FPR::Vector{Float64}
	TPR::Vector{Float64}
end

function roc()
    dir_2 = "/data/yongzhang/cluster/test_1"
    # mat_file = joinpath(dir_2, "out_dengqili/out_5/mat_2.npy")  # top_k=1000
    # idx_file = joinpath(dir_2, "out_dengqili/out_5/idx_2.npy")
    mat_file = joinpath(dir_2, "out_dengqili/out_1/mat.npy")  # top_k=1000
    idx_file = joinpath(dir_2, "out_dengqili/out_1/idx.npy")
    # mat_file = joinpath(dir_2, "out_NothingLC/out_1/mat.npy")  # top_k=1000
    # idx_file = joinpath(dir_2, "out_NothingLC/out_1/idx.npy")
    mat_file = joinpath(dir_2, "out_r124/out_1/mat.npy")  # top_k=1000
    idx_file = joinpath(dir_2, "out_r124/out_1/idx.npy")
    # label_file = joinpath(dir_2, "deepglint.npy")
    label_file = joinpath(dir_2, "deepglint_6000.txt")  # deepglint_1  deepglint_900

    mat = np.load(mat_file)   # 加载会比在py例加载慢很多, 内存使用多
    idx = np.load(idx_file)
    # labels = np.load(label_file)
    labels = readlines(label_file)
    labels = [parse(Int,a) for a in labels]
    labels = hcat(labels)

    # thresholds = Array(range(0.1,0.9, step=0.1))
    # thresholds = Array(range(0.9,0.1, step=-0.05))
    # thresholds = Array(range(0.548,0.546, step=-0.0005))
    thresholds = Array(range(0.5,0.2, step=-0.01))
    # thresholds = Array(range(0.44,0.42, step=-0.005))
    # thresholds = Array(range(0.37,0.35, step=-0.005))
    # thresholds = [0.573, 0.432, 0.37]
    # thresholds = Array(range(0.485,0.465, step=-0.005))
    n_thresholds = length(thresholds)
    FPR = Array{Float64}(undef, n_thresholds)
    TPR = Array{Float64}(undef, n_thresholds)
    for (i, threshold) in enumerate(thresholds)
        tpr, fpr, count_FP = get_scores(mat, idx, labels, threshold)
        # tar = tpr = 1-thresholds[i]  
        # far = fpr = 1-thresholds[i] =1e-8
        TPR[i] = tpr
        FPR[i] = fpr
        println("threshold:", threshold, " TPR:", tpr, " FPR:", fpr, " fp:", count_FP)
    end
    ROCData{eltype(thresholds)}(thresholds, FPR, TPR)
end

function AUC(roc::ROCData)
	auc=zero(Float64)
	for i in 2:length(roc.thresholds)
		dx = roc.FPR[i] - roc.FPR[i-1]
		dy = roc.TPR[i] - roc.TPR[i-1]
		auc += ( (dx*roc.TPR[i-1]) + (0.5*dx*dy) )
    end
    auc += ( ((1-roc.FPR[length(roc.thresholds)])*roc.TPR[length(roc.thresholds)]) + 
    0.5*(1-roc.FPR[length(roc.thresholds)])*(1-roc.TPR[length(roc.thresholds)]) )
	auc
end

@recipe function dummy(curve::ROCData)   # @recipe    ?????????
    xlim := (0,1)
    ylim := (0,1)
    xlab := "false positive rate"
    ylab := "true positive rate"
    title --> "Receiver Operator Characteristic"
    @series begin
        color --> :black
        linestyle --> :dash
        label := ""
        [0, 1], [0, 1]
    end
    @series begin
        curve.FPR, curve.TPR
    end
end


function main1()
    # dir_2 = "/data/yongzhang/22/test_1"
    dir_2 = "/data/yongzhang/cluster/test_1"
    # mat_file = joinpath(dir_2, "out_dengqili/out_5/mat_2.npy")  # top_k=1000
    # idx_file = joinpath(dir_2, "out_dengqili/out_5/idx_2.npy")
    mat_file = joinpath(dir_2, "out_dengqili/out_1/mat.npy")  # top_k=1000.  相似度矩阵
    idx_file = joinpath(dir_2, "out_dengqili/out_1/idx.npy")
    label_file = joinpath(dir_2, "deepglint.npy")

    mat = np.load(mat_file)   # 会比在py例加载慢很多, 内存使用多
    idx = np.load(idx_file)
    labels = np.load(label_file)
    threshold = 0.6
    tpr, fpr, count_FP = get_scores(mat, idx, labels, threshold)  # 计算一个阈值的

end

function main()
    # dir_2 = "/data/yongzhang/22/test_1"
    dir_2 = "/data/yongzhang/cluster/data_3/clean/out_disk_2_jxx_3"
    mat_file = joinpath(dir_2, "disk_2_jxx_mat.npy")  # top_k=1000.  相似度矩阵
    idx_file = joinpath(dir_2, "disk_2_jxx_idx.npy")
    label_file = joinpath(dir_2, "labels_test_1_1.txt")

    mat = np.load(mat_file)   # 会比在py例加载慢很多, 内存使用多
    idx = np.load(idx_file)
    # labels = np.load(label_file)
    labels = readlines(label_file)
    labels = [parse(Int,a) for a in labels]
    labels = hcat(labels)
    threshold = 0.6
    tpr, fpr, count_FP = get_scores(mat, idx, labels, threshold)  # 计算一个阈值的
    println("threshold:", threshold, " TPR:", tpr, " FPR:", fpr, " fp:", count_FP)
    
end

function main2()
    roc_data = roc()
    auc = AUC(roc_data)
    println("auc:", auc)
    # dummy(roc_data)
    # plot!(dummy(roc_data), label="good")  # 画图失败

end


@time main()
# @time main2()

#=
label: 1862120   -1 代表是干扰项,-2代表是删除的样本,其他的就是类别id
评测的时候正样本就是所有》=0的类别的组合，负样本就是所有正样本*（正样本+干扰项）
=#
#=
julia get_mat.jl 
多线程的用 + 做count统计不准确, 输出的顺序不一样, 相关I/O的并行有问题. push!() 也是IO,并行会崩溃.
可以用Threads.Atomic 做原子操作, 也是并行,但是会降低点速度(影响不大).
mem:14G, 正常的

为什么count_FP 差异这么大??  是相似度矩阵的问题??
out_NothingLC  应该好些
-----------------------------
T: 256157*256156/2=32808076246
F: 256157*1426452/2=182697832482
count_F:181560886682
qingcheng_F:332576170328

0.6 count_T:11162233 count_F:365373084341 count_P:6403061 count_TP:6201927 count_FP:990
threshold:0.6 TPR:0.5556170526094555 FPR:2.7095591942290976e-9

0.4 count_T:11154520 count_TP:10372720 count_FP:55734  -1:29318    26416   
TPR:316e-4, 0.93  FPR: 3.0e-7
0.6 count_T:11145043 count_TP:6197680 count_FP:990    -1:2
0.556

0.6 count_T:11162233 count_F:181560886682 count_P:6403061 count_TP:6201927 count_FP:990  12min
threshold:0.6 TPR:0.5556170526094555 FPR:5.452716265557591e-9


threshold:0.9 TPR:0.00288849014350444 FPR:5.50779420763393e-11
threshold:0.85 TPR:0.008857725869008468 FPR:5.50779420763393e-11
threshold:0.8 TPR:0.03649216066355182 FPR:5.50779420763393e-11
threshold:0.75 TPR:0.111345104514482 FPR:7.710911890687502e-11
threshold:0.7 TPR:0.23901946859557582 FPR:2.588663277587947e-10
threshold:0.65 TPR:0.3973196940074625 FPR:1.4155031113619201e-9
threshold:0.6 TPR:0.5556170526094555 FPR:5.452716265557591e-9
threshold:0.55 TPR:0.6925121523623454 FPR:1.561459657864219e-8
threshold:0.5 TPR:0.8000232569952626 FPR:3.345984981137613e-8
threshold:0.45 TPR:0.8777849378345712 FPR:6.94587927524715e-8
threshold:0.4 TPR:0.9294668907198049 FPR:3.0699894133930764e-7
threshold:0.35 TPR:0.9614543971622882 FPR:2.567408699740688e-6
threshold:0.3 TPR:0.9796067686456643 FPR:2.191055062959433e-5
threshold:0.25 TPR:0.9891049577624835 FPR:0.0001687839741258441
threshold:0.2 TPR:0.9921134059824768 FPR:0.0005251710968288253
threshold:0.15 TPR:0.992116989494844 FPR:0.0005260703158345462
threshold:0.1 TPR:0.992116989494844 FPR:0.0005260703158345462   # 因为top_k=1000,FPR高不了. 后边的自己补齐
auc:0.0005196852709187022

auc:0.9960541832114193

本来是NothingLC更高, 期望是clean以后dengqili能更高一点.
tar@far=1e-8 :  neg pairs=fp: 3000


=#


