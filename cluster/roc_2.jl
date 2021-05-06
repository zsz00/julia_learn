# 评估, 取FP.  2019.6.10
# 参考: https://github.com/diegozea/ROC.jl
using PyCall
using ProgressMeter
using RecipesBase  # 画图
# using HDF5
using Plots

np = pyimport("numpy")

function get_scores1(labels)
    # 2h.  threads 之后 很耗内存. 3min, 5min
    #= 
    labels_size:320066
    count_T:1802282 count_F:3800771746   
    count_T:7867148 count_F:26974823212 (0,)
    count_T:9104838 count_F:36179062068 (0,)
    count_T:7729213 count_F:26770920387 (0,)
    count_T:7412722 count_F:24997440284 (0,)
    =#
    count_T = Threads.Atomic{Int64}(0)
    count_F = Threads.Atomic{Int64}(0)  # 
    idx_T = []
    labels_size = size(labels)   # (1862120, 1)
    println("typeof(labels):", typeof(labels), " labels_size:", labels_size[1])
    Threads.@threads for i in range(1,labels_size[1],step=1)  # @showprogress  Threads.@threads
        if labels[i] == -1
            continue
        end
        # println(i, " id:", labels[i], typeof(labels[i]))
        idx_1 = findall(x->(x==labels[i]) , labels[i+1:end])  # 正样本  T
        idx_1_1 = idx_1.+i   # idx   Tuple
        # println("idx_1_1:", size(idx_1_1), " ", idx_1_1[1]) 
        # push!(idx_T,idx_1_1)  # 有问题
        # count_T += length(idx_1_1)
        Threads.atomic_add!(count_T, length(idx_1_1))

        idx_2_1 = findall(x->(x!=labels[i])&&(x!=-1), labels[i+1:end])   # 负样本  F  慢,只用算一次. 332576170328
        # idx_2_2 = findall(x->(x==-1), labels[1:i])
        # idx_2_1 = Tuple.(idx_2)
        # idx_2_2=findall(x->x>(i,1), idx_2_1)
        # idx_2_3 = idx_2_1[idx_2_2]
        # count_F += length(idx_2_1)
        Threads.atomic_add!(count_F, length(idx_2_1))
    end
    count_T = count_T.value
    count_F = count_F.value
    println("count_T:", count_T, " count_F:", count_F, " ", size(idx_T))
    return count_T,count_F,idx_T 
end

function get_scores2(qurey, gallary)
    # 2h.  threads 之后 很耗内存. 3min, 5min
    #= 
    labels_size:320066
    count_T:1802282 count_F:3800771746   
    count_T:7867148 count_F:26974823212 (0,)
    count_T:9104838 count_F:36179062068 (0,)
    count_T:7729213 count_F:26770920387 (0,)
    count_T:7412722 count_F:24997440284 (0,)
    =#
    count_T = Threads.Atomic{Int64}(0)
    count_F = Threads.Atomic{Int64}(0)  # 
    idx_T = []
    labels_size = size(labels)   # (1862120, 1)
    println("typeof(labels):", typeof(labels), " labels_size:", labels_size[1])
    Threads.@threads for i in range(1,labels_size[1],step=1)  # @showprogress  Threads.@threads
        if labels[i] == -1
            continue
        end
        # println(i, " id:", labels[i], typeof(labels[i]))
        idx_1 = findall(x->(x==labels[i]) , labels[i+1:end])  # 正样本  T
        idx_1_1 = idx_1.+i   # idx   Tuple
        # println("idx_1_1:", size(idx_1_1), " ", idx_1_1[1]) 
        # push!(idx_T,idx_1_1)  # 有问题
        # count_T += length(idx_1_1)
        Threads.atomic_add!(count_T, length(idx_1_1))

        idx_2_1 = findall(x->(x!=labels[i])&&(x!=-1), labels[i+1:end])   # 负样本  F  慢,只用算一次. 332576170328
        # idx_2_2 = findall(x->(x==-1), labels[1:i])
        # idx_2_1 = Tuple.(idx_2)
        # idx_2_2=findall(x->x>(i,1), idx_2_1)
        # idx_2_3 = idx_2_1[idx_2_2]
        # count_F += length(idx_2_1)
        Threads.atomic_add!(count_F, length(idx_2_1))
    end
    count_T = count_T.value
    count_F = count_F.value
    println("count_T:", count_T, " count_F:", count_F, " ", size(idx_T))
    return count_T,count_F,idx_T 
end


function get_scores(mat, idx, idx_T, labels, threshold)
    count_P = Threads.Atomic{Int64}(0)
    count_TP = Threads.Atomic{Int64}(0)
    count_FP = Threads.Atomic{Int64}(0)
    Threads.@threads for i in range(1, labels_size[1], step=1)
        sim = mat[i,1:end]
        idx_p_1 = findall(x-> x>threshold, sim)
        # println("idx_p_1:", size(idx_p_1), " ", idx_p_1)
        idx_p_2 = idx[i,1:end][idx_p_1]
        idx_p_2 = idx_p_2 .+ 1
        # println("idx_p_2:", size(idx_p_2), idx_p_2)
        idx_p_3=findall(x-> x>i, idx_p_2)
        idx_p_4 = idx_p_2[idx_p_3]                           # P
        # count_P += length(idx_p_4)
        Threads.atomic_add!(count_P, length(idx_p_4))
        # println("P:", size(idx_p_4), " ")  # , idx_p_4

        idx_1_3 = idx_T[i]
        count_2 = 0
        count_3 = 0
        for idx_5 in idx_p_4
            if (idx_5,1) in idx_1_3                         # TP
                count_2 += 1
                # println("TP:", idx_5)
            else
                if labels[idx_5] != -2                      # FP
                    count_3 += 1
                    # println("FP:", idx_5)
                    # push!(fp,[i-1, idx_5-1])  # 并行有点问题
                end
            end
        end
        # println(i, " id:", labels[i]," TP:", count_2, " FP:", count_3)
        Threads.atomic_add!(count_TP, count_2)
        Threads.atomic_add!(count_FP, count_3)
    end
    
    count_P = count_P.value
    count_TP = count_TP.value
    count_FP = count_FP.value
    # println(threshold, "count_P:", count_P," count_TP:",count_TP," count_FP:", count_FP)
    # np.savetxt("fp_out/out_dengqili/j_$(threshold)_2.txt", fp, fmt="%d")
    return count_TP, count_FP
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
    # label_file = joinpath(dir_2, "deepglint.npy")
    label_file = joinpath(dir_2, "deepglint_1.txt")  # deepglint_1  deepglint_900

    mat = np.load(mat_file)   # 会比在py例加载慢很多, 内存使用多
    idx = np.load(idx_file)
    # labels = np.load(label_file)
    labels = readlines(label_file)
    labels = [parse(Int,a) for a in labels]
    labels = hcat(labels)

    thresholds = Array(range(0.99,0.01, step=-0.01))

    n_thresholds = length(thresholds)
    FPR = Array{Float64}(undef, n_thresholds)
    TPR = Array{Float64}(undef, n_thresholds)

    count_T,count_F,idx_T = get_label_mat_1(labels)
    # count_T,count_F,idx_1_3 = 11162233,332576170328

    for (i, threshold) in enumerate(thresholds)
        count_TP, count_FP = get_scores(mat, idx,idx_T,labels, threshold)
        # tar=TPR = TP/T  far=FPR = FP/F
        TPR[i] = tpr = count_TP/count_T
        FPR[i] = fpr = count_FP/count_F
        if round(fpr*1e-5) == 1000  # ?
            println("==================")
            println("threshold:", threshold, " TPR:", tpr, " FPR:", fpr, " fp:", count_FP)
        end
        println(threshold, " count_T:", count_T, " count_F:", count_F," count_TP:",count_TP," count_FP:", count_FP)
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


function main()
    dir_2 = "/data/yongzhang/cluster/data_3/clean/out_disk_2_jxx_4"
    dir_1 = "/data/yongzhang/cluster/data_3/clean/out_disk_2_jxx_2"
    mat_file = joinpath(dir_2, "disk_2_jxx_mat.npy")  # top_k=1000.  相似度矩阵
    idx_file = joinpath(dir_2, "disk_2_jxx_idx.npy")
    # label_file = joinpath(dir_2, "postproc_jxx_0.45.txt")  # labels_test_1_1 postproc_jxx_0.45
    label_file = joinpath(dir_1, "labels_test_1_1.txt")
    mat = np.load(mat_file)   # 会比在py里加载慢很多, 内存使用多
    idx = np.load(idx_file)
    # labels = np.load(label_file)
    labels = readlines(label_file)
    labels = [parse(Int,a) for a in labels]
    labels = hcat(labels)
    threshold = 0.6
    count_T,count_F,idx_T = get_scores1(labels)
    return 
    
    count_TP,count_FP = get_scores(mat, idx,idx_T,labels, threshold)  # 计算一个阈值的
    tpr = count_TP/count_T
    fpr = count_FP/count_F
    println("threshold:", threshold, " TPR:", tpr, " FPR:", fpr, " fp:", count_FP)
end

function main2()
    roc_data = roc()
    auc = AUC(roc_data)
    println("auc:", auc)
    # dummy(roc_data)
    # plot!(dummy(roc_data), label="good")  # 画图失败
end

function temp()
    dir_2 = "/data/yongzhang/cluster/test_1"
    label_file = joinpath(dir_2, "deepglint.npy")
    labels = np.load(label_file)
    count_T,count_F,idx_T = get_scores1(labels)
    println("count_F:", count_F)
end

@time main()
# @time main2()
# @time temp()

#=
label: 1862120   -1 代表是干扰项,-2代表是删除的样本,其他的就是类别id
评测的时候正样本就是所有》=0的类别的组合，负样本就是所有正样本*（正样本+干扰项）
=#
#=
export JULIA_NUM_THREADS=40
julia roc_2.jl 



------------------------------------------------------------
T: 256157*256156/2=32808076246
F: 256157*1426452/2=182697832482
qingcheng_F:332576170328
count_F:181560886682
count_F:365373084341

0.6 count_T:11162233 count_F:365373084341 count_P:6403061 count_TP:6201927 count_FP:990
threshold:0.6 TPR:0.5556170526094555 FPR:2.7095591942290976e-9    1.4h


0.4 count_T:11154520 count_TP:10372720 count_FP:55734  -1:29318    26416   
TPR:316e-4, 0.93  FPR: 3.0e-7
0.6 count_T:11145043 count_TP:6197680 count_FP:990    -1:2
0.556

0.6 count_T:11162233 count_F:181560886682 count_P:6403061 count_TP:6201927 count_FP:990  12min
threshold:0.6 TPR:0.5556170526094555 FPR:5.452716265557591e-9


auc:0.9960541832114193

tar@far=1e-8 :  neg pairs=fp: 3000



=#


