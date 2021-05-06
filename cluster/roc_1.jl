# 评估, 取FP.  2019.6.10
# 参考: https://github.com/diegozea/ROC.jl
using PyCall
using ProgressMeter
using RecipesBase  # 画图
using HDF5
using GR  # Plots
using Printf
using FileIO, JSON

np = pyimport("numpy")
# from scipy.sparse import csr_matrix,save_npz,load_npz

function get_scores(mat, idx, labels, threshold) 
    count_T = Threads.Atomic{Int64}(0)
    count_F = 24997440284  # Threads.Atomic{Int64}(0)  # 3800771746  26974823212
    count_P = Threads.Atomic{Int64}(0)
    count_TP = Threads.Atomic{Int64}(0)
    count_FP = Threads.Atomic{Int64}(0)
    fp = []
    labels_size = size(labels)   # (1862120, 1)
    # println("threshold:", threshold, " size(mat):", size(mat), " typeof(labels):", typeof(labels), " size(labels):",size(labels))
    Threads.@threads for i in range(1,labels_size[1],step=1)  # @showprogress  Threads.@threads
        if labels[i] == -1
            continue
        end
        
        idx_1 = findall(x->(x==labels[i]) , labels[i+1:end])  # 正样本  T
        idx_1_1 = idx_1.+i   # idx   Tuple
        # if i==1
        #     println(i, " id:", labels[i], " ", typeof(labels[i]))
        #     println("idx_1_1:", size(idx_1_1), " ", idx_1_1, "\n",labels[idx_1_1])
        # end
        # println("idx_1:", size(idx_1), " ", idx_1_1[1])  # , idx_1_1, "\n",
        # idx_1_2=findall(x->x>(i,1), idx_1_1)
        idx_1_3 = idx_1_1   # [idx_1_2]
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
        idx_p_2 = idx[i,1:end][idx_p_1]
        idx_p_2 = idx_p_2 .+ 1
        idx_p_3=findall(x-> x>i, idx_p_2)
        idx_p_4 = idx_p_2[idx_p_3]                                 # P
        # if i==1
        #     println("idx_p_1:", size(idx_p_1), " ", idx_p_1)
        #     println("idx_p_2:", size(idx_p_2), " ", idx_p_2)
        #     println("idx_p_4:", size(idx_p_4), " ", idx_p_4)
        #     println("sim:", sim[1:10])
        # end
        # count_P += length(idx_p_4)
        Threads.atomic_add!(count_P, length(idx_p_4))
        # println("P:", size(idx_p_4), " ")  # , idx_p_4

        count_2 = 0
        count_3 = 0
        for idx_5 in idx_p_4
            # if i==1
            #     println("idx_5:", idx_5)
            # end
            if idx_5 in idx_1_3                         # TP
            # if (idx_5,1) in idx_1_3
                count_2 += 1
                # println("TP:", idx_5)
            else
                if labels[idx_5] != -1                      # FP
                    count_3 += 1
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
    # println(threshold, " count_T:", count_T, " count_F:", count_F," count_P:", count_P," count_TP:",count_TP," count_FP:", count_FP)
    println("threshold:", threshold, " TPR:", tpr, " FPR:", fpr, " fp:", count_FP)
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

function roc(mat, idx, labels)
    # thresholds = Array(range(0.1,0.9, step=0.1))
    thresholds = Array(range(0.5,0.0, step=-0.01))  # 50
    # thresholds = Array(range(0.44,0.42, step=-0.005))
    n_thresholds = length(thresholds)
    FPR = Array{Float64}(undef, n_thresholds)
    TPR = Array{Float64}(undef, n_thresholds)
    for (i, threshold) in enumerate(thresholds)
        tpr, fpr, count_FP = get_scores(mat, idx, labels, threshold)
        # tar = tpr = 1-thresholds[i]  
        # far = fpr = 1-thresholds[i] =1e-8
        TPR[i] = tpr
        FPR[i] = fpr
        if round(fpr, digits=7)==1e-7
            println("TPR:", tpr, " @1e-7 ", fpr, " threshold:", threshold)
        elseif round(fpr, digits=6)==1e-6
            println("TPR:", tpr, " @1e-6 ", fpr, " threshold:", threshold)
        end
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

@recipe function roc_curve(curve::ROCData)   # @recipe    ?????????
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
    dir_2 = "/data/yongzhang/cluster/data_3/clean/out_disk_2_jxx_4"
    dir_1 = "/data/yongzhang/cluster/data_3/clean/out_disk_2_jxx_2"
    mat_file = joinpath(dir_2, "disk_2_jxx_mat.npy")  # topk=1000.  (n,topk)的相似度矩阵
    idx_file = joinpath(dir_2, "disk_2_jxx_idx.npy")
    # label_file = joinpath(dir_2, "postproc_jxx_0.45.txt")  # labels_test_1_1
    label_file = joinpath(dir_1, "labels_test_1_1.txt")
    mat = np.load(mat_file)   # 会比在py例加载慢很多, 内存使用多
    idx = np.load(idx_file)

    # labels = np.load(label_file)
    labels = readlines(label_file)
    labels = [parse(Int,a) for a in labels]
    labels = hcat(labels)
    # threshold = 0.35
    # tpr, fpr, count_FP = get_scores(mat, idx, labels, threshold)  # 计算一个阈值的
    # println("threshold:", threshold, " TPR:", tpr, " FPR:", fpr, " fp:", count_FP)
    rocdata = roc(mat, idx, labels)
    # roc_curve(rocdata)
    data = Dict("thresholds"=>rocdata.thresholds,"FPR"=>rocdata.FPR,"TPR"=>rocdata.TPR)
    data_json = JSON.json(data)
    open("rocdata.json", "w") do f
        write(f, data_json)
    end

    # plot!(rocdata)
    # savefig("myplot.png")
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
-----------------------------
多线程 
export JULIA_NUM_THREADS=40
ENV["JULIA_NUM_THREADS"]=40
julia roc_1.jl 

先执行roc_2.jl算出F, 再通过roc_1.jl画图.  roc_note_4是结果记录
可以把FP打出来,查看误识别
-----------------------------
1. threshold 为0时，达不到（1，1）. 
因为mat是(n，topk),所以P，TP都小，小于T，特别是在threshold很低的时候. mat是(n,n)就可以达到(1,1)了。
普通的阈值时TPR低，也是这个原因。 提高一下topk. topk=1000就行. 可以取出更多低相似度的mat值. 就可以画出漂亮的roc曲线了.
2. 计算TPR,FPR很慢
1371s=23min
需要改成GPU版的
3. 画图和存数有问题

4. T,F,P的计算有问题



-----------------------------



=#


