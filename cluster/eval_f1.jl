#=
eval cluster 
valse测试集, 人脸聚类测试集. 70184img, 3302id.
# https://blog.csdn.net/SAM2un/article/details/85013340 
# 打开图片,显示图片失败. julia图片处理太垃圾了
=# 
using PyCall
using ProgressMeter
using Printf 
using ImageMagick, FileIO
# using Images, ImageDraw, ImageView


#=
function merge(pics, info, output)  # plot
    wid = 10
    height = 5
    show_imsize = 200
    merge_img = Image.new('RGB', (wid * show_imsize, height * show_imsize))  #, 0xffffff
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf', 18)
    draw = ImageDraw.Draw(merge_img)
    img_sfx = "/data/yongzhang/cluster/test_2/"
    for (i, pic) in enumerate(pics)
        # merge_img = paste(merge_img, pic, ((i % wid) * show_imsize, (i / wid) * show_imsize), show_imsize)
        pos = ((i % wid) * show_imsize, (i / wid) * show_imsize)
        fn = joinpath(img_sfx, pic)
        img1 = Image.open(fn).resize((show_imsize, show_imsize), Image.BICUBIC)  # 每个人脸小图片是 100*100
        merge_img.paste(img1, pos)
    end
    draw.text((0, 0), str(info), fill="red", font=font)
    merge_img.save(output, quality=100)
end
=#


function pad_display(img1, img2)
    img1h = length(axes(img1, 1))
    img2h = length(axes(img2, 1))
    mx = max(img1h, img2h);
    grid = hcat(vcat(img1, zeros(RGB{Float64}, max(0, mx - img1h), length(axes(img1, 2)))),
        vcat(img2, zeros(RGB{Float64}, max(0, mx - img2h), length(axes(img2, 2)))))
end

function f_score(cluster, labels, imglist)
    TP = 0  # Threads.Atomic{Int64}(0)
    FP = 0  # Threads.Atomic{Int64}(0)
    TN = 0  # Threads.Atomic{Int64}(0)
    FN = 0  # Threads.Atomic{Int64}(0)
    img_sfx = "/data/yongzhang/cluster/test_2/"
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
                    # println(i, j)
                    if FP<10  # 预测为一个id, gt 不是一个id
                        # @printf("%s, %s\n", i, j)
                        pics = [imglist[i], imglist[j]]
                        dir_1 = "/data/yongzhang/cluster/test_2/"
                        output = dir_1 * "$i-$j.jpg"
                        # merge(pics, "FP, $i, $j", output) 
                        fn1 = joinpath(img_sfx, imglist[i])
                        img1 = load(fn1)  # 在ubuntu14.04上load png img报错, 
                        fn2 = joinpath(img_sfx, imglist[j])
                        println(fn1, "  ", fn2)
                        img2 = load(fn2)
                        imshow(img2)
                        grid = pad_display(img1, img2)  # OK
                        # grid = StackedView(img1, img2)  # 
                        # # ImageDraw.draw!(grid)
                        imshow(grid)
                        save(output, grid)

                    end
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

function test_1()
    dir_1 = "/data/yongzhang/cluster/test_2"
    labels = readlines(joinpath(dir_1, "new_labels.txt"))  #    new_labels
    labels = [parse(Int,a) for a in labels]

    cluster = readlines(joinpath(dir_1, "cluster_3.txt"))   # cluster_3
    cluster = [parse(Int,a) for a in cluster]

    imglist_file = joinpath(dir_1, "imglist.txt")
    imglist = readlines(imglist_file)

    @time f_score(cluster, labels, imglist)
end


function eval_1()
    # eval = pyimport("utils.eval_1")
    # np = pyimport("numpy")
    # pd = pyimport("pandas")

    pushfirst!(PyVector(pyimport("sys")."path"), "")

    py"""
    import os
    import numpy as np
    import pandas as pd
    from utils import eval_1
 
    dir_1 = "/data2/zhangyong/data/pk/pk_13/output_1"
    cluster_path = os.path.join(dir_1, "out_1/out_1_21.csv")
    labels_pred_df = pd.read_csv(cluster_path, names=["obj_id", "person_id"])
    
    gt_path = os.path.join(dir_1, "merged_all_out_1_1_1_21-small_1.pkl")
    gt_sorted_df = pd.read_pickle(gt_path)

    labels_true, labels_pred = eval_1.align_obj(gt_sorted_df, labels_pred_df)

    print(cluster_path)
    metric, info = eval_1.eval(labels_true, labels_pred, is_show=False)
    print(info)
    """
end


eval_1()


#=
用时在10s内
cluster_3.txt:  单链层次聚类. do2.py
fscore:0.9931 acc/RI:0.999958 precision:0.9907 recall:0.9955 count_pairs:2462861836   5.510634 seconds


=#
