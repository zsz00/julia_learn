

def get_label_mat_2(mat, idx, labels):
    """
    label: 1862120   -1 代表是干扰项,-2代表是删除的样本,其他的就是类别id
    评测的时候正样本就是所有》=0的类别的组合，负样本就是所有正样本*（正样本+干扰项）
    """
    # print("id count:", len(set(labels[:, 0])), max(labels), labels.shape)
    # out_file = open("fp_out/fp_list2.txt", 'a')
    threshold = 0.6
    count_T = 0
    count_F = 0
    count_P = 0
    count_TP = 0
    count_FP = 0
    s_1 = ""
    fp = []
    for i in tqdm(range(len(labels))):  # tqdm
        if labels[i] == -2:
            continue

        idx_1 = np.where((labels == labels[i]) &(labels[i]!=-1)& (labels != -2))[0]  # 正样本
        idx_2 = np.where((labels != labels[i]) & (labels != -2))[0]  # 负样本
        idx_1_1 = np.where(idx_1 > i)[0]
        idx_2_1 = np.where(idx_2 > i)[0]
        idx_1 = idx_1[idx_1_1]
        idx_2 = idx_2[idx_2_1]
        # print("gt: i:%s, id:%s, T:%s, F:%s, %s" % (
        #     i, labels[i][0], idx_1.shape[0], idx_2.shape[0], idx_1.shape[0] + idx_2.shape[0]))
        count_T += idx_1.shape[0]
        count_F += idx_2.shape[0]
        # print(idx_1[:15])
        # continue
        sim = mat[i]
        idx_p_1 = np.where(sim > threshold)[0]
        idx_p_2 = idx[i][idx_p_1]
        # idx_4 = idx_3[np.argsort(idx_3)]  # P
        idx_6 = np.where(idx_p_2 > i)[0]
        idx_4 = idx_p_2[idx_6]

        # print("P:", len(idx_4), idx_4[:10])  # idx_4[:10]
        count_P += len(idx_4)

        count_2 = 0
        count_3 = 0
        # s_1 = ""
        for idx_5 in idx_4:
            if idx_5 in idx_1:
                count_2 += 1
                # print("TP:", idx_5)
            else:
                count_3 += 1
                # print("FP:", idx_5)
                # s_1 += "%s %s\n"%(i, idx_5)  # 慢
                fp.append([i, idx_5])
        # if s_1 != "":
        #     print(s_1, file=out_file)
        # print("TP:", count_2, "FP:", count_3)
        count_TP += count_2
        count_FP += count_3
    np.savetxt("fp_2.txt", fp, fmt='%d')
    # print(s_1, file=out_file)
    print("count_T, count_F, count_P, count_TP, count_FP:", count_T, count_F, count_P, count_TP, count_FP)
    TPR = count_TP / float(count_T)
    FPR = count_FP / float(count_F)
    print(threshold, TPR, FPR)  # 2:23:00
    return TPR, FPR
