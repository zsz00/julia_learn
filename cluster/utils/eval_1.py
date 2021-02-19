import os, datetime, time, random
import numpy as np
import pandas as pd
import cv2
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn import metrics
# from PIL import Image, ImageDraw, ImageFont
# import matplotlib.pyplot as plt
import base64, json, struct
from io import BytesIO
import urllib.request, urllib.parse, urllib.error
from collections import Counter
from utils import util_1


def color(data, col='red'):
    # https://www.cnblogs.com/easypython/p/9084426.html
    if col == 'red':
        s = "\033[31m" + str(data) + "\033[0m"
    elif col == 'green':
        s = "\033[32m" + str(data) + "\033[0m"
    elif col == 'blue':
        s = "\033[34m" + str(data) + "\033[0m"
    else:
        s = data
    return s


def r(data):
    return round(data, 5)


def eval_1(X, labels, is_show=True):
    """
    无监督的评估. 计算比较慢.
    """
    silhouette = 0  # metrics.silhouette_score(X, labels, metric='euclidean')  # 轮廓系数. 很慢
    ch = metrics.calinski_harabasz_score(X, labels)  # calinski_harabaz_score
    if is_show:
        print("无监督: silhouette, ch:", silhouette, ch)
    return silhouette, ch


def eval_2(labels_true, labels_pred, is_show=True):
    """
    有监督的评估
    评价指标 越接近 1 越好
    :param labels_true:
    :param labels_pred:
    :param is_show:  是否显示结果
    :return:
    """
    if labels_true == []:
        info = f"cluster: img_sum:{len(labels_pred)}, id_sum:{len(set(labels_pred))}"
        return [], info
    nmi = 0  # metrics.normalized_mutual_info_score(labels_true, labels_pred)  # 归一化互信息
    ari = metrics.adjusted_rand_score(labels_true, labels_pred)  # 调整兰德指数
    # 纯度,散度, v_measure
    homogeneity, completeness, v_measure_score = metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
    fmi = metrics.fowlkes_mallows_score(labels_true, labels_pred)  # 几何平均数
    avg_pre, avg_rec, fscore = fowlkes_mallows_score(labels_true, labels_pred)  # 调和平均数 *****
    k = 0.5
    fscore_2 = 2. * avg_pre * k * avg_rec / (avg_pre * k + avg_rec)

    s_1 = f"gt: img_sum:{len(labels_true)}, id_sum:{len(set(labels_true))}"
    s_2 = f"cluster: img_sum:{len(labels_pred)}, id_sum:{len(set(labels_pred))}"
    s_3 = "有监督: 纯度, 散度, nmi, v_measure, ari:" + f"{r(homogeneity)}, {r(completeness)}, {r(nmi)}, {r(v_measure_score)}, {r(ari)}"
    s_4 = 'avg_pre, avg_rec, fscore, fmi:' + f"{r(avg_pre)}, {r(avg_rec)}, {r(fscore)}, {r(fmi)}"
    info = f"{s_1}\n{s_2}\n{s_3}\n{s_4}"
    if is_show:
        print(info)
    metric = [avg_pre, avg_rec, fscore, fmi]
    return metric, info


def fowlkes_mallows_score(labels_true, pred_labels, sparse=True):
    from sklearn.metrics.pairwise import pairwise_kernels
    n_samples = len(labels_true)
    c = metrics.cluster.contingency_matrix(labels_true, pred_labels, sparse=sparse)
    # print(c.shape, c.data.shape)
    tk = np.dot(c.data, c.data) - n_samples
    pk = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - n_samples
    qk = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - n_samples
    avg_pre = tk / pk
    avg_rec = tk / qk
    tp = tk
    fp = pk - tk
    fn = qk - tk
    # print('test:', tp, fp, fn)
    fscore = 2. * avg_pre * avg_rec / (avg_pre + avg_rec)
    return avg_pre, avg_rec, fscore


def align_obj(gt_sorted_df, labels_pred_df):
    labels_pred = labels_pred_df["person_id"].values
    obj_ids = labels_pred_df["obj_id"].values

    gt_dict = {}  # 对齐obj_id
    for i in range(len(gt_sorted_df)):
        obj_id = gt_sorted_df.iloc[i, 0]
        gt_person_id = gt_sorted_df.iloc[i, 2]
        gt_dict[obj_id] = gt_person_id

    labels_true = [gt_dict[obj_id_1] for obj_id_1 in obj_ids]

    return labels_true, labels_pred


def eval_3(cluster, labels_true, cluster_1, is_show=True):
    #  自定义的
    if labels_true == []:
        return 0, 0, 0

    dic_cluster = label2dic(cluster)
    dic_gt = label2dic(labels_true)
    dic_cluster_1 = label2dic(cluster_1)

    # for i, l in enumerate(cluster):
    #     add(dic_cluster, l, i)
    # for i, l in enumerate(labels_true):
    #     add(dic_gt, l, i)
    # for i, l in enumerate(cluster_1):
    #     add(dic_cluster_1, l, i)

    cluster = np.array(cluster)
    labels_true = np.array(labels_true)
    count_1 = 0
    count_2 = 0
    count_3 = 0
    c = 0
    c1 = 0
    p_1 = []
    r_1 = []
    for id, arr in list(dic_cluster.items()):  # 纯度
        arr = np.array(arr)
        labels_1 = labels_true[arr]
        maxcount = max(Counter(labels_1).values())  # labels_1中每个id最大的行数
        if maxcount != len(arr):
            count_3 += 1
        len_2 = len(dic_cluster_1[id])
        # if len(arr)*2 < len(dic_cluster_1[id]):
        #     len_2 = len(arr)   # len(dic_cluster_1[id])- len(arr)
        id_g = 0
        for k, v in Counter(labels_1).items():
            if v == maxcount:
                id_g = k
                break
        count_1 += maxcount  # recall
        count_2 += len_2
        p_1.append(maxcount / len(arr))  # precision
        r_1.append(maxcount / len(dic_gt[id_g]))
    r_2 = []
    r_2_2 = 0
    for id, arr in list(dic_gt.items()):  # 散度
        arr = np.array(arr)
        clu_1 = cluster[arr]
        maxcount = max(Counter(clu_1).values())
        j = len(set(clu_1))  # group id, np.unique()同set()
        c1 += j
        lens_c = len(arr)
        c += j * lens_c
        r_2_2 += maxcount
        # if len(arr) in (1, 2):
        #     continue
        r_2.append(maxcount / len(arr))  # 召回
    # print "count:", count_1, count_2, len(cluster_1)
    purity1 = count_1 * 1.0 / len(labels_true)
    purity2 = count_1 * 1.0 / count_2  # **************
    purity3 = 1 - count_3 * 1.0 / len(set(cluster))  # labels
    precision = np.mean(p_1)
    recall_1 = np.mean(r_1)
    recall_2 = np.mean(r_2)
    recall_3 = r_2_2 / len(labels_true)
    divergence1 = len(set(cluster)) * 1.0 / len(set(labels_true))
    divergence2 = c1 * 1.0 / len(set(labels_true))  # **************
    divergence3 = 1.0 * c / len(labels_true)
    if is_show:
        print("cluster:", "img_sum:", len(cluster), "id_sum:", len(dic_cluster))  # , sorted(dic_cluster.keys())
        print("gt:", "img_sum:", len(labels_true), "id_sum:", len(set(labels_true)))  # , sorted(dic_gt.keys())
        # print("purity1:", purity1, "purity2:", purity2, "purity3:", purity3, "divergence1:", divergence1,
        # "divergence2:", divergence2,"divergence3:", divergence3)
        print("purity2:", r(purity2), "p:", r(precision), "r_1:", r(recall_3), "r_2:", r(recall_2), "divergence2:",
              r(divergence2))
    return purity2, divergence1, divergence2


def eval_3_2(cluster, labels_true, cluster_1, is_show=True):
    # 自定义的. macro-F1(宏平均)
    if labels_true == []:
        return [], ''

    dic_gt = util_1.label2dic(labels_true)
    dic_cluster = util_1.label2dic(cluster)
    dic_cluster_1 = util_1.label2dic(cluster_1)

    cluster = np.array(cluster)
    labels_true = np.array(labels_true)
    count_1 = 0
    count_2 = 0
    count_3 = 0
    c = 0
    c1 = 0
    p_1 = []
    for id, arr in list(dic_cluster.items()):  # 纯度 和 precision
        # arr = np.array(arr)
        labels_1 = labels_true[arr]
        maxcount = max(Counter(labels_1).values())  # gt中每个id最大的行数
        if maxcount != len(arr):
            count_3 += 1
        len_2 = len(dic_cluster_1[id])
        # if len(arr)*2 < len(dic_cluster_1[id]):
        #     len_2 = len(arr)   # len(dic_cluster_1[id])- len(arr)
        # id_g = 0
        # for k, v in Counter(labels_1).items():
        #     if v == maxcount:
        #         id_g = k
        #         break
        count_1 += maxcount  # 正确聚类的图片数量
        count_2 += len_2  # 聚出的总图片数量
        p_1.append(maxcount / len(arr))  # precision
        # r_1.append(maxcount/len(dic_gt[id_g]))
    r_1 = []
    r_2_2 = 0
    div_1 = []
    for id, arr in list(dic_gt.items()):  # 散度 和 recall
        arr = np.array(arr)
        clu_1 = cluster[arr]
        id_dict = Counter(clu_1)
        maxcount = max(id_dict.values())
        div_1_1 = len(id_dict)   # 此gt簇,聚散成的簇数量, np.unique()同set()
        c1 += div_1_1
        lens_c = len(arr)
        c += div_1_1 * lens_c  # 加权
        div_1.append(div_1_1)
        r_2_2 += maxcount
        # if len(arr) in (1, 2):
        #     continue
        r_1.append(maxcount / len(arr))  # 召回
    # print "count:", count_1, count_2, len(cluster_1)
    purity1 = count_1 * 1.0 / len(labels_true)
    purity2 = count_1 * 1.0 / count_2  # precision              # **************
    purity3 = 1 - count_3 * 1.0 / len(set(cluster))  # labels
    precision_1 = np.mean(p_1)  # macro-F1(宏平均)
    precision_2 = count_1 * 1.0 / count_2   # micro-F1(微平均)

    recall_1 = np.mean(r_1)
    recall_2 = r_2_2 / len(labels_true)

    divergence1 = len(set(cluster)) * 1.0 / len(set(labels_true))
    divergence2 = c1 * 1.0 / len(set(labels_true))  # **** 按道理divergence1 !=divergence2=divergence_1
    divergence3 = 1.0 * c / len(labels_true)
    divergence_1 = np.mean(div_1)  # macro-F1(宏平均)
    fscore_1 = 2. * precision_1 * recall_1 / (precision_1 + recall_1)
    fscore_2 = 2. * precision_2 * recall_2 / (precision_2 + recall_2)
    purity_1 = precision_1

    # s_1 = f"gt: img_sum:{len(labels_true)}, id_sum:{len(set(labels_true))}"
    # s_2 = f"cluster: img_sum:{len(labels_pred)}, id_sum:{len(set(labels_pred))}"
    s_3 = f"macro-F1: p_1:{precision_1:.5f}, r_1:{recall_1:.5f}, fscore_1:{fscore_1:.5f}"
    s_4 = f"micro-F1: p_2:{precision_2:.5f}, r_2:{recall_2:.5f}, fscore_2:{fscore_2:.5f}"
    s_5 = f"purity1:{purity_1:.5f}, divergence1:{divergence_1:.5f}"
    info = f"{s_3}\n{s_4}\n{s_5}"
    if is_show:
        print(info)
    # return purity2, divergence1, divergence2
    metric = [fscore_1, purity_1, divergence_1]
    return metric, info


def eval(labels_true, labels_pred, p_waste_id=0, is_show=True):
    """
    :param labels_true: gt list
    :param labels_pred:  pred list
    :param is_show:
    :return:
    """
    if len(labels_true) == 0:
        info = f"cluster: img_sum:{len(labels_pred)}, id_sum:{len(set(labels_pred))}"
        return [], info
    # p_waste_id = "0"
    sorted_df = pd.DataFrame({"person_id": labels_pred, "gt_person_id": labels_true})
    sorted_df_1 = sorted_df[sorted_df['person_id'] != p_waste_id]
    labels_pred = sorted_df_1['person_id'].values
    labels_true = sorted_df_1['gt_person_id'].values
    # silhouette, ch = eval_1(feats, glabels, is_show)
    metric_1, info_1 = eval_2(labels_true, labels_pred, is_show)
    metric_2, info_2 = eval_3_2(labels_pred, labels_true, labels_pred, is_show)  # 自定义的
    recall = len(labels_pred)/len(sorted_df)

    info = info_1 + '\n' + info_2 + '\n' + f"all_img:{len(sorted_df)}, recall:{recall:.2%}"
    metric = metric_1 + metric_2
    return metric, info



def eval_dgc(labels_true, labels_pred, is_show=True):
    """
    :param labels_true: gt list
    :param labels_pred:  pred list
    :param is_show:
    :return:
    """
    if len(labels_true) == 0:
        info = f"cluster: img_sum:{len(labels_pred)}, id_sum:{len(set(labels_pred))}"
        return [], info

    metric_2, info_2 = eval_3_2(labels_pred, labels_true, labels_pred, is_show)  # 自定义的
    info = info_2
    fscore_1 = metric_2[0]
    if fscore_1 < 0.98:
        print('结果有问题. 没达到基准')


if __name__ == "__main__":
    eval_dgc(labels_true, labels_pred, is_show=True)
