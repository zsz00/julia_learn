# 聚类相关的评估指标
import os, datetime, time, random
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from collections import Counter


def plot(x, y, style='r', label=''):
    plt.plot(x, y, color=style, label=label, linewidth=1)


def getStyle(cnt):
    color = ["r", "b", "g", "c", "y", "k", "m"]
    marker = ["o", "v", "s", "x", "+", "."]
    linestyle = ["", "--", ":"]
    return color[cnt % 7] + linestyle[cnt % 1]


def show_multi(cols, data, path=""):
    # cols = ['label', 'sizeA', 'sizeB', 'mmax', 'mmin', 'mmean', 'var1', 'var2', 'var3']
    # plt.figure(1, figsize=(10, 6))
    plt.figure(1, figsize=(14, 8))
    if len(cols) == 1:
        aa = 111
    if len(cols) == 2:
        aa = 121
    elif len(cols) <= 4:
        aa = 221
    elif len(cols) <= 9:
        aa = 331
    else:
        aa = 441

    for i in range(len(cols)):
        plt.subplot(aa + i)
        plt.title(cols[i])
        plot(x, y, style='r', label='')
        plt.grid(True)
    if path != "":
        print("hist path:", path)
        plt.savefig(path, dpi=200)
    # plt.show()


def show(title, dir_1):
    # plt.xscale("log")
    # plt.xlim([0.8, 1])  # 1e-9
    # plt.ylim([0.95, 1.01])
    plt.xlabel('th')
    plt.ylabel('fscore')
    plt.title(title)
    plt.legend(loc="best", fontsize=5)  # 图例
    plt.grid(True)
    plt.savefig(os.path.join(dir_1, "a.png"), dpi=200)
    # plt.savefig(datetime.datetime.now().strftime("%Y%m%d%H%M") + ".png", dpi=100)
    plt.show()


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


def label2dic(labels):
    dic = {}
    for i, key in enumerate(labels):
        # add(dic, l, i)
        if key not in dic:
            dic[key] = []
        dic[key].append(i)
    # print("img:", len(labels), "id:", len(dic))
    return dic


def align_gt(true_df, pred_df, id_type_true='gt_person_id', id_type_pred='person_id', waste_id=["", " ", "0", "1", "xxxxxx"]):
    # 对齐 gt和 预测的顺序. 速度更快.
    # print('align_gt()')
    size_gt = len(true_df)
    size_pred = len(pred_df)
    print(f'size_gt:{size_gt}, size_pred:{size_pred}')

    gt_dict = {}  # 生成obj_id:gt_person_id字典
    obj_ids_true = true_df['obj_id'].values
    person_ids_true = true_df[id_type_true].values
    for i in range(size_gt):
        obj_id = obj_ids_true[i]
        gt_person_id = person_ids_true[i]
        if gt_person_id not in waste_id:
            gt_dict[obj_id] = gt_person_id

    obj_ids = []
    labels_pred = []
    labels_true = []
    obj_ids_pred = pred_df['obj_id'].values
    person_ids_pred = pred_df[id_type_pred].values
    for i in range(size_pred):
        obj_id_i = obj_ids_pred[i]
        person_id_i = person_ids_pred[i]
        if person_id_i not in waste_id and obj_id_i in gt_dict:
            obj_ids.append(obj_id_i)
            labels_pred.append(person_id_i)
            labels_true.append(gt_dict[obj_id_i])
    return labels_true, labels_pred, obj_ids


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
    nmi = 0  # metrics.normalized_mutual_info_score(labels_true, labels_pred)  # 归一化互信息. 慢
    ari = metrics.adjusted_rand_score(labels_true, labels_pred)  # 调整兰德指数
    # 纯度/同质性,散度/完整性, v_measure
    homogeneity, completeness, v_measure_score = metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
    fmi = metrics.fowlkes_mallows_score(labels_true, labels_pred)  # 几何平均数
    avg_pre, avg_rec, fscore = fowlkes_mallows_score(labels_true, labels_pred)  # 调和平均数 *****
    k = 0.5
    fscore_2 = 2. * avg_pre * k * avg_rec / (avg_pre * k + avg_rec)
    # ap = metrics.average_precision_score()

    info = ''
    if is_show:
        s_1 = f"gt: img_sum:{len(labels_true)}, id_sum:{len(set(labels_true))}"
        s_2 = f"cluster: img_sum:{len(labels_pred)}, id_sum:{len(set(labels_pred))} "
        s_3 = "有监督: 纯度, 散度, nmi, v_measure, ari:" + f"{r(homogeneity)}, {r(completeness)}, {r(nmi)}, {r(v_measure_score)}, {r(ari)}"
        s_4 = 'avg_pre, avg_rec, pairwise-F1, fmi:' + f"{r(avg_pre)}, {r(avg_rec)}, {r(fscore)}, {r(fmi)}"
        info = f"{s_1}\n{s_2}\n{s_3}\n{s_4}"
        print(info)
    metric = {"avg_pre": avg_pre, "avg_rec": avg_rec, "pairwise-F1": fscore, "fmi": fmi}
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


def eval_3_2_new(imgs_num, labels_pred, labels_true, is_show=True):
    if len(labels_true) <= 0:
        return {}

    label2indexes_pred = label2dic(labels_pred)
    label2indexes_true = label2dic(labels_true)

    labels_pred = np.array(labels_pred)
    labels_true = np.array(labels_true)
    c = 0
    c1 = 0
    max_count_true_per_label_pred = 0
    precision_per_label_pred = []
    for label_pred_current, indexes_pred_current in label2indexes_pred.items():  # 纯度, precision
        indexes_pred_current = np.array(indexes_pred_current)
        labels_true_current = labels_true[indexes_pred_current]
        label2count_true_current = Counter(labels_true_current)
        lable_max_count_true = max(label2count_true_current.values())  # labels_true_current中每个id最大的行数
        precision_per_label_pred.append(lable_max_count_true / len(indexes_pred_current))  # precision
        max_count_true_per_label_pred = lable_max_count_true + max_count_true_per_label_pred  # precision

    r_1 = []
    div_1 = []
    for label_true_current, indexes_true_current in label2indexes_true.items():  # 散度
        indexes_true_current = np.array(indexes_true_current)
        labels_pred_current = labels_pred[indexes_true_current]
        lable_max_count_true = max(Counter(labels_pred_current).values())
        uniq_label_count_pred = len(set(labels_pred_current))
        c1 += uniq_label_count_pred
        lens_c = len(indexes_true_current)
        c += uniq_label_count_pred * lens_c  # 加权
        div_1.append(uniq_label_count_pred)
        r_1.append(lable_max_count_true / len(indexes_true_current))  # 召回

    precision_1 = np.mean(precision_per_label_pred)
    recall_1 = np.mean(r_1)
    print(precision_1, "#####", recall_1)

    divergence = np.mean(div_1)
    fscore_1 = 2. * precision_1 * recall_1 / (precision_1 + recall_1)
    purity = max_count_true_per_label_pred * 1.0 / imgs_num
    # 图片的纯度，类的散度，f1，丢图率
    metric = {
        "purity": purity,
        "divergence": divergence,
        "f1": fscore_1,
        "drop": 1 - len(labels_pred) * 1.0 / imgs_num
    }
    return metric


def eval_3_2(cluster, labels_true, cluster_1, is_show=True):
    # 自定义的
    if labels_true == []:
        return [], ''

    dic_cluster = label2dic(cluster)
    dic_gt = label2dic(labels_true)
    dic_cluster_1 = label2dic(cluster_1)

    cluster = np.array(cluster)
    labels_true = np.array(labels_true)
    count_1 = 0
    count_2 = 0
    count_3 = 0
    c = 0
    c1 = 0
    p_1 = []
    for id, arr in list(dic_cluster.items()):  # 纯度, precision
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
        count_1 += maxcount  # 正确聚类数量
        count_2 += len_2  # 聚出的总数量
        p_1.append(maxcount / len(arr))  # precision
        # r_1.append(maxcount/len(dic_gt[id_g]))
    r_1 = []
    r_2_2 = 0
    div_1 = []
    for id, arr in list(dic_gt.items()):  # 散度
        arr = np.array(arr)
        clu_1 = cluster[arr]
        maxcount = max(Counter(clu_1).values())
        j = len(set(clu_1))  # group id, np.unique()同set()
        c1 += j
        lens_c = len(arr)
        c += j * lens_c  # 加权
        div_1.append(j)
        r_2_2 += maxcount
        # if len(arr) in (1, 2):
        #     continue
        r_1.append(maxcount / len(arr))  # 召回
    # print "count:", count_1, count_2, len(cluster_1)
    purity1 = count_1 * 1.0 / len(labels_true)
    purity2 = count_1 * 1.0 / count_2  # precision              # **************
    purity3 = 1 - count_3 * 1.0 / len(set(cluster))  # labels
    precision_1 = np.mean(p_1)
    precision_2 = count_1 * 1.0 / count_2

    recall_1 = np.mean(r_1)
    recall_2 = r_2_2 / len(labels_true)

    divergence1 = len(set(cluster)) * 1.0 / len(set(labels_true))
    divergence2 = c1 * 1.0 / len(set(labels_true))  # **************
    divergence3 = 1.0 * c / len(labels_true)
    divergence_1 = np.mean(div_1)
    fscore_1 = 2. * precision_1 * recall_1 / (precision_1 + recall_1)
    fscore_2 = 2. * precision_2 * recall_2 / (precision_2 + recall_2)
    purity_1 = precision_2

    info = ''
    if is_show:
        # print("cluster:", "img_sum:", len(cluster), "id_sum:", len(dic_cluster))  # , sorted(dic_cluster.keys())
        # print("gt:", "img_sum:", len(labels_true), "id_sum:", len(set(labels_true)))  # , sorted(dic_gt.keys())
        s_1 = f"macro-F1: p_1:{r(precision_1)}, r_1:{r(recall_1)}, fscore_1:{r(fscore_1)}"
        s_2 = f"micro-F1: p_2:{r(precision_2)}, r_2:{r(recall_2)}, fscore_2:{r(fscore_2)}"
        s_3 = f"purity:{r(purity_1)}, divergence:{r(divergence_1)}"
        info = f"{s_1}\n{s_2}\n{s_3}"
        print(info)
    # return purity2, divergence1, divergence2
    metric = {"macro-F1": r(fscore_1), "micro-F1": r(fscore_2), "purity": precision_2, "divergence": divergence_1}
    return metric, info


def eval(labels_true, labels_pred, is_show=True):
    if len(labels_true) == 0:
        info = f"cluster: img_sum:{len(labels_pred)}, id_sum:{len(set(labels_pred))}"
        return {}, info
    # silhouette, ch = eval_1(feats, glabels, is_show)
    metric_1, info_1 = eval_2(labels_true, labels_pred, is_show)
    metric_2, info_2 = eval_3_2(labels_pred, labels_true, labels_pred, is_show)  # 自定义的
    info = info_1 + '\n' + info_2
    metric_1.update(metric_2)
    return metric_1, info


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
