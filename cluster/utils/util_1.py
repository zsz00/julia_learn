import os, datetime, time, random
import numpy as np
import cv2
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn import metrics
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import base64, json, struct
from io import BytesIO
import urllib.request, urllib.parse, urllib.error
from collections import Counter


def get_img(path, type='cv2'):
    if path[:4] == 'http':
        image = url2img(path, type)
    else:
        image = fn2img(path, type)
    return image


def url2img(url, type='cv2'):
    """
    从网络读取图像数据并转换成图片格式.
    支持 PIL和opencv格式.
    :return:
    """
    request = urllib.request.Request(url)
    img = urllib.request.urlopen(request).read()
    if type == 'pil':
        image = Image.open(BytesIO(img))  # PIL
    else:
        image = np.asarray(bytearray(img), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # cv2.imshow('URL2Image', image)
    # cv2.waitKey()
    return image


def fn2img(filename, type='cv2'):
    try:
        if type == 'pil':
            img = Image.open(filename)
        else:
            img = cv2.imread(filename)
    except:
        print('open img errer:', filename)
        img = 0
    return img


def img2fn(img, filename, type=''):
    path = filename  # 此路径不能是url,只能是本地路径.
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, 'wb') as f:
        try:
            f.write(img)
        except:
            print('error...')
            pass
    # cv2.imwrite(path, img)


def draw_face(img, face, info=''):
    # x0, y0, x1, y1
    img = cv2.rectangle(img, (int(face[0]), int(face[1])), (int(face[2]), int(face[3])), (0, 255, 0),
                        2)  # 画框 (0, 255, 0)
    # img = cv2.circle(img, (int(face[0]), int(face[1])), 1, color)  # color  (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX  # 字体
    cv2.putText(img, '%s' % info, (int(face[0]), int(face[1]) - 3), font, 0.6, (0, 0, 255), 1, cv2.LINE_AA)  # 写字
    # cv2.imshow('image', img)  # 用cv2 显示 pil打开的img需要转换rgb.
    # cv2.waitKey(0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def merge(pics, info, img_sfx, output, is_url=False):  # plot.  merge pics
    wid = 10
    height = 5
    show_imsize = 200
    merge_img = Image.new('RGB', (wid * show_imsize, height * show_imsize), 0xffffff)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf', 18)
    draw = ImageDraw.Draw(merge_img)

    for i, pic in enumerate(pics):
        # print("pic:", pic)
        if is_url == False:
            for dir_1 in img_sfx:
                # print('img:', pic)
                # fn = os.path.join(dir_1, str(pic, encoding='utf-8').rstrip())  #  , encoding='utf-8' str(pic, encoding='utf-8')
                fn = os.path.join(dir_1, pic.rstrip())
                if os.path.exists(fn):
                    break
            img1 = Image.open(fn).resize((show_imsize, show_imsize), Image.BICUBIC)
        else:
            url_1 = img_sfx[0]  # "http://192.168.3.112:8000"
            fn = os.path.join(url_1, pic)  # os.path.join(url_1, str(pic, encoding='utf-8').rstrip())
            # print("fn:", fn)
            img = get_img(fn)  # url
            img1 = Image.open(BytesIO(img)).resize((show_imsize, show_imsize), Image.BICUBIC)  # 每个人脸小图片是 100*100

        pos = ((i % wid) * show_imsize, (i // wid) * show_imsize)  # py2, py3
        merge_img.paste(img1, pos)

    draw.text((0, 0), str(info), fill="red", font=font)
    dir_1 = os.path.dirname(output)
    if not os.path.exists(dir_1):
        os.makedirs(dir_1)
    merge_img.save(output, quality=100)
    # merge_img.show(output.split('/')[-1])


def merge2(pics, info, tar, output):  # plot.  merge pics
    wid = 10
    height = 5
    show_imsize = 200
    merge_img = Image.new('RGB', (wid * show_imsize, height * show_imsize), 0xffffff)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf', 18)
    draw = ImageDraw.Draw(merge_img)
    # img_sfx = "/data/yongzhang/cluster/test_2/"
    for i, pic in enumerate(pics):
        # for dir_1 in img_sfx:
        #     fn = os.path.join(dir_1, str(pic).rstrip())
        #     if os.path.exists(fn):
        #         break

        data = get_file(tar, pic)
        img1 = Image.open(data).resize((show_imsize, show_imsize), Image.BICUBIC)  # 每个人脸小图片是 100*100
        pos = ((i % wid) * show_imsize, (i // wid) * show_imsize)
        merge_img.paste(img1, pos)

    draw.text((0, 0), str(info), fill="red", font=font)
    dir_1 = os.path.dirname(output)
    if not os.path.exists(dir_1):
        os.makedirs(dir_1)
    merge_img.save(output, quality=100)


def get_file(tar, filename):
    # get file from tar
    info = tar.getmember(filename)
    data = tar.extractfile(info)
    data1 = data.read(info.size)
    # im = Image.open(BytesIO(data1))
    # im = np.asarray(data1)
    # im = im.transpose([1, 0, 2])
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # cv2.imshow('img', im)
    # cv2.waitKey(0)
    return BytesIO(data1)


def shownpy(file_name):
    data = np.load(file_name, allow_pickle=True)
    print(file_name)
    print("shape:%s, type:%s" % (data.shape, data.dtype))


def hist(score, path=''):
    num_bins = 100
    plt.hist(score, num_bins)
    plt.grid(True)
    # plt.xlim([0, 1])
    if path != "":
        # print("hist path:", path)
        plt.savefig(path, dpi=200)
    # plt.show()


def hist_2(score, path=''):
    figure = plt.figure(1, figsize=(6, 3))
    num_bins = 100
    plt.hist(score, num_bins)
    plt.grid(True)
    # plt.xlim([0, 1])
    if path != "":
        # print("hist path:", path)
        plt.savefig(path, dpi=200)
    # plt.show()
    return figure


def hist_multi(cols, data, path=""):
    # cols = ['label', 'sizeA', 'sizeB', 'mmax', 'mmin', 'mmean', 'var1', 'var2', 'var3']
    # plt.figure(1, figsize=(10, 6))
    figure = plt.figure(1, figsize=(14, 8))
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
        # plt.hist(data[:, i], 100)
        plt.title(cols[i])
        # print("min:", min(data[:, i]))
        bb_min = min(data[:, i])
        bb_max = max(data[:, i])
        if bb_max > 1000:
            plt.hist(data[:, i], 1000)
            plt.xlim(left=bb_min + 1, right=bb_max - 1)  # 不太起作用
        else:
            plt.hist(data[:, i], 100)
        plt.grid(True)
    if path != "":
        print("hist path:", path)
        plt.savefig(path, dpi=200)
    # plt.show()
    return figure


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
    info = ''
    if is_show:
        s_1 = f"gt: img_sum:{len(labels_true)}, id_sum:{len(set(labels_true))}"
        s_2 = f"cluster: img_sum:{len(labels_pred)}, id_sum:{len(set(labels_pred))} "
        s_3 = "有监督: 纯度, 散度, nmi, v_measure, ari:" + f"{r(homogeneity)}, {r(completeness)}, {r(nmi)}, {r(v_measure_score)}, {r(ari)}"
        s_4 = 'avg_pre, avg_rec, fscore, fmi:' + f"{r(avg_pre)}, {r(avg_rec)}, {r(fscore)}, {r(fmi)}"
        # print("gt:", "img_sum:", len(labels_true), "id_sum:", len(set(labels_true)))  # , sorted(dic_gt.keys())
        # print("cluster:", "img_sum:", len(labels_pred), "id_sum:", color(len(set(labels_pred))))  # , sorted(dic_cluster.keys())
        # print("有监督: 纯度, 散度, nmi, v_measure, ari:", r(homogeneity), r(completeness), r(nmi), r(v_measure_score), r(ari))
        # print('avg_pre, avg_rec, fscore, fmi:', r(avg_pre), r(avg_rec), color(r(fscore)), r(fmi))
        info = f"{s_1}\n{s_2}\n{s_3}\n{s_4}"
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
    print('test:', tp, fp, fn)
    fscore = 2. * avg_pre * avg_rec / (avg_pre + avg_rec)
    return avg_pre, avg_rec, fscore


# purity2, divergence1, divergence2 = eval_purity_div(cluster, labels, cluster_1)
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
    purity_1 = precision_1

    info = ''
    if is_show:
        # print("cluster:", "img_sum:", len(cluster), "id_sum:", len(dic_cluster))  # , sorted(dic_cluster.keys())
        # print("gt:", "img_sum:", len(labels_true), "id_sum:", len(set(labels_true)))  # , sorted(dic_gt.keys())
        s_1 = f"macro-F1: p_1:{r(precision_1)}, r_1:{r(recall_1)}, fscore_1:{r(fscore_1)}"
        s_2 = f"micro-F1: p_2:{r(precision_2)}, r_2:{r(recall_2)}, fscore_2:{r(fscore_2)}"
        s_3 = f"purity1:{r(purity_1)}, divergence1:{r(divergence_1)}"
        info = f"{s_1}\n{s_2}\n{s_3}"
        print(info)
    # return purity2, divergence1, divergence2
    metric = [precision_1, recall_1, precision_1, divergence_1]
    return metric, info


def eval(labels_true, labels_pred, is_show=True):
    if len(labels_true) == 0:
        info = f"cluster: img_sum:{len(labels_pred)}, id_sum:{len(set(labels_pred))}"
        return [], info
    # silhouette, ch = eval_1(feats, glabels, is_show)
    metric_1, info_1 = eval_2(labels_true, labels_pred, is_show)
    metric_2, info_2 = eval_3_2(labels_pred, labels_true, labels_pred, is_show)  # 自定义的
    info = info_1 + '\n' + info_2
    metric = metric_1 + metric_2
    return metric, info


def l2norm(vec):
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec


def is_l2norm(features, size):
    rand_i = random.choice(range(size))
    norm_ = np.dot(features[rand_i, :], features[rand_i, :])
    return abs(norm_ - 1) < 1e-6


def add(dic, k, v):
    if k not in dic:
        dic[k] = []
    dic[k].append(v)


def change_label(labels):
    """
    把原始label格式化为int型的id标签. 并且会把id 标准化为从0到n的int.
    """
    dic = {}
    labels = np.array(labels)
    for i, l in enumerate(labels):
        add(dic, l, i)
    # print("cluster:", "img:", len(labels), "id:", len(dic))
    count = 0
    for k, arr in list(dic.items()):
        labels[arr] = count
        count += 1
    labels = labels.astype('int32')
    return labels


def label2dic(labels):
    dic = {}
    for i, key in enumerate(labels):
        # add(dic, l, i)
        if key not in dic:
            dic[key] = []
        dic[key].append(i)
    # print("img:", len(labels), "id:", len(dic))
    return dic


def clusters2labels(clusters):
    # 慢
    idx2lb = {}
    for lb, cluster in enumerate(clusters):
        try:
            for v in cluster:
                idx2lb[v] = lb
        except:
            print(cluster)
    return idx2lb


def clusters2labels2(clusters, size):
    # id 都是 数字
    idx2lb = np.zeros((size,), 'int32')  # size
    for lb, cluster in clusters.items():
        idx2lb[cluster] = lb
    return idx2lb


def read_bin(filename, dtype=np.float32, verbose=True):
    """
    解析公司的bin文件. 结构不通用
    :param filename:
    :param dtype:
    :param verbose:
    :return:
    """
    t1 = time.time()
    file = open(filename, 'rb')
    rows, feat_dim = struct.unpack('qq', file.read(2 * 8))
    feats = []
    for i in range(rows):
        file.read(16)
        prob = np.fromfile(file, dtype=dtype, count=feat_dim)
        feats.append(prob)
    feats = np.array(feats)
    file.close()
    is_norm = is_l2norm(feats, 10)
    if is_norm == False:
        feats = l2norm(feats)
    if verbose:
        print('[{}] shape:{}, used time:{}s'.format(filename, feats.shape, int(time.time() - t1)))
    return feats


def read_txt(filename, dtype='int32'):
    data = open(filename, 'r').readlines()
    data = [aa.strip().split(' ') for aa in data]
    data = np.array(data).astype(dtype)
    return data


def decode_from_buffer(s, feat_dim=384):
    cc = base64.b64decode(s)
    if len(cc) != 4 * feat_dim:
        print("B64 DECODE ERROR")
        raise ValueError("the length of the decoded string is not correct.")
    ret = np.frombuffer(cc, dtype=np.float32)
    return ret


def byte_to_base64(data):
    s_data = base64.b64encode(data)
    s_data = s_data.decode()
    return s_data


def get_var(feats):
    # 方差和.(0,1). 类内多个图片的距离,类内纯度
    return np.sum(np.var(feats, axis=0))
