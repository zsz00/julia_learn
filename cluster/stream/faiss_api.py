import os
import numpy as np
import faiss


def get_index(feat_dim=384, gpus='0'):
    # feat数据存储在这里面. 数据量巨大时,容易爆显存
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    # print(os.environ['CUDA_VISIBLE_DEVICES'])
    if gpus == '':
        ngpus = 0
    else:
        ngpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if ngpus == 0:
        cpu_index = faiss.IndexFlatL2(feat_dim)
        gpu_index = cpu_index
    elif ngpus == 1:
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0  # int(gpus[0])
        res = faiss.StandardGpuResources()
        gpu_index = faiss.GpuIndexFlatL2(res, feat_dim, flat_config)  # use one gpu.  初始化很慢
    else:
        # print('use all gpu')
        cpu_index = faiss.IndexFlatL2(feat_dim)
        gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)  # use all gpus
    index = gpu_index
    return index


def rank5(index, query, gallery, topk=1000, flag=1):
    """
    增量的add(gallery)
    """
    if flag == 1:
        index.add(gallery)  # <8M
    # print("len(index):", len(index))
    topk = topk  # min(topk, gallery.shape[0])
    dists = np.zeros((query.shape[0], topk), "f")  # 很大.  n*k*4
    idxs = np.zeros((query.shape[0], topk), "int32")  # 很大.  n*k*4
    bs = np.ceil(1e7 * 1.0 / topk)  # 1e6 /  topk=100w/1000  1000, 1w
    bs = int(bs)
    qbatch = int(np.ceil(query.shape[0] * 1.0 / bs))
    # print("#$# qbatch:", qbatch)
    for i in range(qbatch):
        # print(i + 1, "/", qbatch)
        dist, idx = index.search(query[i * bs:(i + 1) * bs], topk)  # L2距离
        dists[i * bs:(i + 1) * bs] = dist
        idxs[i * bs:(i + 1) * bs] = idx
    return idxs, 1 - dists / 2  # 转换为cos相似度


def rank2(query, gallery, topk=1000, metric='cos', ngpus=''):
    # ngpus = 0  # faiss.get_num_gpus()
    # print("number of GPUs:", ngpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = ngpus
    if ngpus == '':
        ngpus = 0
    else:
        ngpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    feat_dim = query.shape[1]

    if ngpus == 0:
        cpu_index = faiss.IndexFlatL2(feat_dim)
        gpu_index = cpu_index
    elif ngpus == 1:
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        res = faiss.StandardGpuResources()
        gpu_index = faiss.GpuIndexFlatL2(res, feat_dim, flat_config)  # use one gpu
    else:
        cpu_index = faiss.IndexFlatL2(feat_dim)
        gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)  # use all gpus
    index = gpu_index

    index.add(gallery)  # <8M
    topk = min(topk, gallery.shape[0])
    dists = np.zeros((query.shape[0], topk), "f")  # 很大.  n*k*4  1m->4G
    idxs = np.zeros((query.shape[0], topk), "int32")  # 很大.  n*k*4
    bs = np.ceil(1e7 * 1.0 / topk)  # 1e6 /  topk=100w/1000  1000, 1w
    bs = int(bs)
    qbatch = int(np.ceil(query.shape[0] * 1.0 / bs))

    # print("#$# qbatch:", qbatch)
    for i in range(qbatch):
        # print(i + 1, "/", qbatch)
        dist, idx = index.search(query[i * bs:(i + 1) * bs], topk)  # L2距离
        dists[i * bs:(i + 1) * bs] = dist
        idxs[i * bs:(i + 1) * bs] = idx

    if metric == 'cos':
        dists = 1 - dists / 2   # 转换为cos相似度
    else:
        dists = dists  # l2 相似度
    return idxs, dists
