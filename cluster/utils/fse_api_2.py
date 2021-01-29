import os, time, sys, random, math, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import requests
import json
import base64
import asyncio
import aiohttp
import uvloop

# api params. 应该是传参的
# fse_addr = "http://192.168.101.0:8001"
fse_addr = "http://192.168.3.199:8002"
# fse_addr = "http://192.168.3.192:8001"
# repo params
capacity = 100000000
version = "2.7.3.0"
hashing_num = 1
knn_threshold = 0.5
knn_neighbors = 5

# log config
log_level = 1


def log(level, msg):
    if (level <= log_level):
        print(msg)


async def send_request(url, data):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data) as response:
            return await response.text()


async def index_add(index, feats, idxs):
    await index.add_batch(feats, idxs)


async def index_search(index, query, dists, idxs, topk, threshold, normaliztion):
    await index.search_batch(query, dists, idxs, topk, threshold, normaliztion)


class IndexFseKnn:
    def __init__(self, dim, bgpu=True, itype="int8", repo="repo"):
        self.dim = dim
        self.bgpu = bgpu
        self.itype = itype
        self.repo_id = repo
        self.ntotal = 0
        self.fse_addr = ''
        self.x_api = fse_addr + "/x-api/v1/repositories"

    # feats - features matrix
    # idxs - numeric id of every feature
    def add(self, feats, idxs=None):
        size = feats.shape[0]
        if idxs is None:
            idxs = np.arange(self.ntotal, self.ntotal + size)
        else:
            size = max(size, len(idxs))

        st = time.time()
        batch_size = 100
        num_iter = math.ceil(size / batch_size)
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        loop = asyncio.get_event_loop()
        for i in trange(num_iter):
            batch = batch_size if i != num_iter - 1 else size - i * batch_size
            start = i * batch_size
            loop.run_until_complete(index_add(self, feats[start: start + batch], idxs[start: start + batch]))
        ed = time.time()
        log(1, "add feature completed, elapsed: %fs" % (ed - st))

    async def add_batch(self, feats, idxs):
        size = max(feats.shape[0], len(idxs))

        st = time.time()
        tasks = []
        for i in range(size):
            feat_str = base64.b64encode(feats[i])
            feat_data = {"id": str(idxs[i]),
                         "data": {"value": feat_str.decode(), "type": "feature"},
                         "location_id": "1",
                         "time": 1}
            tasks.append(send_request(self.x_api + "/%s/entities" % self.repo_id, json.dumps(feat_data)))
        await asyncio.gather(*tasks)
        ed = time.time()

        self.ntotal += size
        log(2, "add feature success, ntotal: %d, qps: %d" % (self.ntotal, size / (ed - st)))

    # query - query feature
    # topk - topk
    # threshold - similarity threshold
    # normalization - whether conduct score verification
    def search(self, query, topk, threshold=0, normaliztion="false"):
        print('search()')
        # size = query.shape[0]
        size = 1
        dists = np.zeros((size, topk), "float")
        idxs = np.zeros((size, topk), "int32")

        st = time.time()
        batch_size = 10
        num_iter = math.ceil(size / batch_size)
        # asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        for i in range(num_iter):
            batch = batch_size if i != num_iter - 1 else size - i * batch_size
            start = i * batch_size
            loop.run_until_complete(index_search(self, query[start: start + batch],
                                                 dists[start: start + batch],
                                                 idxs[start: start + batch],
                                                 topk, threshold, normaliztion))
        ed = time.time()
        log(1, "search completed, elapsed: %fs" % (ed - st))
        dists = dists * 2 - 1
        return dists, idxs

    def search_2(self, query, topk, threshold=0, normaliztion="false"):
        print('search_2')
        # feat_str = base64.b64encode(query)   query是base64 string
        query_data = {"include": [{"data": {"value": query, "type": "feature"}}],
                      "include_threshold": threshold,
                      "repositories": [self.repo_id],
                      "max_candidates": topk,
                      "options": {"normalization": normaliztion}}
        # print(query_data)
        res = requests.post(self.x_api + "/search", json.dumps(query_data))
        if res.status_code != 200:
            log(1, "search failed, %s" % json.load(res.content.decode("utf-8")))
            return
        log(1, "search success")

        # convert results
        response = json.loads(res.content.decode("utf-8"))
        print(response)
        results = response["results"]
        dists = np.zeros(topk, "float")
        idxs = np.zeros(topk, "int32")
        for i in range(len(results)):
            dists[i] = float(results[i]["similarity"])
            idxs[i] = int(results[i]["id"])
        dists = dists * 2 - 1
        return dists, idxs

    async def search_batch(self, query, dists, idxs, topk, threshold=0, normaliztion="false"):
        size = 1  # query.shape[0]

        st = time.time()
        tasks = []
        for i in range(size):
            feat_str = base64.b64encode(query[i])
            query_data = {"include": {"data": {"value": feat_str.decode(), "type": "feature"}},
                          "include_threshold": threshold,
                          "repositories": [self.repo_id],
                          "max_candidates": topk,
                          "options": {"normalization": normaliztion}}
            tasks.append(send_request(self.x_api + "/batch_search", json.dumps(query_data)))
        response = await asyncio.gather(*tasks)

        # convert results
        for i in range(len(response)):
            res_data = json.loads(response[i])
            if "results" not in res_data:
                continue
            results = res_data["results"]
            for j in range(len(results)):
                dists[i][j] = float(results[j]["similarity"])
                idxs[i][j] = int(results[j]["id"])
        ed = time.time()
        log(2, "search success, qps: %d" % (size / (ed - st)))

    def create_repo(self, is_knn='false'):
        feat_type = "face" if self.dim == 384 else "motor"
        level = "gpu" if self.bgpu else "ram"
        repo_data = {"id": self.repo_id,
                     "type": feat_type,
                     "index_type": self.itype,
                     "level": level,
                     "capacity": capacity,
                     "default_version": {"name": version},
                     "options": {"hashingNum": str(hashing_num),
                                 "UseFeatureIDMap": "true",
                                 "PreFilter": "false",
                                 "KnnEnabled": str(is_knn),  # true
                                 "KnnThreshold": str(knn_threshold),
                                 "KnnNeighbors": str(knn_neighbors)}}

        res = requests.post(self.x_api, json.dumps(repo_data))
        if res.status_code != 201:
            log(0, "fail to create repo, response: %s" % json.loads(res.content.decode("utf-8")))
            exit(-1)
        log(1, "create repo success, repo_id: \"%s\"" % self.repo_id)

    def delete_repo(self):
        requests.delete(self.x_api + "/%s" % self.repo_id)
        log(1, "delete repo \"%s\"" % self.repo_id)


def main():
    # feats = np.load("longhu_0301/feats.npy")
    dir_1 = '/home/zhangyong/cluster/data/longhu_1'
    dir_1 = '/data2/zhangyong/data/longhu_1/sorted_2'
    feats = np.load(os.path.join(dir_1, "feats.npy"))

    index = IndexFseKnn(384, bgpu=True, repo="repo1")
    index.delete_repo()
    index.create_repo()

    index.add(feats[:10000])
    dist, idx = index.search(feats[:10], 10)
    print(dist)
    print(idx)


def main_2():
    # init fse repo
    dir_1 = '/data/zhangyong/data/pk/pk_11/output_1'
    file_name = os.path.join(dir_1, 'merged_all_out_1_1_1_10.pkl')  # merged_all_out_1_1_1_6
    obj_info_df = pd.read_pickle(file_name)
    feats = obj_info_df['feature'].values  # feature
    feats = np.array([feat for feat in feats])
    print('feats:', feats.shape)
    # fse / milvus
    index = IndexFseKnn(384, bgpu=True, repo="repo_pk_lg2")
    # index = IndexFseKnn(128, bgpu=True, repo="repo_head_should")

    index.delete_repo()
    index.create_repo(is_knn='false')

    index.add(feats)  # 全加入底库
    print("insert into fse over...")

    # return obj_info_df, index


if __name__ == "__main__":
    # main()
    main_2()

# 1305.410488s
