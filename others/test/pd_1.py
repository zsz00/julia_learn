import time
import pandas as pd
from tqdm import tqdm


def temp_2():
    df = pd.read_pickle("/mnt/zy_data/data/gongzufang/merged_all_out_2_1_1_4-2.pkl")
    t1 = time.time()
    data_dict = dict()
    # for index, row in df.iterrows():   # tqdm()
    #     # print(row["obj_id"])
    #     data_dict[row["obj_id"]] = row["gt_person_id"]

    # 花费在类型检查
    # for row in df.itertuples(index=True, name='Pandas'):
    #     # print(getattr(row, "obj_id"), getattr(row, "gt_person_id"))
    #     data_dict[getattr(row, "obj_id")] = getattr(row, "gt_person_id")

    # # 花费在构建namedtuple
    for tup in zip(df['obj_id'], df['gt_person_id']):
        # print(tup,type(tup[1:]))
        data_dict[tup[0]] = tup[1]
    # # 原生tuple的性能
    # # 速度: zip() > itertuples() > iterrows() 
        
    print(len(data_dict))
    print("time:", time.time()-t1, "s")
