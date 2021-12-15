# dataframe test
using Dates
using DataFrames, PrettyTables
using PythonCall
using ProgressMeter
using Dagger


function test_1()
    # df遍历速度测试
    println("test_1()")
    @py import pandas as pd
    input_table = pd.read_pickle("/mnt/zy_data/data/gongzufang/merged_all_out_2_1_1_4-2.pkl")
    df = DataFrame(input_table)
    # println(size(df))
    # show(df)
    # aa = combine(filter(t -> nrow(t) == 1, groupby(df, [:obj_id])), :)   # julia 的去重
    # println(size(aa))
    # show(aa[:, [1,3]], truncate=0)
    # return
    t1 = now()
    data_dict = Dict()
    for i in range(1, size(df)[1], step=1)   # @showprogress 
        data = df[i, [:obj_id,:gt_person_id]]   # [:obj_id,:gt_person_id]
        # println(data.obj_id)
        data_dict[data.obj_id] = data.gt_person_id
    end
    println(length(data_dict))  # 1954606
    t2 = now()
    println("time:", (t2-t1).value/1000, "s")
end

function test_2()
    println("test_2()")
    @py import os, sys
    sys.path.insert(0, Py("/home/zhangyong/codes/julia_learn/others/test/"))
    @py import pd_1
    pd_1.temp_2()
end

function test_3()
    println("test_3()")
    @py import pandas as pd
    path_1 = "/mnt/zy_data/data/testset/jianhang_2/img_list_2.csv_2.pkl"
    path_2 = "/mnt/zy_data/data/gongzufang/merged_all_out_2_1_1_4-2.pkl"
    input_table = pd.read_pickle(path_1)
    df = DataFrame(input_table)
    println(size(df))
    d = DTable(df, 200; tabletype=DataFrame)
    println(d)
    println(size(fetch(d)))
    show(fetch(d))  # 此show 结果有点问题,列数对不上

end


# test_1()
# test_2()
@time test_3()




#=
export JULIA_PYTHONCALL_EXE=/home/zhangyong/miniconda3/bin/python

julia --project=/home/zhangyong/codes/julia_learn/cluster/stream/Project.toml \
/home/zhangyong/codes/julia_learn/others/test/test_2.jl


1954606
DataFrames df[i,:]    2.7s
df.iterrows()         100s
df.itertuples(index=True, name='Pandas')   2.54s
zip(df['obj_id'], df['gt_person_id'])      1.06s

=#
