using CSV
using DataFrames
using JDF
using Statistics
using Pickle



function test_1()
    file_path = "/mnt/zy_data/data/nj_0601/nodeInfo.csv.0529.head"
    out_path = "/mnt/zy_data/data/nj_0601/nodeInfo.csv.0529.head.jdf"
    data = DataFrame(CSV.File(file_path, header=["nid", "cid", "blur", "hat", "mask"])) 
    # data = identity.(data)
    println(first(data, 5))
    aa = data[data.blur .>0.5, :]
    println(nrow(aa))
    
    gdf1 = groupby(data, :cid)
    df3 = combine(gdf1, :mask => (x -> count(x.==2)) => :mask_num) 
    df4 = combine(groupby(data, :cid), nrow => :num)  # 没有count
    df5 = innerjoin(df3, df4, on=:cid)

    # df4 = df3[df3.mask_num .>=2, :]
    println(first(df5,5))
    println(data[data.cid .=="b9151bdf-81ca-4018-a505-42a0c561cb33", [:nid,:mask]])
    # println(nrow(data), nrow(df3), length(gdf1))

    # 导出 JDF, csv
    # jdffile = JDF.save(out_path, data)
    # println(jdffile)
    out_path = "/mnt/zy_data/data/nj_0601/nodeInfo.csv.0529.head.csv"
    CSV.write(out_path, df5, writeheader=false)
    antijoin(df1, df2, on = :ID)   # df1-df2 差集

end


function test_2()
    file_path = "/mnt/zy_data/data/nj_0601/nodeInfo.csv.0529.head.pkl"
    x = Pickle.load(open(file_path))
    println(dump(x))

end


@time test_2()


#=
功能可以实现, 速度比pandas的慢. 1w行数据. 慢4倍. 

2.3亿

=#


