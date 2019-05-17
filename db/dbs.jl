using SQLite, MySQL, DataFrames, ExcelFiles
using Gadfly

function sqlite_1(file_path::AbstractString)
    db = SQLite.DB(file_path)
    q = DataFrame(SQLite.query(db, "SELECT * FROM star  limit 10"))
    print(q.star_name) 
end


function mysql_1()
    conn = MySQL.connect("39.106.167.208", "root", "000000"; db="mysql", port=3306, opts = Dict())
    q = DataFrame(MySQL.query(conn, "select User from user limit 2"))
    print(q) 
    MySQL.disconnect(conn)
end


function xlsx()
    data = DataFrame(load(s"C:\zsz\ML\code\julia\julia_learn\db\star.xlsx", "Sheet2"))  # 读取xlsx文件

    row, col = size(data)
    num_1 = Int(col/6)
    dd = []
    for i in 1:Int(num_1)
        data2 = Dict{String,Any}()
        data2["no"] = data[1, (i-1)*6+2]
        data_2 = data[:, (i-1)*6+1:i*6]
        for j in 1:5 
            # data2["month"] = data_2[2,j]
            data2["$j"] = [Int(a) for a in data_2[3:end, j] if ismissing(a)==false]
        end
        
        println(data2)
    end
end

function show()
    data = data[1:end-2,:]
    data = DataFrames.stack(data)    # 行列变换
    p1 = plot(data, color="stars", x="variable", y="value",Geom.bar(position=:dodge, tag=:identity),Theme(bar_spacing=8mm),Guide.xlabel("languages"),Guide.ylabel("num_stars"));   # 簇状柱形图
    p2 = plot(data, color="stars", x="variable", y="value",Geom.bar(position=:stack, tag=:identity),Theme(bar_spacing=8mm),Guide.xlabel("languages"),Guide.ylabel("num_stars"));  # 堆积柱形图
    hstack(p1)  # 画图

end


# file_path = "db/example.db"
# sqlite_1(file_path)
# mysql_1()
xlsx()
