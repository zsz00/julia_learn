using DataFrames, ExcelFiles
data = DataFrame(load(s"C:\zsz\ML\code\julia\db/star.xlsx", "Sheet2"))  # 读取xlsx文件

row, col = size(data)
num_1 = Int(col/6)
dd = []
for i in 1:Int(num_1)
    data2 = Dict{String,Any}()
    data2["no"] = data[1, (i-1)*6+2]
    data_2 = data[:, (i-1)*6+1:i*6]
    for j in 1:5 
        # data2["month"] = data_2[2,j]
        data2["$j"] = [Int(a) for a in data_2[3:end, j] if ismissing(a)==false ]
    end
    
    println(data2)
end


