#=
joinquant.com 的 data api 
https://blog.csdn.net/wowotuo/article/details/88072212 
=#
using HTTP, JSON
using DataFrames

function get_token()
    url ="https://dataapi.joinquant.com/apis";
    params_0= Dict("method" => "get_token","mob" =>"13641140370","pwd" => "000000");  # 替换其中密码和账户
    r = HTTP.post( url, body=JSON.json(params_0))
    return String(r.body);
end

function get_security_info(token::String,code::String,date::String)
    url ="https://dataapi.joinquant.com/apis";
    params_1=Dict("method" => "get_security_info","token" => token,"code" => code,"date" =>date);
    data = HTTP.post(url, body=JSON.json(params_1))
    return String(data.body);
end

token =get_token();
println("获得token:{}",token);
println("请等待获取数据......");
data =get_security_info(token,"502050.XSHG","2019-01-15")
println("数据如下：");
println(data)
