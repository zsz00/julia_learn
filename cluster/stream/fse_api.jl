# fse api. 有两种方式:1.julia写, 2.调用python的
using HTTP, JSON3


function fse_api()
    api_url = "http://192.168.3.199:38080/vse/version"
    header = Dict("Content-Type"=>"application/json")  
    response = HTTP.request("POST", api_url, headers=header)
    # a ? b : c
    status = response.status == 200 ? "OK" : "requests get failed."   # status  status_code
    println(status)

    # println(String(response.body))
    data_text = String(response.body)   # text
    println(data_text)
    data = JSON3.read(data_text)
    println(data)

end



fse_api()

