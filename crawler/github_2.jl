using HTTP, JSON3
using GitHub
using DataFrames


# EventListener settings
myauth = GitHub.authenticate("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")
# mysecret = ENV["MY_SECRET"]

# lang_repo = Dict("julia"=>"JuliaLang/julia")
# results, page_data = GitHub.stargazers(lang_repo["julia"]; auth = myauth)
# println(length(results))

function commen_api(component, method, body="", show=true)
    # api_url = "tcp://192.168.3.199:19530/$component"   # 19530  19121 
    api_url = "https://api.github.com/$component/q=language:assembly&stars>100&sort=stars&order=desc"
    headers = Dict("Authorization"=>"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",
                   "Accept"=>"application/vnd.github.v3+json")  # , "Content-Type" => "application/json"

    response = HTTP.request(method, api_url, headers=headers, body=body)
    status = response.status  #  == 200 ? "OK" : "requests get failed."
    if show println(status) end

    data_text = String(response.body)   # text
    if data_text == "" 
        data_text = "{}"
    end
    data = JSON3.read(data_text)  # string to dict
    if show println("data: ", data) end
    return data
end

function search()
    component = "search/repositories"
    method = "GET"
    body_dict = Dict("q" => "q", "code"=>"language:assembly&sort=stars&order=desc")
    body = JSON3.write(body_dict) 
    commen_api(component, method)

end


search()
