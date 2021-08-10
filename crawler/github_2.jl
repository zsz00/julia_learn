using HTTP, JSON3
using GitHub
using DataFrames


# EventListener settings
myauth = GitHub.authenticate("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")
# mysecret = ENV["MY_SECRET"]

lang_repo = Dict("julia"=>"JuliaLang/julia")
repo = GitHub.Repo(lang_repo["julia"])
results = GitHub.repo(repo; auth = myauth)

# stat = Status(results.id)
# println(stat)
# results = GitHub.stats(repo, "commits", auth=myauth)
# results, page_data = GitHub.commits(repo, auth=myauth)

println(results)


function commen_api(component, method, body="", show=true)
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


# search()
