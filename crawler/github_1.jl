using HTTP
using Gumbo  # 解析网页元素
# using Cascadia
# using AbstractTrees
using DataFrames
import Dates
using JLD2


# TODO 用github api做. 
function stars()
    # url_1 = "https://github.com/search?l=r&q=stars%3A%3E1000&s=updated&type=Repositories"
    # langs = ["matlab", "r", "lua", "javascript", "swift", "php", "c", "cpp", "java", "python", "julia", "rust", "typescript", "go"]
    langs = ["julia", "matlab", "r", "swift", "python", "rust", "typescript", "go"]
    # langs = ["julia", "matlab"]
    # sort!(langs)
    all = []
    for pl in langs
        print(pl, "\t")
        lang = []
        push!(lang, pl)
        for num in [1, 10, 100 ,1000, 10000]
            url = "https://github.com/search?l=$(pl)&q=stars%3A%3E$(num)&s=updated&type=Repositories"
            # print(pl, " ", num)

            res = HTTP.get(url)
            body = String(res.body)
            html = Gumbo.parsehtml(body)

            qdat = Base.eachmatch(r".* repository results", body);  # html.root    # 正则匹配
            # application-main 
            # println(qdat)
            data = collect(qdat)
            # println(data)
            if length(data) == 0
                star = 1
            else
                star = data[1]
                try
                aa = star.match   # 匹配项字符串
                star = match(r"\d*,*\d+", aa).match
                star = replace(star, ","=>"")
                star = parse(Int, star)
                catch
                    print("star: ", star)
                    star = 0
                end
            end
            push!(lang, star)
            print(" ", star)
        end
        push!(all, lang)
        print("\n")
        sleep(10)
    end
    return all
end

all = stars()
@save "all.jld2" all
# @load "all.jld2" all
data = DataFrame(all)

star = DataFrame(data[2:end, :])
name = Symbol.(data[1:1, :])
name = convert(Matrix, name)
name = reshape(name, (8,))
# names!(star, name)  # 要求shape匹配. 4×8 DataFrame,8-element Array{Symbol,1} 
rename!(star, name)
println(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
println(star)

