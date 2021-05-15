using HTTP
using Gumbo  # 解析网页元素
using DataFrames
import Dates
using JLD2
using PrettyTables
using DataStructures


# TODO 用github api做. 
function stars()
    # url_1 = "https://github.com/search?l=r&q=stars%3A%3E1000&s=updated&type=Repositories"
    # langs = ["matlab", "r", "lua", "javascript", "swift", "php", "c", "cpp", "java", "python", "julia", "rust", "typescript", "go"]
    langs = ["julia", "matlab", "r", "swift", "python", "rust", "typescript", "go"]
    # langs = ["julia", "matlab"]
    lang_dict = OrderedDict()
    lang_dict["stars"] = [1, 10, 100 ,1000, 10000]
    for pl in langs
        print(pl, "\t")
        lang_dict[pl] = []
        for num in [1, 10, 100 ,1000, 10000]
            url = "https://github.com/search?l=$(pl)&q=stars%3A%3E$(num)&s=updated&type=Repositories"
            # print(pl, " ", num)

            res = HTTP.get(url)
            body = String(res.body)
            html = Gumbo.parsehtml(body)

            qdata = Base.eachmatch(r".* repository results", body);  # html.root    # 正则匹配
            data = collect(qdata)
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
            push!(lang_dict[pl], star)
            print(" ", star)
        end
        print("\n")
        sleep(10)
    end
    return lang_dict
end

lang_dict = stars()
@save "crawler/all.jld2" lang_dict
@load "crawler/all.jld2" lang_dict
data = DataFrame(lang_dict)

println(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
# println(data)

pretty_table(data, tf=tf_markdown)

