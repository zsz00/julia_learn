using HTTP
using Gumbo  # 解析网页元素
# using Cascadia
# using DataFrames
# using AbstractTrees
using DataFrames
import Dates


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
        for num in [10, 100 ,1000, 10000]
            url = "https://github.com/search?l=$(pl)&q=stars%3A%3E$(num)&s=updated&type=Repositories"
            # print(pl, " ", num)

            res = HTTP.get(url)
            body = String(res.body)
            html = Gumbo.parsehtml(body);
            # for elem in PostOrderDFS(html.root) 
            #     println(tag(elem)) 
            # end

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
        sleep(40)
    end
    return all
end

all = stars()
data = DataFrame(all)
# data = DataFrame(all[:, 2:end]', Symbol.(all[:, 1]))
println(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
println(data)


#=
20190309
│ 1   │ matlab │ r    │ lua  │ javascript │ swift │ php   │ c     │ cpp   │ java  │ python │ julia │ rust │ typescript │ go    │
│ 2   │ 2111   │ 4986 │ 4039 │ 151659     │ 13290 │ 40095 │ 31790 │ 36326 │ 65818 │ 93305  │ 1211  │ 5804 │ 11716      │ 24676 │
│ 3   │ 199    │ 584  │ 626  │ 30007      │ 3480  │ 6338  │ 5852  │ 6518  │ 13767 │ 16949  │ 138   │ 1204 │ 2417       │ 5975  │
│ 4   │ 11     │ 25   │ 57   │ 4412       │ 585   │ 762   │ 673   │ 778   │ 2036  │ 1949   │ 5     │ 142  │ 385        │ 1021  │
│ 5   │ 1      │ 1    │ 4    │ 345        │ 27    │ 20    │ 21    │ 43    │ 70    │ 82     │ 1     │ 6    │ 29         │ 60    │


=#
