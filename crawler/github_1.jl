using HTTP
using Gumbo  # 解析网页元素
# using Cascadia
# using DataFrames
# using AbstractTrees
using DataFrames

function stars()
    # url_1 = "https://github.com/search?l=r&q=stars%3A%3E1000&s=updated&type=Repositories"
    langs = ["matlab", "r", "lua", "javascript", "swift", "php", "c", "cpp", "java"]  # ["python", "julia", "rust", "typescript", "go"]
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
                aa = star.match   # 匹配项字符串
                star = match(r"\d*,*\d+", aa).match
                star = replace(star, ","=>"")
                star = parse(Int, star)
            end
            push!(lang, star)
            print(" ", star)
        end
        push!(all, lang)
        print("\n")
        sleep(30)
    end
    return all
end

all = stars()
data = DataFrame(all)
print(data)


#=
20190220
python 10  star num: RegexMatch("    91,792 repository results")
python 100  star num: RegexMatch("    16,660 repository results")
python 1000  star num: RegexMatch("    1,907 repository results")
python 10000  star num: RegexMatch("    80 repository results")
julia 10  star num: RegexMatch("    1,185 repository results")
julia 100  star num: RegexMatch("    135 repository results")
julia 1000  star num: RegexMatch("    5 repository results")
julia 10000  star num: 0
rust 10  star num: RegexMatch("    5,655 repository results")
rust 100  star num: RegexMatch("    1,169 repository results")
rust 1000  star num: RegexMatch("    138 repository results")
rust 10000  star num: RegexMatch("    6 repository results")
typescript 10  star num: RegexMatch("    11,353 repository results")
typescript 100  star num: RegexMatch("    2,345 repository results")
typescript 1000  star num: RegexMatch("    367 repository results")
typescript 10000  star num: RegexMatch("    28 repository results")
go 10  star num: RegexMatch("    24,339 repository results")
go 100  star num: RegexMatch("    5,890 repository results")
go 1000  star num: RegexMatch("    998 repository results")
go 10000  star num: RegexMatch("    58 repository results")
matlab 10  star num: RegexMatch("    2,064 repository results")
matlab 100  star num: RegexMatch("    198 repository results")
matlab 1000  star num: RegexMatch("    11 repository results")
matlab 10000  star num: 0
r 10  star num: RegexMatch("    4,895 repository results")
r 100  star num: RegexMatch("    574 repository results")
r 1000  star num: RegexMatch("    25 repository results")
r 10000  star num: 0
lua 10  star num: RegexMatch("    4,002 repository results")
lua 100  star num: RegexMatch("    619 repository results")
lua 1000  star num: RegexMatch("    55 repository results")
lua 10000  star num: RegexMatch("    4 repository results")
javascript 10  star num: RegexMatch("    150,292 repository results")
javascript 100  star num: RegexMatch("    29,737 repository results")
javascript 1000  star num: RegexMatch("    4,387 repository results")
javascript 10000  star num: RegexMatch("    346 repository results")
swift 10  star num: RegexMatch("    13,137 repository results")
swift 100  star num: RegexMatch("    3,436 repository results")
swift 1000  star num: RegexMatch("    580 repository results")
swift 10000  star num: RegexMatch("    27 repository results")
php 10  star num: RegexMatch("    39,750 repository results")
php 100  star num: RegexMatch("    6,300 repository results")
php 1000  star num: RegexMatch("    750 repository results")
php 10000  star num: RegexMatch("    20 repository results")
c 10  star num: RegexMatch("    31,434 repository results")
c 100  star num: RegexMatch("    5,780 repository results")
c 1000  star num: RegexMatch("    665 repository results")
c 10000  star num: RegexMatch("    21 repository results")
cpp 10  star num: RegexMatch("    35,763 repository results")
cpp 100  star num: RegexMatch("    6,401 repository results")
cpp 1000  star num: RegexMatch("    760 repository results")
cpp 10000  star num: RegexMatch("    43 repository results")
java 10  star num: RegexMatch("    65,168 repository results")
java 100  star num: RegexMatch("    13,627 repository results")
java 1000  star num: RegexMatch("    2,009 repository results")
java 10000  star num: RegexMatch("    69 repository results")
=#
