using HTTP
using Gumbo
using Cascadia
using DataFrames
using AbstractTrees


function stars()
    # url_1 = "https://github.com/search?l=r&q=stars%3A%3E1000&s=updated&type=Repositories"
    for pl in ["r", "python", "julia", "rust", "typescript", "go"]
        for num in [10, 100 ,1000, 10000]
            url = "https://github.com/search?l=$(pl)&q=stars%3A%3E$(num)&s=updated&type=Repositories"
            print(pl, " ", num)

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
                star = 0
            else
                star = data[1]
            end
            println("  star num: ", star)
        end
        sleep(10)
    end
end

stars()
