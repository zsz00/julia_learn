
using HTTP
using Gumbo
using Cascadia
using DataFrames

# spaceX API
# https://documenter.getpostman.com/view/2025350/RWaEzAiG
function spaceX()
    url = "https://api.spacexdata.com/v3/capsules/C112"
    rest = HTTP.get(url)
    body = String(rest.body)
    println(body)
    html = parsehtml(body);
    println(html)
end

function stars()
    url = "https://github.com/search?l=r&q=stars%3A%3E1000&s=updated&type=Repositories"
    res = HTTP.get(url)
    body = String(res.body)
    # println(body)
    html = parsehtml(body);
    qdat = eachmatch(sel".`d-flex flex-column flex-md-row flex-justify-between border-bottom pb-3 position-relative`", html.root);
    data = qdat  # scraper(qdat);

    println(data)
end

stars()


