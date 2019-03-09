
using HTTP
using Gumbo


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




