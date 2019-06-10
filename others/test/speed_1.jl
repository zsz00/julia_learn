
function aaa()
    aa = 0
    for i in range(1,1000000,step=1)
        for j in range(1,1000000,step=1)
            aa += i+j
        end
    end
    println(aa)
end

@time aaa()

# 1.54 seconds

