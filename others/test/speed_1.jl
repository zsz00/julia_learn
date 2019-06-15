
function aaa()
    aa = []
    Threads.@threads for i in range(1,10000,step=1)   # @showprogress  Threads.@threads
        push!(aa, 1)   
        # for j in range(1,1000000,step=1)
        #     aa += 1
        # end
    end
    println("aa:", size(aa))   # 不是10000,并且变动
end

function aaa2()
    # aa = 0
    aa = Threads.Atomic{Int64}(0)
    Threads.@threads for i in range(1,100000000,step=1)   # @showprogress  Threads.@threads
        # aa += 1
        Threads.atomic_add!(aa, 1)
    end
    println("aa:", aa)   # 对的,并行的
end

function aaa3()
    dir_2 = raw"C:\zsz\ML\code\DL\face_cluster\test\data_1\test_1"
    label_file = joinpath(dir_2, "deepglint_900.txt")
    data = readlines(label_file)[1:100]
    data2 = [parse(Int,a) for a in data]
    data2=hcat(data2)

    # for a in data
    #     data2=hcat((data2, parse(Int,a)))
    # end
    # convert(Array{Int64,2}, data2)
    # data = convert(Array{Int64,1}, data)
    println(size(data), typeof(data2))
end


@time aaa2()

# 1.54 seconds
#=
aa:1000000   0.105776 seconds

aa:500273  0.120048 seconds

=#
