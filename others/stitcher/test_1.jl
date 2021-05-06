# import Dates

# println(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS.s"))


function reverse(num_1)
    # -1230 -> -321 
    if num_1 >= 2^31-1 || num_1 <= -2^31 
        return 0
    else
        if num_1 >= 0
            num_1 = string(num_1)
            num_2 = num_1[end:-1:1]
        elseif num_1 < 0
            num_1 = string(num_1)[2:end]
            num_2 = join(["-", num_1[end:-1:1]])
        end
        num_2 = parse(Int32, num_2)
        # println(num_2)
    end
end

# num = -12340
# reverse(num)


@vlplot(
    data=df,
    :bar, # 等价于 mark = :trail 等价于 mark={typ=:trail}
    x={
        "language",
        axis={format="%X"}
    },
    y={
        "stars",
        axis={format="%Y"}
    },
    color=:Origin,
    width=400,
    height=400
)

