# threads

function test()
    a = zeros(10000, 10000)
    Threads.@threads for i = 1:10000
        for j=1:10000
            for k=1:10000
                a[i,j] = i+j*k
            end
        end
    end
end


@time test()

