# threads

print("threads: ", Threads.nthreads())


function test()
    a = zeros(1000, 1000)
    Threads.@threads for i = 1:1000
        for j=1:1000
            for k=1:1000
                a[i,j] = i+j*k
            end
        end
    end
end


@time test()


"""
export JULIA_NUM_THREADS=4  (Linux/MacOS bash)
set JULIA_NUM_THREADS=4     (win cmd)

threads: 1  0.950776 seconds
threads: 2  0.481812 seconds
threads: 4  0.363780 seconds
threads: 8  0.282029 seconds


"""