# threads test
using ProgressMeter, BenchmarkTools
using Strs

println("threads: ", Threads.nthreads())

function aaa1()
    data = []
    n = 100
    p = Progress(n)
    l = ReentrantLock()
    Threads.@threads for i in range(1,n,step=1)   # @showprogress  Threads.@threads
        lock(l) do
            push!(data, i) 
        end
        sleep(0.1)  
        next!(p)
        # for j in range(1,1000000,step=1)
        #     aa += 1
        # end
    end
    println(f"\ndata: \(length(data)), \(data[1:10])")   # size不是n,并且变动
end

function aaa2()
    # 原子操作(只支持: +,-,max,min, and, or, xor). 不建议这种方式. 
    aa = Threads.Atomic{Int64}(0)
    bb = 0
    Threads.@threads for x in range(1,10000,step=1)   # Threads.@threads
        bb += x*x
        Threads.atomic_add!(aa, x*x)   # 并行原子加法
    end
    println(f"aa: \(aa[]), \(bb)")   # aa != bb, aa是对的结果
end

function aaa3()
    l = ReentrantLock()
	rst = 0
	Threads.@threads for x in range(1,10000,step=1)
		tmp = x*x # 并行
		lock(l) do
			# 因为加锁变成了串行模型
			rst += tmp
		end
	end
    println(rst)
end


function test_1()
    # 并行写入数据,有数据竞争,无序.
    a = zeros(Int, (1000, 1000))
    Threads.@threads for i = 1:1000
        for j=1:1000
            for k=1:1000
                a[i,j] = i+j*k
            end
        end
    end
    println(f"a: \(a[1:5, 1:10])")
end


@time aaa2()
@time aaa3()
# @time test_1()


#=
export JULIA_NUM_THREADS=4  (Linux/MacOS bash)
set JULIA_NUM_THREADS=4     (win cmd)

1.@threads没有内置的reduce函数支持. 只有简单的静态调度程序.
2.不要使用锁或原子！(除非你知道自己在做什么)
3. Julia在出现数据竞争时不是内存安全的.


# 1.54 seconds
aa:1000000   0.105776 seconds
aa:500273  0.120048 seconds

threads: 1  0.950776 seconds
threads: 2  0.481812 seconds
threads: 4  0.363780 seconds
threads: 8  0.282029 seconds

=#
