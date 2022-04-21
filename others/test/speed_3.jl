using BenchmarkTools
using FLoops

function estimate_pi_serial(attempts)
    hits = 0
    for i in 1:attempts
        x = rand()
        y = rand()
        if (x^2 + y^2) <= 1
            hits += 1
        end
    end
    return 4.0 * (hits / attempts)
end

function estimate_pi_threads(attempts)
    hits = zeros(Int,Threads.nthreads())
    Threads.@threads :static for i in 1:attempts
        x = rand()
        y = rand()
        if (x^2 + y^2) <= 1
            hits[Threads.threadid()] += 1
        end
    end
    return 4.0 * (sum(hits) / attempts)
end

function estimate_pi_threads_2(attempts)
    hits = zeros(Int,Threads.nthreads()*10)
    Threads.@threads :static for i in 1:attempts
        x = rand()
        y = rand()
        if (x^2 + y^2) <= 1
            hits[Threads.threadid()*10] += 1
        end
    end
    return 4.0 * (sum(hits) / attempts)
end

function estimate_pi_floop(attempts)
    hits = 0
    @floop for i in 1:attempts
        x = rand()
        y = rand()
        if (x^2 + y^2) <= 1
            @reduce(hits += 1)
        end
    end
    return 4.0 * (hits / attempts)
end

function estimate_pi_floop_2(attempts)
    @floop for i in 1:attempts
        x = rand()
        y = rand()
        if (x^2 + y^2) <= 1
            @reduce(hits = 0 + 1)   # FLoops的奇怪的语法. hits默认值是0, 不能改为hits = 1
            # @reduce(hits = ⊗₁(init₁, x₁))  # +(0,1) = 0 + 1. 这不是常规编程语法,是函数式语法.
        end
    end
    return 4.0 * (hits / attempts)
end

const ATTEMPTS = 500_000_000


println(Threads.nthreads())
# 40

@btime estimate_pi_serial($ATTEMPTS);
# 1.621 s (0 allocations: 0 bytes)

@btime estimate_pi_threads($ATTEMPTS);
# 2.341 s (203 allocations: 19.53 KiB)

@btime estimate_pi_threads_2($ATTEMPTS);
# 96.605 ms (202 allocations: 22.36 KiB)

@btime estimate_pi_floop($ATTEMPTS);
# 2.938 s (234375864 allocations: 3.49 GiB)

@btime estimate_pi_floop_2($ATTEMPTS);
# 89.680 ms (544 allocations: 34.62 KiB)


#=
估计pi的简单的速度基准测试. 速度不符合预期
https://app.slack.com/client/T68168MUP/C67TK21LJ/thread/C67TK21LJ-1648651263.569389
export JULIA_NUM_THREADS=40
julia "/home/zhangyong/codes/julia_learn/others/test/speed_3.jl"

10.9.1.8   2022.3.31, julia 1.7.2
  1.624 s (0 allocations: 0 bytes)
  2.324 s (202 allocations: 19.50 KiB)   # 速度不符合预期
  94.277 ms (202 allocations: 22.36 KiB)
  2.836 s (234375868 allocations: 3.49 GiB)  # 速度不符合预期
  89.753 ms (541 allocations: 34.53 KiB)

10.9.1.8   2022.3.31, julia 1.9.0-DEV.252 (2022-03-26)
40 threads
  1.594 s (0 allocations: 0 bytes)
  2.643 s (242 allocations: 21.38 KiB)
  96.851 ms (241 allocations: 24.20 KiB)
  2.064 s (125002502 allocations: 1.86 GiB)
  92.041 ms (527 allocations: 34.72 KiB)

速度不符合预期, 大部分原因是因为rand()的性能问题?


=#
