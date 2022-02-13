using Random
using ThreadsX

Random.seed!(123)
a = randn(10^9)
# println(a)
println("sum:")
@time b=sum(a)
println(b)
@time b=ThreadsX.sum(a)
println(b)


#=
export JULIA_NUM_THREADS=4
在多线程时, 有速度提升. 


=#

