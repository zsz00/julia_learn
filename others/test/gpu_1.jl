
using CuArrays   # 使用GPU

xs = cu(rand(5, 5))
ys = cu[1, 2, 3]
xs_cpu = collect(xs)

