using CUDA


ngpus = length(CUDA.devices())

println(ngpus)

println(CUDA.device!(2))

println(CUDA.device())


