using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle, gpu
using Base.Iterators: repeated, partition
# using CuArrays

# Classify MNIST digits with a convolutional network

imgs = MNIST.images(:train)
labels = onehotbatch(MNIST.labels(:train), 0:9)

# Partition into batches of size 1,000
train = [(cat(float.(imgs[i])..., dims = 4), labels[:,i]) for i in partition(1:60_000, 1000)]
println("a1: ", size(train), typeof(train))
train = repeated(train, 4)
println("a2: ", size(train), typeof(train))
train = gpu.(train)

# Prepare test set (first 1,000 images)
tX = cat(float.(MNIST.images(:test)[1:1000])..., dims = 4) |> gpu  
tY = onehotbatch(MNIST.labels(:test)[1:1000], 0:9) |> gpu  # |> 是左结合， 从左到右依次执行

m = Chain(
  Conv((2,2), 1=>16, relu),  # 卷积层， relu激活函数
  x -> maxpool(x, (2,2)),    # 匿名函数， 最大池化
  Conv((2,2), 16=>8, relu),
  x -> maxpool(x, (2,2)),    
  x -> reshape(x, :, size(x, 4)),
  Dense(288, 10),    # 全连接层， 组合应用特征，完成 分类任务。
  softmax) |> gpu

println("aaa: ", size(train))        # 480
println("bbb: ", size(train[1][1]))  # 6272000
m(train[1][1])   # 一批

loss(x, y) = crossentropy(m(x), y)

accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))

evalcb = throttle(() -> @show(accuracy(tX, tY)), 10)
opt = ADAM(params(m))

Flux.train!(loss, train, opt, cb = evalcb)


# 怎么repeat   ??

