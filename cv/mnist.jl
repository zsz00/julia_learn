using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle, gpu
using Base.Iterators: repeated, partition
using CuArrays   # 使用GPU
using BSON: @save, @load

# Classify MNIST digits with a convolutional network
imgs = MNIST.images(:train)
labels = onehotbatch(MNIST.labels(:train), 0:9)

# Partition into batches of size 1000, img size 28*28
train = [(cat(float.(imgs[i])..., dims = 4), labels[:,i]) for i in partition(1:60_000, 1000)]
println("a1: ", size(train), typeof(train))
# train = repeated(train, 4)
# println("a2: ",  typeof(train))   # size(train),
# train = cat(train, train, train, train, train, train, train, train, train, train, dims=1)
# train = cat(train, train, train, train, train, train, train, train, train, train, dims=1) |> gpu
train = gpu.(train)   # 把所有的数据都加载到GPU里了，不是bacth模式的

# Prepare test set (first 1,000 images)
tX = cat(float.(MNIST.images(:test)[1:1000])..., dims = 4) |> gpu  
tY = onehotbatch(MNIST.labels(:test)[1:1000], 0:9) |> gpu  # |> 是左结合， 从左到右依次执行

@load "mymodel.bson" model # 加载模型
m = model
# m = Chain(
#   Conv((3,3), 1=>16, relu),  # 卷积层， relu激活函数
#   x -> maxpool(x, (2,2)),    # 匿名函数， 最大池化
#   Conv((3,3), 16=>32, relu), 
#   Conv((3,3), 32=>64, relu),
#   x -> maxpool(x, (2,2)), 
#   Conv((1,1), 64=>128, relu),
#   Conv((3,3), 128=>256, relu),
#   x -> maxpool(x, (2,2)),   
#   x -> reshape(x, :, size(x, 4)),
#   Dense(256, 10),    # 全连接层， 组合应用特征，完成 分类任务。288
#   softmax) |> gpu

println("aaa: ", size(train))        # 480
println("bbb: ", size(train[1][1]))  # 6272000
m(train[1][1])   # 一批

loss(x, y) = crossentropy(m(x), y)

accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))
evalcb = throttle(() -> @show(accuracy(tX, tY)), 10)
opt = ADAM()

Flux.train!(loss, params(m), train, opt, cb = evalcb)

# @save "mymodel.bson" m    # save model to file,  checkpoint
# # @load "mymodel.bson" model # 加载模型

# weights = Tracker.data.(params(m));
# @save "mymodel_w.bson" weights  # 仅保存模型参数
# @load "mymodel.bson" weights
# Flux.loadparams!(model, weights)   # 加载模型参数


# 怎么repeat   ??
#=
accuracy(tX, tY) = 0.865  60*4= batch_szie
accuracy(tX, tY) = 0.904  60*6
accuracy(tX, tY) = 0.935  60*12=720
accuracy(tX, tY) = 0.952  60*12*12=8640
accuracy(tX, tY) = 0.982
accuracy(tX, tY) = 0.990


Allocations: 93152515 (Pool: 92969827; Big: 182688); GC: 221
Aborted (core dumped)
显存爆了

=#

