# cifar10 + vgg
using Flux, Metalhead
using Flux: onehotbatch, onecold, crossentropy, throttle
using Metalhead: trainimgs
using Images: channelview
using Statistics: mean
using Base.Iterators: partition
using CuArrays   # 使用GPU
# using BSON: @save
# include("nets/vgg.jl")
include("nets/small.jl")
include("nets/resnet.jl")


# Function to convert the RGB image to Float64 Arrays
getarray(X) = Float64.(permutedims(channelview(X), (2, 3, 1)))  # RGB转换为BGR

# Fetching the train and validation data and getting them into proper shape
Metalhead.download(CIFAR10)     # 下载数据集
X = trainimgs(CIFAR10)
imgs = [getarray(X[i].img) for i in 1:50000]  # 32*32
labels = onehotbatch([X[i].ground_truth.class for i in 1:50000],1:10)
println("imgs: ", size(imgs))
# Partition into batches of size 1000, img size 32*32
train = [(cat(imgs[i]..., dims=4), labels[:,i]) for i in partition(1:4900, 10)]  # 4900
println("11: ", size(train)[1], " batch, bs:", size(train[1][1]))

# train = cat(train, train, train, train, train, train, train, train, train, train, dims=1)  # 49*10=490
# train = cat(train, dims=1) 
# train = Iterators.repeat(train, 10)    # ???
# train = Iterators.repeated(train, 10)
# train = collect(train)
# println("train-----: ", typeof(train))
# train = gpu.(train)  # |> gpu   # 把所有的数据都加载到GPU里了，不是bacth模式的
train = train |> gpu
println("train: ", size(train)[1], " batch, bs:", size(train[1][1]))
valset = collect(49001:50000)
valX = cat(imgs[valset]..., dims = 4) |> gpu
valY = labels[:, valset] |> gpu

# Defining the model, loss and accuracy functions
# m = vgg16() |> gpu
m = small_1 |> gpu
# m = resnet50() |> gpu

loss(x, y) = crossentropy(m(x), y)
accuracy(x, y) = mean(onecold(m(x), 1:10) .== onecold(y, 1:10))

# Defining the callback and the optimizer
evalcb = throttle(() -> @show(accuracy(valX, valY)), 10)
opt = ADAM()

# Starting to train models
Flux.train!(loss, params(m), train, opt, cb = evalcb)
# @save "mymodel.bson" m    # save model to file 


# Fetch the test data from Metalhead and get it into proper shape.
# CIFAR-10 does not specify a validation set so valimgs fetch the testdata instead of testimgs
# test = valimgs(CIFAR10)
# testimgs = [getarray(test[i].img) for i in 1:10000]
# testY = onehotbatch([test[i].ground_truth.class for i in 1:10000], 1:10) |> gpu
# testX = cat(testimgs..., dims=4) |> gpu

# # Print the final accuracy
# @show(accuracy(testX, testY))


#=
accuracy(valX, valY) = 0.614
  PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND                                                                                                
20932 test      20   0 4325636 3.290g  93744 R 569.1 21.1  85:37.60 julia -q vgg.jl
accuracy(valX, valY) = 0.588

small, vgg16, resnet50 在CPU, GPU[win10, ubuntu18.04]上都可以跑起来. 2019.2.2
bs=20, train 不能大， 不然会溢出

------------------------
repeat data 
保存模型， 加载预训练模型
间隔 val  
loss曲线画图

参考：
https://github.com/FluxML/Metalhead.jl
https://github.com/FluxML/model-zoo

问题：
应为用户少，稳定性不够，工具链不完善，资料少。
=#
