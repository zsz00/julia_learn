using GeometricFlux
using Flux
using Flux: throttle
using Flux.Losses: logitbinarycrossentropy
using Flux: @epochs
using JLD2
using Statistics: mean
using SparseArrays
using LightGraphs.SimpleGraphs
using LightGraphs: adjacency_matrix
using CUDA


function gae_1()
    # gae, gpu  demo
    @load "data/cora_features.jld2" features
    @load "data/cora_graph.jld2" g

    num_nodes = 2708
    num_features = 1433
    hidden1 = 32
    hidden2 = 16
    target_catg = 7
    epochs = 200

    ## Preprocessing data
    adj_mat = Matrix{Float32}(adjacency_matrix(g)) |> gpu
    train_X = Float32.(features) |> gpu  # dim: num_features * num_nodes
    train_y = adj_mat  # dim: num_nodes * num_nodes

    ## Model
    encoder = Chain(GCNConv(adj_mat, num_features=>hidden1, relu),
                    GCNConv(adj_mat, hidden1=>hidden2))
    model = Chain(GAE(encoder, σ)) |> gpu

    ## Loss
    loss(x, y) = logitbinarycrossentropy(model(x), y)

    ## Training
    ps = Flux.params(model)
    train_data = [(train_X, train_y)]
    opt = ADAM(0.01)
    evalcb() = @show(loss(train_X, train_y))

    @epochs epochs Flux.train!(loss, ps, train_data, opt, cb=throttle(evalcb, 10))

end


function gat_1()
    # gat, 节点分类 demo.  
    @load "data/cora_features.jld2" features
    @load "data/cora_labels.jld2" labels
    @load "data/cora_graph.jld2" g

    num_nodes = 2708
    num_features = 1433
    heads  = 4
    hidden = 8
    target_catg = 7
    epochs = 10

    ## Preprocessing data. 把所有数据都先放到gpu里. 
    train_X = Matrix{Float32}(features) |> gpu  # dim: num_features * num_nodes
    train_y = Matrix{Float32}(labels) |> gpu  # dim: target_catg * num_nodes
    adj_mat = Matrix{Float32}(adjacency_matrix(g)) |> gpu  # 邻接矩阵

    ## Model
    model = Chain(GATConv(g, num_features=>hidden, heads=heads),
                Dropout(0.6),
                GATConv(g, hidden*heads=>target_catg, heads=heads, concat=false)
                ) |> gpu
    # test model
    # @show model(train_X)

    ## Loss
    loss(x, y) = logitcrossentropy(model(x), y)
    accuracy(x, y) = mean(onecold(cpu(model(x))) .== onecold(cpu(y)))

    # test loss
    # @show loss(train_X, train_y)

    # test gradient
    # @show gradient(X -> loss(X, train_y), train_X)

    ## Training
    ps = Flux.params(model)
    train_data = DataLoader(train_X, train_y, batchsize=num_nodes)
    opt = ADAM(0.01)
    evalcb() = @show(accuracy(train_X, train_y))

    @epochs epochs Flux.train!(loss, ps, train_data, opt, cb=throttle(evalcb, 10))
end


gat_1()




#=
2021.1
2021.8.10

=# 




