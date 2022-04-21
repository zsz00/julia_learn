# An example of semi-supervised node classification

using Flux
using Flux: onecold, onehotbatch
using Flux.Losses: logitcrossentropy
using GraphNeuralNetworks
using MLDatasets: Cora
using Statistics, Random
using CUDA
CUDA.allowscalar(false)

function eval_loss_accuracy(X, y, ids, model, g)
    ŷ = model(g, X)
    l = logitcrossentropy(ŷ[:,ids], y[:,ids])
    acc = mean(onecold(ŷ[:,ids]) .== onecold(y[:,ids]))
    return (loss = round(l, digits=4), acc = round(acc*100, digits=2))
end

# arguments for the `train` function 
Base.@kwdef mutable struct Args
    η = 1f-3             # learning rate
    epochs = 200          # number of epochs
    seed = 17             # set seed > 0 for reproducibility
    usecuda = true      # if true use cuda (if available)
    nhidden = 128        # dimension of hidden features
    infotime = 10 	     # report every `infotime` epochs
end

function train(; kws...)
    args = Args(; kws...)

    args.seed > 0 && Random.seed!(args.seed)
    
    if args.usecuda && CUDA.functional()
        device = gpu
        args.seed > 0 && CUDA.seed!(args.seed)
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    # LOAD DATA
    data = Cora.dataset()
    g = GNNGraph(data.adjacency_list) |> device
    X = data.node_features |> device
    y = onehotbatch(data.node_labels, 1:data.num_classes) |> device
    train_ids = data.train_indices |> device
    val_ids = data.val_indices |> device
    test_ids = data.test_indices |> device
    ytrain = y[:,train_ids]

    nin, nhidden, nout = size(X,1), args.nhidden, data.num_classes 
    
    ## DEFINE MODEL
    model = GNNChain(GATv2Conv(nin => nhidden, relu),   # GCNConv  GATConv GATv2Conv
                     Dropout(0.5),
                     BatchNorm(nhidden),     # Apply batch normalization on node features (nodes dimension is batch dimension)
                     GATv2Conv(nhidden => nhidden, relu), 
                    #  BatchNorm(nhidden),
                     Dropout(0.25),
                     Dense(nhidden, nout))  |> device

    ps = Flux.params(model)
    opt = ADAM(args.η)

    @info g
    
    ## LOGGING FUNCTION
    function report(epoch)
        train = eval_loss_accuracy(X, y, train_ids, model, g)
        test = eval_loss_accuracy(X, y, test_ids, model, g)        
        println("Epoch: $epoch   Train: $(train)   Test: $(test)")
    end
    
    ## TRAINING
    report(0)
    for epoch in 1:args.epochs
        gs = Flux.gradient(ps) do
            ŷ = model(g, X)
            logitcrossentropy(ŷ[:,train_ids], ytrain)
        end

        Flux.Optimise.update!(opt, ps, gs)
        
        epoch % args.infotime == 0 && report(epoch)
    end
end

train()


#=
2022.3.24
base on GraphNeuralNetworks. 官方示例,基本测试.    
julia cluster/gcn/examples/node_classification_cora.jl


数据是一次性进显存的,没有用dataloader. 对于大数据集是不行的
应用的少, 接口还不稳定.  

=#
