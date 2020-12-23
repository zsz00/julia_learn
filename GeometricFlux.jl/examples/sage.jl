using Flux
using GraphSAGE
using LightGraphs
using JLD2
using Printf
using MLBase: roc, f1score
using Random
using StatsBase
import Flux: train!, Tracker
using Flux: onehotbatch, onecold, logitcrossentropy, throttle
using CSV
# using Tracker


function get_data()
    @load "../data/cora_features.jld2" features
    @load "../data/cora_labels.jld2" labels
    @load "../data/cora_graph.jld2" g

    return features, labels, g 
end 

function rand_split(n, ptr)
    """
    Args:
         n: total number of data points
       ptr: percentage of training data

    Returns:
         L: indices for training data points
         U: indices for testing data points
    """

    randid = randperm(n);
    ll = Int64(ceil(ptr*n));

    L = randid[1:ll];
    U = randid[ll+1:end];

    return L, U;
end

function train!(loss, θs::Vector, mini_batches::Vector, opts::Vector; start_opts=zeros(Int,length(opts)), cb=()->(), cb_skip=1)
    """
    extend training method to allow using different optimizers for different parameters
    """

    ps = Tracker.Params(vcat(collect.(θs)...));
    for (i,mini_batch) in enumerate(mini_batches)
        gs = Tracker.gradient(ps) do
            loss(mini_batch...);
        end

        for (θ,opt,start_opt) in zip(θs,opts,start_opts)
            (i > start_opt) && Tracker.update!(opt, θ, gs);
        end

        (i % cb_skip == 0) && cb();
    end
end

function read_cora(dim_reduction=false, dim_embed=8)
    cnt = CSV.read("../data/cora/cora.content", header=0)
    cls = sort(unique(cnt[:,end]))
    cid2num = Dict(id=>num for (num,id) in enumerate(cls))
    pubs = cnt[:,1]

    adj = CSV.read("../data/cora/cora.cites", header=0)
    hh = adj[:,1]
    tt = adj[:,2]
    @assert all(sort(pubs) .== sort(unique(union(hh,tt)))) "unexpected citation graph"
    pid2num = Dict(id=>num for (num,id) in enumerate(pubs))
    g = LightGraphs.Graph(length(pid2num))

    for (h,t) in zip(hh,tt)
        add_edge!(g, pid2num[h], pid2num[t])
    end

    y = map(x->cid2num[x], Array(cnt[:,end]))

    ff = Matrix{Float32}(cnt[:,2:end-1])
    f = [ff[i,:] for i in 1:size(ff,1)]

    if dim_reduction
        U,S,V = svds(hcat(f...); nsv=dim_embed)[1]
        UU = U .* sign.(sum(U,dims=1)[:])'
        f = [UU'*f_ for f_ in f]
        fbar = mean(f)
        f = [f_ - fbar for f_ in f]
    end

    return g, [adjacency_matrix(g)], y, f
end


function train_1()
    # 基于learn.jl, julia v1.5, flux=0.9   可以跑通
    dim_h, dim_r = 16, 8
    n_step = 2000
    ptr = 0.01
    dataset = "cora_false_0"
    p = match(r"cora_([a-z]+)_([0-9]+)", dataset)
    G, adj, y, f = read_cora(parse(Bool, p[1]), parse(Int, p[2]))
    feats = hcat(f...)
    labels = Flux.onehotbatch(y, 1:7)
    n = LightGraphs.nv(G)

    enc = GraphSAGE.graph_encoder(size(feats,1), dim_r, dim_h, repeat(["SAGE_Mean"], 2); σ=relu)
    reg = Chain(Dense(dim_r, size(labels,1)), softmax)
    model(node_list) = reg(hcat(enc(G, node_list, u->feats[:,u])...))

    loss(L) = Flux.crossentropy(model(L), labels[:,L])

    ## Training
    L, VU = rand_split(n, ptr)
    V, U = VU[1:div(length(VU),2)], VU[div(length(VU),2)+1:end]

    n_batch = Int(round(length(L) * 0.50))
    mini_batches = [tuple(sample(L, n_batch, replace=false)) for _ in 1:n_step]

    cb() = @printf("%6.3f,    %6.3f  \n", loss(L), loss(V))

    # Flux.@epochs epochs Flux.train!(loss, ps, train_data, opt, cb=throttle(evalcb, 10))
    train!(loss, [Flux.params(enc, reg)], mini_batches, [ADAM(0.001)]; cb=cb, cb_skip=10)

end


function train_2()
    # 基于 GeometricFlux
    @load "../data/cora_features.jld2" features
    @load "../data/cora_labels.jld2" labels
    @load "../data/cora_graph.jld2" g

    println(g)

    num_nodes = 2708
    num_features = 1433
    heads  = 8
    hidden = 8
    target_catg = 7
    epochs = 10

    dim_h, dim_r = 16, 8

    ## Preprocessing data
    train_X = Matrix{Float32}(features) # |> gpu  # dim: num_features * num_nodes
    train_y = Matrix{Float32}(labels)  # |> gpu  # dim: target_catg * num_nodes
    adj_mat = Matrix{Float32}(adjacency_matrix(g))  # |> gpu
    feats = features
    G = adj_mat
    # n = LightGraphs.nv(G)
    println(size(train_X), size(train_y), size(features), size(labels))   # (1433, 2708)(7, 2708)

    enc = GraphSAGE.graph_encoder(size(feats,1), dim_r, dim_h, repeat(["SAGE_Mean"], 2); σ=relu)
    reg = Chain(Dense(dim_r, size(labels,1)), softmax)
    model(node_list) = reg(hcat(enc(G, node_list, u->feats[:,u])...))

    loss(x, y) = Flux.logitcrossentropy(model(x), y)

    ## Training
    ps = Flux.params(model)
    train_data = [(train_X, train_y)]
    opt = ADAM(0.01)
    evalcb() = @show(loss(train_X, train_y))

    Flux.@epochs epochs Flux.train!(loss, ps, train_data, opt, cb=throttle(evalcb, 10))

end


train_1()


#=
2020.12.17
julia --project=/home/zhangyong/.julia/environments/gcn/Project.toml sage.jl   # 可以跑通


=#
