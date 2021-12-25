using GraphNeuralNetworks, Graphs, Flux, CUDA, Statistics


function gnn_1()
    all_graphs = GNNGraph[];

    for _ in 1:1000
        g = GNNGraph(random_regular_graph(10, 4),  
                    ndata=(; x = randn(Float32, 16,10)),  # input node features
                    gdata=(; y = randn(Float32)))         # regression target   
        push!(all_graphs, g)
    end

    gbatch = Flux.batch(all_graphs)

    device = CUDA.functional() ? Flux.gpu : Flux.cpu;
    println(device)
    model = GNNChain(GCNConv(16 => 64),
                            BatchNorm(64),     # Apply batch normalization on node features (nodes dimension is batch dimension)
                            x -> relu.(x),     
                            GCNConv(64 => 64, relu),
                            GlobalPool(mean),  # aggregate node-wise features into graph-wise features
                            Dense(64, 1)) |> device;

    ps = Flux.params(model);
    opt = ADAM(1f-4);

    # train
    gtrain = getgraph(gbatch, 1:800)
    gtest = getgraph(gbatch, 801:gbatch.num_graphs)
    train_loader = Flux.Data.DataLoader(gtrain, batchsize=32, shuffle=true)
    test_loader = Flux.Data.DataLoader(gtest, batchsize=32, shuffle=false)

    loss(g::GNNGraph) = mean((vec(model(g, g.ndata.x)) - g.gdata.y).^2)

    loss(loader) = mean(loss(g |> device) for g in loader)

    for epoch in 1:100
        for g in train_loader
            g = g |> device
            grad = gradient(() -> loss(g), ps)
            Flux.Optimise.update!(opt, ps, grad)
        end
        @info (; epoch, train_loss=loss(train_loader), test_loss=loss(test_loader))
    end
end


gnn_1()


#=
2021.12.25



=#


