using GraphNeuralNetworks, Graphs, Flux, CUDA, Statistics


function gnn_1()
    all_graphs = GNNGraph[]
    # 做数据. 1000个子图. 
    for _ in 1:1000
        g = GNNGraph(
            random_regular_graph(10, 4);  # 随机的规则无向图. 10个顶点,每个顶点的度是4.
            ndata=(; x=randn(Float32, 16, 10)),  # input node features  10*16, 特征维度是16
            gdata=(; y=randn(Float32)),
        )         # regression target    顶点的label,类别
        push!(all_graphs, g)
    end

    gbatch = Flux.batch(all_graphs)

    device = CUDA.functional() ? Flux.gpu : Flux.cpu
    println(device)
    model = device(GNNChain(
                    GATConv(16 => 64),
                    BatchNorm(64),     # Apply batch normalization on node features (nodes dimension is batch dimension)
                    x -> relu.(x),
                    GATConv(64 => 64, relu),
                    GlobalPool(mean),  # aggregate node-wise features into graph-wise features
                    Dense(64, 1),
                ))

    ps = Flux.params(model)
    opt = ADAM(1.0f-4)

    # train
    gtrain = getgraph(gbatch, 1:800)
    gtest = getgraph(gbatch, 801:(gbatch.num_graphs))
    train_loader = Flux.Data.DataLoader(gtrain; batchsize=32, shuffle=true)
    test_loader = Flux.Data.DataLoader(gtest; batchsize=32, shuffle=false)

    loss(g::GNNGraph) = mean((vec(model(g, g.ndata.x)) - g.gdata.y) .^ 2)

    loss(loader) = mean(loss(device(g)) for g in loader)

    for epoch in 1:400
        for g in train_loader
            g = device(g)
            grad = gradient(() -> loss(g), ps)
            Flux.Optimise.update!(opt, ps, grad)
        end
        @info (; epoch, train_loss=loss(train_loader), test_loss=loss(test_loader))
    end
end


gnn_1()


#=
2021.12.25
base on GraphNeuralNetworks. 
可正常work
没有邻居抽样. 


=#


