ENV["CUDA_VISIBLE_DEVICES"]=4 
ENV["JULIA_PYTHONCALL_EXE"] = "/home/zhangyong/miniconda3/bin/python"
# using LinearAlgebra, Statistics  #, GLM  # Compat GLM
using CSV, DataFrames, PrettyTables
# using Plots, UnicodePlots
using PythonCall
# using StatsModels
# using MLJLinearModels
using MLJ
using StableRNGs
using Flux, MLJFlux
using Faiss

# LinearRegressor = @load RidgeRegressor pkg=MLJLinearModels  # verbosity=0  GLM  MLJLinearModels
# DecisionTreeRegressor = @load DecisionTreeRegressor pkg=DecisionTree
# LGBMRegressor = @load LGBMRegressor pkg=LightGBM 
NeuralNetworkRegressor = @load NeuralNetworkRegressor


mutable struct MyBuilder2 <: MLJFlux.Builder
    n1 :: Int
    n2 :: Int
    n3 :: Int
    n4 :: Int
end

function MLJFlux.build(nn::MyBuilder2, rng, n_in, n_out)
    #init = Flux.glorot_uniform(rng)
    init = Flux.kaiming_uniform(rng)
    σ =NNlib.leakyrelu  # relu leakyrelu
    return Flux.Chain(Dense(n_in, nn.n1,  init=init),
                                 BatchNorm(nn.n1, relu),
                 Dense(nn.n1, nn.n2, init=init),
                                 BatchNorm(nn.n2, relu),
                                 Dense(nn.n2, nn.n3, init=init),
                                 BatchNorm(nn.n3, relu),
                                 Dense(nn.n3, nn.n4, init=init),
                                 BatchNorm(nn.n4, relu),
                                # Dropout(0.5),
                 Dense(nn.n4, n_out, init=init))
end

function test_1()
    # MLJ方式. train a regression model to calculate distance from two GPS points.
    file_path = "/mnt/zy_data/data/tmp/device_dist.csv"
    # lng1 lat1 lng2 lat2 device_id_i device_id_j dist
    data = DataFrame(CSV.File(file_path, types=Dict(:lng1 => Float64, :lat1 => Float64, :lng2 => Float64, 
    :lat2 => Float64,:dist => Float64,:device_id_i => String,:device_id_j => String), limit=100000, ntasks=2));
    data = DataFrame(CSV.File(file_path, types=Dict(:lng1 => Float32, :lat1 => Float32, :lng2 => Float32, 
    :lat2 => Float32,:dist => Float32,:device_id_i => String,:device_id_j => String), limit=50000, ntasks=2));
    rng = StableRNG(566)   # 设置随机数种子. 得到可复现的随机数
    train, test = partition(eachindex(data.dist), 0.7, shuffle=true, rng=rng);
    x_df = select(data[1:10000,:], [:lng1,:lat1,:lng2,:lat2]);
    y_df = select(data[1:10000,:], [:dist]);

    # DataFrame(models(matching([data.lng1,data.lat1,data.lng2,data.lat2], data.dist)))  # 根据数据,自动选择/搜索 模型

    # model = DecisionTreeRegressor()
    # model = NeuralNetworkRegressor()  # nn
    
    builder=MyBuilder2(32, 64, 128,128)
    MLJFlux.build(builder, rng, 4, 1)
    # MLJFlux.@builder MLJFlux.MLP(;hidden=(50,))
    model = NeuralNetworkRegressor(builder=builder,loss=Flux.Losses.mae) 
    model.batch_size=4000
    model.epochs=40000
    model.acceleration=CUDALibs()

    # iterated_model = IteratedModel(model=EvoTreeClassifier(rng=123, η=0.005),
    #                            resampling=Holdout(), measures=log_loss,
    #                            controls=[Step(5), Patience(2), NumberLimit(100)],
    #                            retrain=true)
    # mach = machine(iterated_model, X, y)   # 此model可用IteratedModel()包装下,提供功过自定义功能

    mach = machine(model, x_df, y_df.dist)   

    fit!(mach, verbosity=2)
    # MLJ.fit!(mach; rows=train)
    fp = fitted_params(mach)
    @show fp

    yhat = predict(mach, x_df_2[:,[:lng1,:lat1,:lng2,:lat2]]);
    # y_d = [yhat[i,1].μ for i in 1:nrow(x_df)]   # GLM
    # y_d = yhat
    # aa = mean(L2Loss(yhat, x_df_2.dist))
    rms(yhat, x_df_2.dist)   
    # 2321.414(LinearRegressor), 10663.6(RidgeRegressor), 796.7(DecisionTreeRegressor) 
    # 4423.78(LGBMRegressor), 344.6(nn)
    x_data = convert(Array, data[!, :x])
    y_data = convert(Array,data[!, :y])

    s = scatterplot(x_data, y_data, title="My Scatterplot", border=:dotted; color=:green, blend=false)
    s = scatterplot!(s, x_data, y_d; color=:red)

    # display(s)
    # UnicodePlots.savefig(s, "mlj_2.txt", color=true);

end

function test_2()
    # MLJ方式. 回归两个feat的cos相似度
    @py np = pyimport("numpy")
    feat_path = "/mnt/zy_data/data/shop_1/shopface.npy"
    feats = np.load(feat_path);
    feats = pyconvert(Array{Float32, 2}, feats);  # py to julia Matrix

    vs_query = feats[1:2000, :];
    vs_gallery = vs_query;
    dists, idxs = local_rank(vs_query, vs_gallery, k=100, metric="IP", gpus="5");

    y_all = Nothing
    for i in range(1, 100; step=10)
        x_df = hcat(vs_query, vs_query[idxs[:,i],:]);   # 384*2*100000
        y_df = dists[:, i];
        if i == 1
            x_all = x_df
            y_all = y_df
        else
            x_all = vcat(x_all, x_df)
            y_all = append!(y_all, y_df)
        end
    end
    x_all = Tables.table(x_all);   # 需要table类型的输入
    rng = StableRNG(566)

    builder=MyBuilder2(1024, 2048, 1024, 512)
    MLJFlux.build(builder, rng, 768, 1)
    model = NeuralNetworkRegressor(builder=builder,loss=Flux.Losses.mae) 
    model.batch_size=8000
    model.epochs=10000
    # model.acceleration=CUDALibs()
    # iterated_model = IteratedModel(model=model)
    mach = machine(model, x_all, y_all)
    fit!(mach, verbosity=2)
    # 奇怪不能用gpu. libcublas.so: symbol cublasLtLegacyGemmSSS version 
    # libcublasLt.so.11 not defined in file libcublasLt.so.11 with link time reference
    MLJ.save("/mnt/zy_data/data/nn_model_feat_2.jlso", mach, compression=:none)

    # [ Info: Loss is 0.005396
    # [ Info: Loss is 0.00167

end


test_2()



#=
2022.2.21
回归 gps距离计算
export JULIA_NUM_THREADS=40
ENV["CUDA_VISIBLE_DEVICES"]=5 
ENV["JULIA_PYTHONCALL_EXE"] = "/home/zhangyong/miniconda3/bin/python"
julia others/alg/mlj_3.jl

NeuralNetworkRegressor nn回归


=#
