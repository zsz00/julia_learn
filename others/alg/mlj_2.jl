using CSV, LinearAlgebra, Statistics  #, GLM  # Compat GLM
using DataFrames, PrettyTables
using Plots, UnicodePlots
# using RDatasets
using MLJ
# using GLM  # 广义线性模型
using StableRNGs
using StatsModels
using MLJLinearModels
LinearRegressor = @load LinearRegressor pkg=GLM  # verbosity=0  GLM  MLJLinearModels


function test_1()
    # GLM 原生方式
    x = rand(100)
    data = DataFrame(x=x, y = 2x .+ 0.1*rand(100))

    println(GLM.lm(StatsModels.@formula(y ~ x), data))  # UndefVarError: @formula not defined. 

    x_data = convert(Array, data[!, :x])
    y_data = convert(Array,data[!, :y])
    s = scatter(x_data, y_data )

    theta_1 = 0.0547491
    theta_2 = 1.99782
    plot!(s, x_data, x_data * theta_2 .+ theta_1)
    display(s)

end

function test_1_2()
    # MLJ方式
    x = rand(100)
    y = 2x .+ 1.4*rand(100)
    data = DataFrame(x=x, y=y)
    # DataFrame(models(matching(X,y)))  # 根据数据,自动选择/搜索 模型

    # LinearRegressor = @load LinearRegressor pkg=GLM  # verbosity=0  GLM  MLJLinearModels
    # LinearRegressor = Base.invokelatest(LinearRegressor)
    # lm = LinearRegressor(fit_intercept=true, solver=nothing)
    lm = LinearRegressor()

    rng = StableRNG(566)   # 设置随机数种子. 得到可复现的随机数
    train, test = partition(eachindex(y), 0.7, shuffle=true, rng=rng);
    x_df = select(data, :x)
    mach = machine(lm, x_df, data.y)
    # fit!(mach)
    MLJ.fit!(mach; rows=train)
    fp = fitted_params(mach)
    @show fp

    yhat = predict(mach, x_df)
    y_d = [yhat[i,1].μ for i in 1:nrow(x_df)]   # GLM
    # y_d = yhat
    # aa = mean(L2Loss(yhat, y[test]))

    x_data = convert(Array, data[!, :x])
    y_data = convert(Array,data[!, :y])

    s = scatterplot(x_data, y_data, title="My Scatterplot", border=:dotted; color=:green, blend=false)
    s = scatterplot!(s, x_data, y_d; color=:red)
    
    display(s)
    UnicodePlots.savefig(s, "mlj_2.txt", color=true);

end


function test_2()

    # iris = load_iris();
    # selectrows(iris, 1:3)  |> pretty
    # iris = DataFrames.DataFrame(iris);
    # y, X = unpack(iris, ==(:target), colname -> true; rng=123);
    # first(X, 3) |> pretty

    # @show models(matching(X,y))

    DecTree = @iload DecisionTreeClassifier pkg=DecisionTree
    tree = DecTree()
    # evaluate(tree, X, y, resampling=CV(shuffle=true),
    #                     measures=[log_loss, accuracy],
    #                     verbosity=0)

end


test_1_2()



#=
GLM:
1.999520578070724, 0.700793714513881
MLJLinearModels:
coefs = [:x => 2.00010848034552], intercept = 0.7003164970434766

=#
