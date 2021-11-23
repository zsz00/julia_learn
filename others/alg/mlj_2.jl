using CSV, LinearAlgebra, Statistics, DataFrames  #, GLM  # Compat GLM
using Plots
using RDatasets
using MLJ 


function test_1()
    x = rand(100)
    data = DataFrame(x=x, y = 2x .+ 0.1*rand(100))
    # println(data.x)
    lm = @load lm pkg=GML
    println(lm(@formula(y ~ x), data))

    x_data = convert(Array, data[!, :x])
    y_data = convert(Array,data[!, :y])
    s = scatter(x_data, y_data )

    theta_1 = 0.0547491
    theta_2 = 1.99782
    plot!(s, x_data, x_data * theta_2 .+ theta_1)
    # display(s)

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


test_1()
