using Pkg
using CSV, LinearAlgebra, Statistics, Compat, GLM, DataFrames
using Plots
using RDatasets
using MLJ 



## The sum of difference
function total_error(theta_1, theta_2, x_data, y_data)
    error = 0.
    for i in 1:m
        error = error + (y_data[i] - (theta_1 + theta_2 * x_data[i]))^2
    end
    return error/m
end

## Gradient descent   梯度下降 线性回归
function gradient_descent(theta_1, theta_2, x_data, y_data, learning_rate, it_times)
    for i in 1:n
        theta_1_grad = 0
        theta_2_grad = 0
        for j in 1:m
            theta_1_grad = theta_1_grad + (1/m) * ((theta_1 + theta_2 * x_data[j]) - y_data[j])
            theta_2_grad = theta_2_grad + (1/m) * ((theta_2 + theta_2 * x_data[j]) - y_data[j]) * x_data[j]
        end
        theta_1 = theta_1 - learning_rate * theta_1_grad
        theta_2 = theta_2 - learning_rate * theta_2_grad
    end
    return theta_1, theta_2
end


function gd_lr()
    # data = CSV.read("admit.csv", normalizenames=true)
    # rename!(data, [:id, :y, :x, :label])
    # println(typeof(data))

    ## Set learning rate
    learning_rate = 0.0001
    ## Intercept
    theta_1 = 0
    ## coef
    theta_2 = 0
    #### Set them to 0 is a normal setup
    ## Iteration times
    n = 50
    ## Data amount
    m = length(x_data)
    ##
    error_change = Float64[]

    theta_1, theta_2 = gradient_descent(theta_1, theta_2, x_data, y_data, learning_rate, n)

    println("Intercept:",theta_1)
    println("Coef:", theta_2)
    s = scatter(x_data, y_data)
    plot!(s, x_data, x_data * theta_2 + repeat([theta_1],99))

end


function test_2()
    # data = CSV.read("admit.csv", normalizenames=true)
    # rename!(data, [:id, :y, :x, :label])
    # println(typeof(data))
    # println(size(data))
    # println(data)
    data = DataFrame(y = rand(100), x = categorical(repeat([1, 2, 3, 4], 25)))
    println(data)
    println(lm(@formula(y ~ x), data))

    #### use matrix calculation
    # Y = data[:,1]
    # constant = ones(99)
    # X = vcat(data[:,2]', constant')
    # coef = inv(X*X')*X*Y

    s = scatter(data[:,2], data[:,1])
    plot!(s)

end


function test_3()
    aa = 1
    
end


test_2()
