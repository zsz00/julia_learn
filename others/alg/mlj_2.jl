using CSV, LinearAlgebra, Statistics, DataFrames, GLM  # Compat GLM
using Plots
using RDatasets
using MLJ 

x = rand(100)
data = DataFrame(x=x, y = 2x .+ 0.1*rand(100))
# println(data.x)
println(lm(@formula(y ~ x), data))

x_data = convert(Array, data[!, :x])
y_data = convert(Array,data[!, :y])
s = scatter(x_data, y_data )

theta_1 = 0.0547491
theta_2 = 1.99782
plot!(s, x_data, x_data * theta_2 .+ theta_1)
display(s)
