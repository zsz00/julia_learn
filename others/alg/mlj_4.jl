# cluster, MLJ

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

NeuralNetworkRegressor = @load NeuralNetworkRegressor





