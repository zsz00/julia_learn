using Dates, CSV, JuliaDB, Plots, PlotThemes, ProgressMeter
# Interact, HTTP, SingularSpectrumAnalysis, ProgressMeter
using OnlineStats
theme(:bright)


o = Partition(Series(Mean(), Extrema()))

y = randn()

@showprogress for _ in 1:10^3
    fit!(o,  global y += randn())
end

plot(o)

