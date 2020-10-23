using Dates, CSV, Plots, PlotThemes
using ProgressMeter
# Interact, HTTP, SingularSpectrumAnalysis, JuliaDB,
using OnlineStats

# plotly()
# theme(:bright)
ENV["GKSwstype"] = "100"
ENV["GKS_ENCODING"]="utf8"
gr()


o = Partition(Series(Mean(), Extrema()))
y = randn()

@showprogress for _ in 1:10^3
    fit!(o,  global y += randn())
end

# println(o)
# default(show=true)
plot(o)   
# display()


