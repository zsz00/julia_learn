# plot_embedding.  embedding通过 t-SNE 或 PCA 方法降维到3维(或2维),用这3维数据做为坐标的(x, y, z),然后画散点图.

using PyCall, Makie, Plots, TSne

np = pyimport("numpy")
sklearn = pyimport("sklearn.manifold")
TSNE = sklearn."TSNE"
# from sklearn.manifold import TSNE

features = np.load(raw"D:\download\features\512.fea.npy")
labels = np.load(raw"D:\download\features\512.labels.npy")
println("load over...")
feat_tsne = TSNE(n_components=2).fit_transform(features)   # TSNE 降维方法. python cpu:100%  5min 
println("tsne over...")
function plot_embedding(features, labels)
    x = features[1]
    y = features[2]
    colors = rand(size(features)[1])
    println("begin scatter")
    scene = scatter(x, y, color = colors)   # ???
    # LoadError: No overload for Scatter{...} and also no overload for trait AbstractPlotting.PointBased() found! Arguments: (Float32, Float32)
end 

# X = features;
# Y = tsne(X', 2);   # 512.feat.npy (18171, 384)  很耗内存 7.8G  cpu:800%/100%   need 1.2day
# println("tsne over...")
# theplot = scatter(Y[:,1], Y[:,2], marker=(2,2,:auto,stroke(0)), color=Int.(allabels[1:size(Y,1)]))
# Plots.pdf(theplot, "myplot2.pdf")

plot_embedding(feat_tsne, labels)


"""
ERROR: OutOfMemoryError()


"""


