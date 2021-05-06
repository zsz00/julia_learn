# test PyCall 
# 从目录里调用 *.py文件. OK
using PyCall

# # pushfirst!(PyVector(pyimport("sys")."path"),"")
# pushfirst!(PyVector(pyimport("sys")."path"), "c:\\zsz\\ML\\code\\julia\\julia_learn\\cv")
# println("python path:", pyimport("sys")."path")
# np = pyimport("numpy")
# os_win = pyimport("os_win") 
# os_win.main()

sp = pyimport("scipy.sparse") 
# from scipy.sparse import csr_matrix,save_npz,load_npz

dir_2 = "/data/yongzhang/cluster/data_3/clean/out_disk_2_jxx_2"
mat_file = joinpath(dir_2, "disk_2_jxx_local_mat.npz")
csr = sp.load_npz(mat_file)
println("csr:", csr.shape)
mat = csr.data
idx = csr.indices
println(size(mat), " idx:", size(idx))


