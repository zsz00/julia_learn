# test PyCall 
# 从目录里调用 *.py文件. OK
using PyCall
# pushfirst!(PyVector(pyimport("sys")."path"),"")
pushfirst!(PyVector(pyimport("sys")."path"), "c:\\zsz\\ML\\code\\julia\\julia_learn\\cv")

println("python path:", pyimport("sys")."path")

np = pyimport("numpy")
os_win = pyimport("os_win") 

os_win.main()

