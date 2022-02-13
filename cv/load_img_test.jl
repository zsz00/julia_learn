# test image read 
using BenchmarkTools
using JpegTurbo, FileIO, Images   # ImageMagick JpegTurbo
# using Images  #, ImageView
using PyCall
cv2 = pyimport("cv2")


function julia_img(img1)
    im1 = load(img1);
    # im1 = Gray.(im1);
    # im1 = imresize(im1, (300, 300));
    return 0
#     imshow(im1)
end


function python_img(img1)
    img = cv2.imread(img1) 
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_CUBIC)
    return 0
#     imshow(im1)
end


img1 = s"others/stitcher/imgs/1.jpg"
# img1 = "/root/julia/imgs/00000000210000000_02868.jpg"  # stata-1.png

@time for i in 1:100
    julia_img(img1)
end

@time for i in 1:100
    python_img(img1)
end


@btime rand(RGB{Float64}, 20, 20);
@btime load(img1);

@btime julia_img(img1)
@btime python_img(img1)


#=
python3 : time: 0.19165117740631105 s
2018.11
读图片 julia Images 比 python opencv 慢5倍.  test on ubuntu16.04 阿里云

------------
  7.030900 seconds (10.27 M allocations: 608.996 MiB, 4.43% gc time)
  9.890474 seconds (116.12 M allocations: 4.527 GiB, 10.21% gc time)

  4.989357 seconds (6.07 M allocations: 378.117 MiB, 3.98% gc time)
  5.844481 seconds (83.96 M allocations: 3.373 GiB, 7.90% gc time)
-------------------------------------------------------------------------
2020.10.28    
Images v0.23.1   test on ubuntu16.04  local server 3.199
others/stitcher/imgs/1.jpg
第三次: julia: python      load only
12.174273 seconds (25.03 M allocations: 1.278 GiB, 7.13% gc time)
  0.721582 seconds (1.01 M allocations: 78.845 MiB, 3.71% gc time)
只读图片: julia Images 比 python opencv 慢17倍.
---------------
第三次: julia: python      load+Gray+resize
 13.718103 seconds (27.95 M allocations: 1.446 GiB, 7.03% gc time)
  1.783276 seconds (2.68 M allocations: 172.914 MiB, 3.21% gc time)
  17.596 μs (1 allocation: 9.50 KiB)
---------------
2020.10.28   
改了 using ImageMagick, FileIO, Images
others/stitcher/imgs/1.jpg
第三次: julia: python      load only
  8.032526 seconds (23.18 M allocations: 1.183 GiB, 8.62% gc time)
  0.749043 seconds (1.01 M allocations: 78.864 MiB, 5.47% gc time)
  18.117 μs (1 allocation: 9.50 KiB)
  43.513 ms (240 allocations: 7.92 MiB)
只读图片: julia Images 比 python opencv 慢10倍.
---------------
第三次: julia: python      load+Gray+resize
  9.342521 seconds (26.10 M allocations: 1.350 GiB, 8.17% gc time)
  1.856433 seconds (2.68 M allocations: 172.939 MiB, 3.30% gc time)
  17.911 μs (1 allocation: 9.50 KiB)
  44.158 ms (240 allocations: 7.92 MiB)
读图片+Gray+resize: julia Images 比 python opencv 慢5倍.


结论: jpg格式的图片
只读图片: julia Images 比 python opencv 慢10倍.
读图片+Gray+resize: julia Images 比 python opencv 慢5倍.

ImageIO 里目前只实现了png,读png的速度快些. 其他的都还是用比较慢的 ImageMagick

-------------------------------------------------------------------------
2022.2.11    
Images v0.25.1 FileIO v1.13.0  test on ubuntu18.04   10.9.1.8
others/stitcher/imgs/1.jpg

第三次: julia: python      load only
ImageMagick:
  8.105774 seconds (115.49 M allocations: 4.040 GiB, 9.07% gc time, 31.74% compilation time)
  0.576424 seconds (1.22 M allocations: 92.740 MiB, 2.80% gc time, 60.08% compilation time)
  3.455 μs (1 allocation: 9.50 KiB)
  22.865 ms (103 allocations: 5.34 MiB)
JpegTurbo:
  2.549560 seconds (4.58 M allocations: 299.852 MiB, 4.03% gc time, 89.48% compilation time)
  0.617354 seconds (1.43 M allocations: 104.300 MiB, 5.09% gc time, 63.18% compilation time)
  3.633 μs (1 allocation: 9.50 KiB)
  22.898 ms (103 allocations: 5.34 MiB)
  22.875 ms (103 allocations: 5.34 MiB)
  18.908 ms (41 allocations: 2.64 MiB)
---------------
第三次: julia: python      load+Gray+resize
  3.328274 seconds (7.11 M allocations: 453.854 MiB, 4.46% gc time, 90.61% compilation time)
  1.975920 seconds (4.09 M allocations: 257.401 MiB, 3.08% gc time, 81.97% compilation time)
  3.446 μs (1 allocation: 9.50 KiB)
  22.898 ms (103 allocations: 5.34 MiB)
  26.348 ms (110 allocations: 6.31 MiB)
  27.962 ms (149 allocations: 3.61 MiB)

结论: jpg格式的图片.  有了JpegTurbo.jl,用@btime计时更合理
只读图片: julia Images 比 python opencv 耗时 1.2:1.
读图片+Gray+resize: julia Images 比 python opencv 耗时 1:1.

为什么 @time 的和 @btime的统计差异那么大? 因为第一次调用有编译,很耗时,90%的编译时间.

=#

