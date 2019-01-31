# test image read ,  ssim
using Images  #, ImageView
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


img1 = s"stitcher\test\imgs\1.jpg"
# img1 = "/root/julia/imgs/00000000210000000_02868.jpg"  # stata-1.png

@time for i in 1:10
    julia_img(img1)
end

@time for i in 1:10
    python_img(img1)
end


#=
python3 : time: 0.19165117740631105 s

@time ssim(img1, img2)
  0.978499 seconds (551 allocations: 143.073 MiB, 11.40% gc time)
绝大部分是 load的时间.
读图片 julia Images 比 python opencv 慢5倍.  test on ubuntu16.04 阿里云

time: 0.03723652362823486 s
time: 0.06383512020111085 s

------------
  7.030900 seconds (10.27 M allocations: 608.996 MiB, 4.43% gc time)
  9.890474 seconds (116.12 M allocations: 4.527 GiB, 10.21% gc time)

  4.989357 seconds (6.07 M allocations: 378.117 MiB, 3.98% gc time)
  5.844481 seconds (83.96 M allocations: 3.373 GiB, 7.90% gc time)
=#

