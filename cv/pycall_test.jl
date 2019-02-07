# test PyCall
using PyCall, FileIO, ImageMagick 

# plt = pyimport("matplotlib.pyplot")
np = pyimport("numpy")
cv2 = pyimport("cv2") 

# img_path = s"C:\zsz\ML\code\DL\img_tools\test\img\2\2.jpg"
# img_path = s"stitcher\test\imgs\1.jpg"
# img = cv2.imread(img_path)  # 新版支持
# img = load(img_path)
# img = np.ndarray(img)
# plt.imshow(img); plt.show()   # WIN10 上显示有问题, 弹Qt的报错.
# cv2.imshow("raw", img); cv2.waitKey()

# opencv stitcher class 
function stitcher()
    # img_path1 = s"stitcher\test\imgs\1.jpg"
    # img_path2 = s"stitcher\test\imgs\2.jpg"
    # img1 = cv2.imread(img_path1)
    # img2 = cv2.imread(img_path2)
    # imgs = [img1, img2]
    path = s"stitcher\test\imgs"
    filenames = [joinpath(path, name) for name in readdir(path)]
    imgs = [cv2.imread(each) for each in filenames]
    # cv2.imshow("img1 ", img1); cv2.waitKey()

    stitcher = cv2.createStitcher()  # createDefault  createStitcher
    retval, pano = stitcher.stitch(imgs)  # 拼接. 效果还可以. 默认是球形拼接
    # print('stitched status:',  stat[retval])  # 0是正常, 1是失败, 2是单应性评估错误, 3是相机参数调整失败
    if retval == 0
        out_file = "cv/out.jpg"
        cv2.imwrite(out_file, pano)
        print("write pano file done !!")
        cv2.imshow("pano ", pano)  # 可能显示不全
        cv2.waitKey()
    end
end

stitcher()

#=
pycall old mode
# @pyimport numpy as np
# @pyimport matplotlib.pyplot as plt
# cv2 = PyCall.pyimport("cv2")
# img_path = "test/16.jpg"
# img = cv2[:imread](img_path)
# cv2[:imshow]("raw", img); cv2[:waitKey]()
# plt[:imshow](img); plt[:show]()

=#

