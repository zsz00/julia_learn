# test video
import ImageView, Makie, VideoIO   

# VideoIO.viewcam()  # open the camera.  Makie 必须要先import.  
# f = VideoIO.opencamera(0)
video_file = raw"C:\Users\zsz\Downloads\81203450-1-208.mp4"
f = VideoIO.openvideo(video_file)

# One can seek to an arbitrary position in the video
seek(f,2.5)  ## The second parameter is the time in seconds and must be Float64
img = read(f)

# canvas, _ = ImageView.view(img)
# while !eof(f)
#     read!(f, img)
#     ImageView.imshow(canvas, img)
#     #sleep(1/30)
# end

scene = Makie.Scene(resolution = size(img))
makieimg = Makie.image!(scene, buf, show_axis = false, scale_plot = false)[end]  # buf ???
Makie.rotate!(scene, -0.5pi)
display(scene)

while !eof(f)
    read!(f, img)
    makieimg[1] = img
    #sleep(1/30)
end
