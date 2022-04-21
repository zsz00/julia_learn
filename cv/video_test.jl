# test video
import ImageView, Makie
using VideoIO   
using Base.Filesystem, CSV, JSON3, UUIDs
using FileIO, Images
using ProgressMeter


function test_1()
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
end

function test_2()
    # 视频抽帧
    video_file = raw"/data/zhangyong/data/tj_vedios/nh_tj_vedios/0900-1000/按时间下载/宝坻八门城支行_宝坻八门城支行新世电01_津-宝坻八门城加钞间_45_03-29 09-00-00__03-29 10-00-00.md"
    # f_v = open(video_file)
    num_frames = VideoIO.get_number_frames(video_file)   # nothing, 不起作用
    f = VideoIO.openvideo(video_file)
    dir_1 = splitext(video_file)[1]
    if !isdir(dir_1) mkdir(dir_1) end
    # seek(f,2.5)  # 跳到多少秒开始, float64
    println(num_frames)

    p = Progress(10000)
    cnt = 0
    while !eof(f)
        cnt += 1
        next!(p)
        if cnt % 10 == 1
            img = read(f)
            save(joinpath(dir_1, "$(cnt).jpg"), img)
        end
    end
end


function test_3()
    # 把一个目录下的所有文件重命名为一个uuid.
    uuid_map = Dict()
    videos_dir = "/data/zhangyong/data/tj_vedios/nh_tj_vedios/1400-1500/按时间下载/"
    for (root, dirs, files) in walkdir(videos_dir)
        println("Directories in $root")
        for dir in dirs
            println(joinpath(root, dir)) # path to directories
        end
        println("Files in $root")
        for file in files
            println(joinpath(root, file)) # path to files
            file_path = joinpath(root, file)
            # 如果是文件, 则重命名
            if isfile(file_path)
                uuid = uuid4()
                file_split = splitext(file)
                println(file, "=>", "$(uuid)$(file_split[end])")   # 保持后缀名
                rename(joinpath(root, file), joinpath(root, "$(uuid)$(file_split[end])"))
                # 记录 原文件名和uuid的映射, 如果不存在, 则创建一个    
                uuid_map[uuid] = file
            end
        end
    end
    # 写入文件
    uuid_map_file = joinpath(videos_dir, "../uuid_map.json")
    # CSV.write(uuid_map_file, uuid_map)
    JSON3.write(uuid_map_file, uuid_map)
end

function test_4()
    # 把一个目录下的所有文件uuid名还原回去.
    videos_dir = "/data/zhangyong/data/tj_vedios/nh_tj_vedios/0900-1000/按时间下载/"
    json_string = read(joinpath(videos_dir, "../uuid_map.json"), String)
    uuid_map = JSON3.read(json_string)
    for (root, dirs, files) in walkdir(videos_dir)
        println("Files in $root")
        for file in files
            println(joinpath(root, file)) # path to files
            file_path = joinpath(root, file)
            # 如果是文件, 则重命名
            if isfile(file_path)
                # uuid = uuid4()
                uuid = splitext(file)[1]
                org_name = uuid_map[uuid]
                println(file, "=>", "$(org_name)")   # 保持后缀名
                rename(joinpath(root, file), joinpath(root, "$(org_name)"))
                # 记录 原文件名和uuid的映射, 如果不存在, 则创建一个    
                # uuid_map[uuid] = file
            end
        end
    end
    # 写入文件
    # uuid_map_file = joinpath(videos_dir, "../uuid_map.json")
    # CSV.write(uuid_map_file, uuid_map)
    # JSON3.write(uuid_map_file, uuid_map)
end

