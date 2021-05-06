# PELCO-D协议的云台控制   2018.12.6   test on win10/ubuntu16.04
# 在linux上需要替换下默认的内核驱动(https://stackoverflow.com/questions/13419691/accessing-a-usb-device-with-libusb-1-0-as-a-non-root-user)
# 天津中安视通 ZA-PT520云台
using LibFTD2XX  # add https://github.com/Gowerlabs/LibFTD2XX.jl.git
# Julia wrapper for FTD2XX driver.

function init()
    # 初始化
    devs = createdeviceinfolist()
    println("devs: ", devs)
    list, elnum = getdeviceinfolist(devs)
    description = String(list[1].description)
    handle = ftopen(0)   # 开
    # handle = LibFTD2XX.open(description, OPEN_BY_DESCRIPTION)   # 开 
    # close(handle)   # 关
    open = isopen(handle)  # 检测是否 打开
    datacharacteristics(handle, wordlength = BITS_8, stopbits = STOP_BITS_1, parity = PARITY_NONE)
    baudrate(handle, 9600)
    return open, handle
end

function ctl_vertical(handle, degree=0, sleep_time=1)
    println("垂直旋转: $(degree) 度")
    # ctl = UInt8[0xFF;0x01;0x00;0x4D;0x80;0xE8;0xB6]
    # ------------------------------------------------
    if degree < 0
        degree = -degree*100
    else
        degree = 36000-degree*100
    end
    a1 = [0xFF,0x01,0x00,0x4D]
    rotate = hex2bytes(string(degree, base = 16, pad = 4))   # 旋转30度, 3000
    ctl_1 = vcat(a1, rotate)
    ctl_chk = UInt8(sum(ctl_1[2:end])%0x100)  # 校验
    ctl = vcat(ctl_1, ctl_chk)
    # println(hcat(ctl))
    ss = write(handle, ctl)  # 发送控制指令ctl_1
    # println("status: ", ss)
    sleep(sleep_time)
end

cc = 1
function ctl_horizontal(handle, degree=0, sleep_time=1)
    println("水平旋转: $(degree) 度")
    a1 = [0xFF,0x01,0x00,0x4B]
    rotate = hex2bytes(string(degree*100, base = 16, pad = 4))   # 旋转, 3000
    ctl_1 = vcat(a1, rotate)
    # 检验码计算公式：（字节2+字节3+字节4+字节5+字节6）% 0x100
    ctl_chk = UInt8(sum(ctl_1[2:end])%0x100)  # 校验码
    ctl = vcat(ctl_1, ctl_chk)

    ss = write(handle, ctl)  # 发送控制指令
    # println("status: ", ss)  # 7 FT_INVALID_PARAMETER
    sleep(sleep_time)   # 停止 3s

    # 垂直 旋转   ---------------------------------------
    degree_2 = 20  # 垂直 步长
    n = Int32(40/degree_2)  # 4
    println("垂直 $(2n)步, 每步 $(degree_2)度")
    global cc = -1*cc
    n = cc*n
    for i in (-n: cc : n)  # (-20, 20)
       # println("垂直 i: ", i)
        ctl_vertical(handle, (i)*degree_2, sleep_time)   # 垂直旋转
    end
end

function main()
    open = false
    open, handle = init()  # 初始化设备连接
    println("device open: ", open)
    # ss = write(handle, UInt8[0xFF;0x01;0x00;0x4B;0x00;0x00;0x4C])  # 水平转到0 
    # ctl_vertical(handle, 0, 1)    # 垂直旋转  (-73, 40)
    # # ctl_horizontal(handle, 0, 1)  # 水平旋转  (0, 360)
    # close(handle)
    # return 0

    sleep_time = 5  # 停止间隔
    degree_1 = 31    # 水平步长
    n = Int32(ceil(360/degree_1))
    for i in (1:n)
        # println("水平 i: ", i)
        ctl_horizontal(handle, (i-1)*degree_1, sleep_time)  # 水平旋转
    end
    close(handle)
end


main()  


# ubuntu run: $ sudo julia PTZ_ctl.jl   
