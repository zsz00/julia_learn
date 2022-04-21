# struct
using StructIO

@io struct TwoUInt
    x::UInt16
    y::UInt8
end

@io struct MBR
    x::UInt16
    y::UInt8
end

function test_1()
    buf = IOBuffer(collect(UInt8[0x77;0x6f;0x72;0x72])); 
    seekstart(buf)
    aa=unpack(buf, TwoUInt)
    println(aa.x)   # 0x6f77
    println(Int(aa.x))  # 28535 
end

function mbr_1()
    # buf = IOBuffer(collect(UInt8[0x77;0x6f;0x72;0x72])); 
    buf=open("\\\\.\\PHYSICALDRIVE1", "r")
    seekstart(buf, 510)
    aa = read(buf, Int16)
    println(aa)
    # aa=unpack(buf, TwoUInt)
    # println(aa.x)   # 0x6f77
    # println(Int(aa.x))  # 28535 
end



# mbr_1()
# test_1()

