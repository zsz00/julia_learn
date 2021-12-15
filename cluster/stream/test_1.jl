using Transducers
using Transducers: R_, DefaultInit, _reducingfunction, PrivateState
using Strs
using BangBang
include("keyby.jl")


mutable struct Node <: Any
    n_id::Int  # node id
    n_type::String
    n_cnt::Int
end

function test_1()
    # 先把100nodes, 按分组进行 奇数*3, 偶数*2的处理, 然后再 把分组处理的结果按顺序各自+1,最后都放到一个数组. 
    # First, process 100nodes by grouping odd*3, even*2, and then put the grouped results into an array with +1 each. 
    # this is a pseudo-code, This code doesn't work properly. 
    data = []
    for i in 1:10
        # type = string(isodd(i))
        type = string(i % 3)
        push!(data, Node(i, type, 0))
    end

    println(data)
    op_1 = Map(f_op_1)   # 太简单,无状态
    op_2 = Map(f_op_2)
    node_1 = Transducers.foldl(right, data |> KeyBy(x->x.n_type, op_1) |> op_2 |> collect)  
    # |> GroupBy((x->x.n_type), op_1)  eachrow(data)   |> op_2  push!!
end


function f_op_1(x)
    # println(f"\(typeof(x)), \(x)")
    key_1 = x[1]   # Key
    node_1 = x[2]   # Node
    node_1.n_cnt += 1
    if node_1.n_type == "1"
        node_1.n_id = node_1.n_id * 10
    else
        node_1.n_id = node_1.n_id * 1
    end
    return node_1
end

function f_op_2(x)
    # x: GroupByViewDict
    println(f"x.state: \(x)")   # state, 没有key
    # println(f"\(collect(values(x.state[key])))")
    nodes = x  # collect(values(x))  # n 个, n是分组数
    # Transducers.GroupByViewDict()
    # Transducers.combine()
    # for node_1 in nodes:
    #     node_1.n_id = node_1.n_id + 1
    # println(node_1)
    # return node_1
end

function test_async()
    k = 0
    bs = 99999
    idxs = zeros(Int32, 1000)
    @sync for i in 1:1000
        @async begin 
            k = i+1
            if i == 9
                bs = 99   
            end
            sleep(1)
            kk1 = get_data1(i)  # 4s
            kk2 = get_data2(kk1)  # 2s
            println(f"\(i), \(i+1), \(bs)")
            idxs[i] = kk2
        end
    end
    println(idxs)
end


function get_data1(i)
    sleep(4)
    kk = i + 1
    return kk
end

function get_data2(i)
    sleep(2)
    kk = i*2
    return kk
end


# test_1()
@time test_async()


#=
# 问题: group by后,怎么reduce和继续op处理
example:
data:
Any[Node(1, "true"), Node(2, "false"), Node(3, "true"), Node(4, "false"), Node(5, "true"), 
Node(6, "false"), Node(7, "true"), Node(8, "false"), Node(9, "true"), Node(10, "false")]

export JULIA_NUM_THREADS=4


=#

