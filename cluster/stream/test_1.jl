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


test_1()


#=
# 问题: group by后,怎么reduce和继续op处理
example:
data:
Any[Node(1, "true"), Node(2, "false"), Node(3, "true"), Node(4, "false"), Node(5, "true"), 
Node(6, "false"), Node(7, "true"), Node(8, "false"), Node(9, "true"), Node(10, "false")]

GroupBy out:
"true" : [1,3,5,7,9]
"flase": [2,4,6,8,10]

node_1 out:
[3,4,9,8,15,12,21,16,27,20]

"true" => [Node(3, "true"), Node(9, "true"), Node(15, "true"), Node(21, "true"), Node(27, "true")],
"false" => [Node(4, "false"), Node(8, "false"), Node(12, "false"), Node(16, "false"), Node(20, "false")])


Dict{String,Node}("true" => Node(3, "true"),"false" => Node(4, "false"))
combine 联结

=#

