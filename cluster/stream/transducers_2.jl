# Stateful transducer
using Transducers
using Transducers: start, complete, wrap, unwrap, wrapping
using Transducers: Transducer, R_, next, inner, xform
using Random


struct RandomRecall <: Transducer
    history::Int
    seed::Int
end

RandomRecall() = RandomRecall(3, 0)   # 初始化结构体

# 初始化
function Transducers.start(rf::R_{RandomRecall}, result)
    buffer = []
    rng = MersenneTwister(xform(rf).seed)  # 随机数生成器
    private_state = (buffer, rng)   # 私有状态
    return wrap(rf, private_state, start(inner(rf), result))  # 包装
end

# iter
function Transducers.next(rf::R_{RandomRecall}, result, input)
    wrapping(rf, result) do (buffer, rng), iresult   # 包装
        if length(buffer) < xform(rf).history
            push!(buffer, input)
            iinput = rand(rng, buffer)
        else
            i = rand(rng, 1:length(buffer))
            iinput = buffer[i]
            buffer[i] = input
        end
        iresult = next(inner(rf), iresult, iinput)
        return (buffer, rng), iresult
    end
end

# 完成
function Transducers.complete(rf::R_{RandomRecall}, result)
    _private_state, inner_result = unwrap(rf, result) # 解包
    return complete(inner(rf), inner_result)
end


a = collect(RandomRecall(), 1:5)
b = collect(RandomRecall(), 3:8)
print(a, b)

 