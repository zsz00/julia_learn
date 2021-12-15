# online MeanVar  2020.11.18
using Transducers
using Transducers: R_, start, next, complete, inner, xform, wrap, unwrap, wrapping
using Folds, FoldsThreads
using Test


struct MeanVar <: Transducer
end


function Transducers.start(rf::R_{MeanVar}, result)
    private_state = (0, 0.0, 0.0)   # 初始化私有状态
    result = wrap(rf, private_state, start(inner(rf), result))
    return result
end

function Transducers.next(rf::R_{MeanVar}, result, input)
    wrapping(rf, result) do st, iresult
        (n, μ, M2) = st
        n += 1
        δ = input - μ
        μ += δ/n
        δ2 = input - μ
        M2 += δ*δ2
        iinput = (μ, M2 / (n-1))
        iresult = next(inner(rf), iresult, iinput)
        return (n, μ, M2), iresult
    end
end

function Transducers.complete(rf::R_{MeanVar}, result)
    _private_state, inner_result = unwrap(rf, result)
    result = complete(inner(rf), inner_result)
    return result
end


# aa = collect(MeanVar(), 1:10^6)
# println(aa)

# @time println(foldl(right, MeanVar(), 1:10^8))
# @time 1:10^8  |> MeanVar() |> collect    # 6s
@time foldxt(right, 1:10^8 |> Partition(100,100) |> Map(sum) |> tcollect)  # 88s
# ERROR: LoadError: Stateful transducer MeanVar() does not support `combine`
# |> Partition(100,100)
# KeyBy((x -> x.device_id), op_st_1)


