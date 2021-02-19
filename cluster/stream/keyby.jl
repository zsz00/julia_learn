using Transducers
using Transducers: R_, start, next, complete, inner, xform, wrap, unwrap, wrapping, DefaultInitOf
using Transducers: _reducingfunction, _realbottomrf, unwrap_all, PrivateState, DefaultInit
using BangBang.Experimental: modify!!, mergewith!!
using Strs


struct KeyBy{K, R, T} <: Transducer
    key::K
    rf::R
    init::T
end

function KeyBy(key, xf::Transducer, step = right, init = DefaultInit)
    rf = _reducingfunction(xf, step; init = init)
    return KeyBy(key, rf, init)
end

KeyBy(key, rf) = KeyBy(key, _asmonoid(rf), DefaultInit)


function Transducers.start(rf::R_{KeyBy}, result)
    gstate = Dict{Union{},Union{}}()
    return wrap(rf, gstate, start(inner(rf), result))
end

function Transducers.next(rf::R_{KeyBy}, result, input)
    wrapping(rf, result) do gstate, iresult
        key = xform(rf).key(input)
        # println(f"key:\(key), input:\(input)")
        gstate, somegr = modify!!(gstate, key) do value
            # 如果dict里没有就新建,有就更新
            if value === nothing
                gr0 = start(xform(rf).rf, xform(rf).init)
            else
                gr0 = something(value)
            end
            return Some(next(xform(rf).rf, gr0, key => input))  # Some(gr)
        end

        gr = something(somegr)
        # out_1 = KeyByViewDict(gstate, xform(rf))
        # out = out_1.state[key]  # get(out_1, key)
        out = gstate[key]
        # println(f"out:\(out), key:\(key), input:\(input)")
        iresult = next(inner(rf), iresult, out)
        # println(iresult isa Reduced)
        if gr isa Reduced && !(iresult isa Reduced)
            return gstate, reduced(complete(inner(rf), iresult))
        else
            return gstate, iresult
        end
    end
end

Transducers.complete(rf::R_{KeyBy}, result) = Transducers.complete(inner(rf), unwrap(rf, result)[2])

# function Transducers.combine(rf::R_{KeyBy}, a, b)
#     gstate_a, ira = unwrap(rf, a)
#     gstate_b, irb = unwrap(rf, b)
#     gstate_c = mergewith!!(gstate_a, gstate_b) do ua, ub
#         combine(xform(rf).rf, ua, ub)
#     end
#     irc = combine(inner(rf), ira, irb)
#     irc = next(inner(rf), irc, KeyByViewDict(gstate_c, xform(rf)))
#     return wrap(rf, gstate_c, irc)
# end


struct KeyByViewDict{K,V,S<:DefaultInitOf,D<:AbstractDict{K}} <: AbstractDict{K,V}
    state::D
end

_typesubtract(::Type{Larger}, ::Type{Smaller}) where {Larger,Smaller} =
    _typesubtract_impl(Smaller, Larger)
_typesubtract_impl(::Type{T}, ::Type{T}) where {T} = Union{}
_typesubtract_impl(::Type{T}, ::Type{Union{T,S}}) where {S,T} = S
_typesubtract_impl(::Type, ::Type{S}) where {S} = S

_bottom_state_type(::Type{T}) where {T} = T
_bottom_state_type(::Type{Union{}}) = Union{}
_bottom_state_type(::Type{<:PrivateState{<:Any,<:Any,R}}) where {R} = R

function KeyByViewDict(state::AbstractDict{K,V0}, xf::KeyBy) where {K,V0}
    S = typeof(DefaultInit(_realbottomrf(xf.rf)))
    V = _typesubtract(_bottom_state_type(V0), S)
    return KeyByViewDict{K,V,S,typeof(state)}(state)
end

function Base.get(dict::KeyByViewDict{<:Any,<:Any,S}, key, default) where {S}
    value = unwrap_all(unreduced(dict.state[key]))
    return value isa S ? default : value
end
