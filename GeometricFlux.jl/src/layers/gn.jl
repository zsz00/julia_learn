const AGGR2FUN = Dict(:add => sum, :sub => sum, :mul => prod, :div => prod,
                      :max => maximum, :min => minimum, :mean => mean)

abstract type GraphNet end

@inline update_edge(gn::T, e, vi, vj, u) where {T<:GraphNet} = e
@inline update_vertex(gn::T, ē, vi, u) where {T<:GraphNet} = vi
@inline update_global(gn::T, ē, v̄, u) where {T<:GraphNet} = u

@inline function update_batch_edge(gn::T, adj, E, V, u) where {T<:GraphNet}
    n = size(adj, 1)
    edge_idx = edge_index_table(adj)
    hcat([apply_batch_message(gn, i, adj[i], edge_idx, E, V, u) for i in 1:n]...)
end

@inline function apply_batch_message(gn::T, i, js, edge_idx, E, V, u) where {T<:GraphNet}
    hcat([update_edge(gn, get_feature(E, edge_idx[(i,j)]), get_feature(V, i), get_feature(V, j), u) for j = js]...)
end

@inline function update_batch_vertex(gn::T, Ē, V, u) where {T<:GraphNet}
    hcat([update_vertex(gn, get_feature(Ē, i), get_feature(V, i), u) for i = 1:size(V,2)]...)
end

@inline function aggregate_neighbors(gn::T, aggr::Symbol, E, accu_edge) where {T<:GraphNet}
    @assert !iszero(accu_edge) "accumulated edge must not be zero."
    cluster = generate_cluster(E, accu_edge)
    pool(aggr, cluster, E)
end

@inline function aggregate_neighbors(gn::T, aggr::Nothing, E, accu_edge) where {T<:GraphNet}
    @nospecialize E accu_edge num_V num_E
end

@inline function aggregate_edges(gn::T, aggr::Symbol, E) where {T<:GraphNet}
    u = vec(AGGR2FUN[aggr](E, dims=2))
    aggr == :sub && (u = -u)
    aggr == :div && (u = 1 ./ u)
    u
end

@inline function aggregate_edges(gn::T, aggr::Nothing, E) where {T<:GraphNet}
    @nospecialize E
end

@inline function aggregate_vertices(gn::T, aggr::Symbol, V) where {T<:GraphNet}
    u = vec(AGGR2FUN[aggr](V, dims=2))
    aggr == :sub && (u = -u)
    aggr == :div && (u = 1 ./ u)
    u
end

@inline function aggregate_vertices(gn::T, aggr::Nothing, V) where {T<:GraphNet}
    @nospecialize V
end

function propagate(gn::T, fg::FeaturedGraph, naggr=nothing, eaggr=nothing, vaggr=nothing) where {T<:GraphNet}
    E, V, u = propagate(gn, adjacency_list(fg), fg.ef, fg.nf, fg.gf, naggr, eaggr, vaggr)
    FeaturedGraph(graph(fg), V, E, u)
end

function propagate(gn::T, adj::AbstractVector{S}, E::R, V::Q, u::P,
                   naggr=nothing, eaggr=nothing, vaggr=nothing) where {T<:GraphNet,S<:AbstractVector,R,Q,P}
    E = update_batch_edge(gn, adj, E, V, u)
    Ē = aggregate_neighbors(gn, naggr, E, accumulated_edges(adj))
    V = update_batch_vertex(gn, Ē, V, u)
    ē = aggregate_edges(gn, eaggr, E)
    v̄ = aggregate_vertices(gn, vaggr, V)
    u = update_global(gn, ē, v̄, u)
    E, V, u
end
