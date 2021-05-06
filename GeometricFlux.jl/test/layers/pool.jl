cluster = [1 1 1 1; 2 2 3 3; 4 4 5 5]
X = Array(reshape(1:24, 2, 3, 4))

@testset "pool" begin
    @testset "GlobalPool" begin
        glb_cltr = [1 1 1 1; 1 1 1 1; 1 1 1 1]
        p = GlobalPool(:add, 3, 4)
        @test p(X) == sumpool(glb_cltr, X)
    end

    @testset "LocalPool" begin
        p = LocalPool(:add, cluster)
        @test p(X) == sumpool(cluster, X)
    end

    @testset "TopKPool" begin
        N = 10
        k, in_channel = 4, 7
        X = rand(in_channel, N)
        for T = [Bool, Float64]
            adj = rand(T, N, N)
            p = TopKPool(adj, k, in_channel)
            @test eltype(p.p) === Float32
            @test size(p.p) == (in_channel,)
            @test eltype(p.Ã) === T
            @test size(p.Ã) == (k, k)
            y = p(X)
            @test size(y) == (in_channel, k)
        end
    end
end
