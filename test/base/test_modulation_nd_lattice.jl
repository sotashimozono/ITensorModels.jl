using ITensorModels
using LatticeCore
using LatticeCore: position, num_sites
using Lattice2D
using Test

@testset "center_position: LineLattice (1D-equivalent)" begin
    # LatticeCore's LineLattice uses 1-indexed positions: site k sits at
    # position(lat, k) = SVector{1, Float64}(k). Geometric and
    # bounding-box centers both land on (L + 1) / 2.
    L = 8
    lat = LineLattice(L, OpenAxis())
    rc_geo = ITensorModels.center_position(GeometricCenter(), lat)
    rc_bbx = ITensorModels.center_position(BoundingBoxCenter(), lat)
    @test rc_geo[1] ≈ (L + 1) / 2
    @test rc_bbx[1] ≈ (L + 1) / 2
end

@testset "center_position: honeycomb (BoundingBox)" begin
    lat = Lattice2D.honeycomb(6, 6; boundary=OpenAxis())
    ps = collect(LatticeCore.positions(lat))
    rc = ITensorModels.center_position(BoundingBoxCenter(), lat)
    # BoundingBoxCenter is per-axis midpoint of (min, max).
    for d in 1:length(rc)
        lo = minimum(p[d] for p in ps)
        hi = maximum(p[d] for p in ps)
        @test rc[d] ≈ (lo + hi) / 2
    end
end

@testset "center_position: ExplicitCenter validates dimension" begin
    lat = Lattice2D.honeycomb(4, 4; boundary=OpenAxis())
    @test_throws ErrorException ITensorModels.center_position(
        ExplicitCenter([0.0, 0.0, 0.0]), lat   # 3D vector for 2D lattice
    )
    rc = ITensorModels.center_position(ExplicitCenter([1.5, 2.5]), lat)
    @test rc[1] ≈ 1.5
    @test rc[2] ≈ 2.5
end

@testset "distance_at: defers to distance_at_position" begin
    lat = Lattice2D.honeycomb(4, 4; boundary=OpenAxis())
    rc = ITensorModels.center_position(BoundingBoxCenter(), lat)
    p1 = position(lat, 1)
    expected = sqrt(sum((p1 .- rc).^2))
    @test ITensorModels.distance_at(EuclideanDistance(), lat, 1, rc) ≈ expected
    @test ITensorModels.distance_at(AxialDistance(1), lat, 1, rc) ≈ abs(p1[1] - rc[1])
end

@testset "site_weight / site_envelope: honeycomb Spherical SSD" begin
    lat = Lattice2D.honeycomb(6, 6; boundary=OpenAxis())
    rc = ITensorModels.center_position(BoundingBoxCenter(), lat)
    ps = collect(LatticeCore.positions(lat))
    # Choose R = max distance from center so the most distant site sits
    # at the envelope boundary (weight ≈ 0).
    R = sqrt(maximum(sum((p .- rc).^2) for p in ps))
    env = RadialEnvelope(BoundingBoxCenter(), EuclideanDistance(), SinSquareProfile(R))

    weights = [ITensorModels.site_weight(env, lat, k) for k in 1:num_sites(lat)]
    for k in 1:num_sites(lat)
        @test ITensorModels.site_envelope(env, lat, k) ≈ weights[k]
    end
    @test all(w -> 0 <= w <= 1, weights)
    # The most-central site carries near-unit weight; corners are near zero.
    k_central = argmin([sqrt(sum((ps[k] .- rc).^2)) for k in 1:length(ps)])
    @test weights[k_central] >= 0.9
    k_corner = argmax([sqrt(sum((ps[k] .- rc).^2)) for k in 1:length(ps)])
    @test weights[k_corner] <= 0.05
end

@testset "site_weight: AxialDistance reduces the envelope to one axis" begin
    lat = Lattice2D.honeycomb(6, 6; boundary=OpenAxis())
    rc = ITensorModels.center_position(BoundingBoxCenter(), lat)
    ps = collect(LatticeCore.positions(lat))
    Rx = maximum(abs(p[1] - rc[1]) for p in ps)
    env = RadialEnvelope(BoundingBoxCenter(), AxialDistance(1), SinSquareProfile(Rx))

    # Sites with the same x-coordinate but different y must carry the
    # same site_weight (envelope is uniform in y).
    by_x = Dict{Float64,Vector{Int}}()
    for k in 1:num_sites(lat)
        x = ps[k][1]
        push!(get!(by_x, x, Int[]), k)
    end
    same_x_groups = [v for v in values(by_x) if length(v) > 1]
    @test !isempty(same_x_groups)
    for group in same_x_groups
        ws = [ITensorModels.site_weight(env, lat, k) for k in group]
        @test all(w -> w ≈ ws[1], ws)
    end
end

@testset "bond_weight: midpoint evaluation" begin
    lat = Lattice2D.honeycomb(4, 4; boundary=OpenAxis())
    rc = ITensorModels.center_position(BoundingBoxCenter(), lat)
    ps = collect(LatticeCore.positions(lat))
    R = sqrt(maximum(sum((p .- rc).^2) for p in ps)) + 0.5
    env = RadialEnvelope(BoundingBoxCenter(), EuclideanDistance(), SinSquareProfile(R))

    bs = collect(bonds(lat))
    for b in bs[1:min(end, 10)]
        r_mid = (position(lat, b.i) + position(lat, b.j)) / 2
        d = sqrt(sum((r_mid .- rc).^2))
        expected = d >= R ? 0.0 : 1 - sin(pi * d / (2R))^2
        @test ITensorModels.bond_weight(env, lat, b.i, b.j) ≈ expected
    end
end

@testset "bond_weight: order-independence (symmetric in i, j)" begin
    lat = Lattice2D.honeycomb(4, 4; boundary=OpenAxis())
    rc = ITensorModels.center_position(BoundingBoxCenter(), lat)
    ps = collect(LatticeCore.positions(lat))
    R = sqrt(maximum(sum((p .- rc).^2) for p in ps))
    env = RadialEnvelope(BoundingBoxCenter(), EuclideanDistance(), SinSquareProfile(R))
    bs = collect(bonds(lat))
    for b in bs[1:min(end, 8)]
        @test ITensorModels.bond_weight(env, lat, b.i, b.j) ≈
            ITensorModels.bond_weight(env, lat, b.j, b.i)
    end
end
