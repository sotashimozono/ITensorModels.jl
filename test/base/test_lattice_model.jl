using ITensorModels
using ITensorModels: LatticeModel, Heisenberg1D, build_opsum, bond_term, site_type
using ITensors, ITensorMPS
using LatticeCore

# LatticeCore ships a minimal 1D chain lattice suitable for smoke tests
# without pulling Lattice2D/QuasiCrystal; we build our own
# minimal-shape AbstractLattice subtype here so these tests don't
# depend on external lattice packages. Lattice2D / QuasiCrystal
# integration is exercised in examples/ once LatticeModel lands.

struct _LineChain <: LatticeCore.AbstractLattice{1,Float64}
    N::Int
end
LatticeCore.num_sites(l::_LineChain) = l.N
LatticeCore.position(l::_LineChain, i::Int) = LatticeCore.StaticArrays.SVector(Float64(i))
function LatticeCore.neighbors(l::_LineChain, i::Int)
    if i == 1
        [2]
    elseif i == l.N
        [l.N - 1]
    else
        [i - 1, i + 1]
    end
end
LatticeCore.boundary(::_LineChain) = LatticeCore.OpenAxis()
LatticeCore.size_trait(::_LineChain) = LatticeCore.FiniteSize()
# Default `bonds(lat)` builds Bond{D,T} objects from `neighbors`.

@testset "LatticeModel on a 1D line chain reproduces plain Heisenberg" begin
    N = 8
    lat = _LineChain(N)

    # Heisenberg on every nearest-neighbour bond.
    model_lat = LatticeModel(;
        lattice=lat, bond_models=Dict(:nearest => Heisenberg1D(; J=1.0))
    )

    sites = siteinds("S=1/2", N)
    H_lat = MPO(build_opsum(model_lat, sites), sites)

    # Reference: direct 1D Heisenberg opsum over (k, k+1) pairs.
    ref_opsum = OpSum()
    for k in 1:(N - 1)
        ref_opsum += bond_term(Heisenberg1D(; J=1.0), k, k + 1)
    end
    H_ref = MPO(ref_opsum, sites)

    # Both Hamiltonians should have identical expectation on the Néel
    # product state (a cheap fingerprint — exactly-equal MPOs would
    # also pass but that's too strict under bond-dim reshaping).
    neel = [isodd(k) ? "Up" : "Dn" for k in 1:N]
    ψ = MPS(sites, neel)
    @test inner(ψ', H_lat, ψ) ≈ inner(ψ', H_ref, ψ) rtol = 1e-12
end

@testset "LatticeModel: ordering remaps MPS positions" begin
    N = 6
    lat = _LineChain(N)

    # Reverse ordering — lattice site k ↔ MPS position N+1-k.
    rev = collect(N:-1:1)
    model = LatticeModel(;
        lattice=lat, bond_models=Dict(:nearest => Heisenberg1D(; J=1.0)), ordering=rev
    )
    sites = siteinds("S=1/2", N)
    opsum = build_opsum(model, sites)

    # Every bond_term should reference MPS positions (rev[i], rev[j])
    # for a LatticeCore bond (i, j). Since every lattice bond is
    # (k, k+1), the MPS pairs are (N+1-k, N-k) — i.e. reversed.
    hit = Set{Tuple{Int,Int}}()
    for term in opsum
        sts = sort(unique(ITensors.sites(term)))
        length(sts) == 2 && push!(hit, (sts[1], sts[2]))
    end
    expected = Set((k, k + 1) for k in 1:(N - 1))
    @test hit == expected
end

@testset "LatticeModel: bond_models dict dispatches on bond.type" begin
    # Mixed coupling model: bond model dict keyed by :nearest fallback.
    lat = _LineChain(4)
    model = LatticeModel(; lattice=lat, bond_models=Dict(:nearest => Heisenberg1D(; J=0.3)))
    sites = siteinds("S=1/2", 4)
    opsum = build_opsum(model, sites)
    # The Heisenberg OpSum on each bond has 3 terms (S+S-, S-S+, SzSz);
    # 3 bonds × 3 terms = 9 non-trivial terms.
    @test length(opsum) == 3 * 3
end

@testset "LatticeModel: mismatched bond_models site_type errors" begin
    lat = _LineChain(4)
    model = LatticeModel(;
        lattice=lat,
        bond_models=Dict(
            :nearest => Heisenberg1D(; J=1.0),
            :type_2 => ITensorModels.TFIM(; site=ITensors.SiteType("Qubit")),
        ),
    )
    @test_throws ErrorException site_type(model)
end
