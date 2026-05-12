using ITensorModels
using ITensors
using ITensors: SiteType
using ITensorMPS
using Random
using Test

@testset "PXP1D: construction" begin
    m = PXP1D()
    @test m isa AbstractLatticeModel
    @test m.Ω == 1.0
    @test site_type(m) == SiteType("Qubit")
end

@testset "PXP1D: bond_term is empty (3-site model)" begin
    m = PXP1D()
    @test length(collect(ITensors.terms(bond_term(m, 1, 2)))) == 0
    @test length(collect(ITensors.terms(bond_coupling_term(m, 1, 2)))) == 0
    @test length(collect(ITensors.terms(onsite_term(m, 1)))) == 0
end

@testset "PXP1D: local_ham_terms emits N OpSums" begin
    m = PXP1D(; Ω=1.0)
    N = 5
    terms = local_ham_terms(m, 1:N; boundary=:full)
    @test length(terms) == N
    @test length(collect(ITensors.terms(terms[1]))) == 2
    @test length(collect(ITensors.terms(terms[N]))) == 2
    @test length(collect(ITensors.terms(terms[3]))) == 4
end

@testset "PXP1D: build_opsum on Qubit sites" begin
    N = 6
    m = PXP1D(; Ω=1.0)
    sites = siteinds("Qubit", N)
    H = build_opsum(m, sites; phys_sites=1:N, boundary=:full)
    @test length(collect(ITensors.terms(H))) == 4 * N - 4
end

@testset "PXP1D: onsite_observable_op" begin
    m = PXP1D()
    @test onsite_observable_op(m, :x) == "X"
    @test onsite_observable_op(m, :z) == "Z"
    @test_throws ErrorException onsite_observable_op(m, :bogus)
end

@testset "PXP1D: DMRG smoke" begin
    N = 8
    m = PXP1D(; Ω=1.0)
    sites = siteinds("Qubit", N)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)
    psi0 = random_mps(MersenneTwister(0xa9), sites; linkdims=12)
    sweeps = Sweeps(15)
    maxdim!(sweeps, 20, 50, 100, 150, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200)
    cutoff!(sweeps, 1e-12)
    E, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test isfinite(E)
end
