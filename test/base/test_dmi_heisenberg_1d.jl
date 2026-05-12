using ITensorModels
using ITensors
using ITensors: SiteType
using ITensorMPS
using Random
using Test

@testset "DMIHeisenberg1D: construction" begin
    m = DMIHeisenberg1D()
    @test m isa AbstractLatticeModel
    @test m.J == 1.0
    @test m.D == (0.0, 0.0, 1.0)
    @test site_type(m) == SiteType("S=1/2")
end

@testset "DMIHeisenberg1D: bond_term has 9 terms" begin
    m = DMIHeisenberg1D(; J=1.0, D=(0.5, 0.3, 0.7))
    H = bond_term(m, 1, 2)
    @test length(collect(ITensors.terms(H))) == 9
end

@testset "DMIHeisenberg1D: onsite_observable_op" begin
    m = DMIHeisenberg1D()
    @test onsite_observable_op(m, :sx) == "Sx"
    @test onsite_observable_op(m, :sy) == "Sy"
    @test onsite_observable_op(m, :sz) == "Sz"
    @test_throws ErrorException onsite_observable_op(m, :bogus)
end

@testset "DMIHeisenberg1D: build_opsum + ModulatedModel smoke" begin
    N = 6
    m = DMIHeisenberg1D(; J=1.0, D=(0.0, 0.0, 0.3))
    sites = siteinds("S=1/2", N)
    H = build_opsum(m, sites; phys_sites=1:N, boundary=:full)
    @test length(collect(ITensors.terms(H))) == 9 * (N - 1)

    m_ssd = modulated(m; L=N, modulation=SSD())
    H_ssd = build_opsum(m_ssd, sites; phys_sites=1:N, boundary=:full)
    @test length(collect(ITensors.terms(H_ssd))) > 0
end

@testset "DMIHeisenberg1D: DMRG smoke" begin
    N = 8
    m = DMIHeisenberg1D(; J=1.0, D=(0.0, 0.0, 0.3))
    sites = siteinds("S=1/2", N)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)
    psi0 = random_mps(MersenneTwister(0xd1), sites; linkdims=16)
    sweeps = Sweeps(15)
    maxdim!(sweeps, 20, 50, 100, 150, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200)
    cutoff!(sweeps, 1e-12)
    E, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test isfinite(E)
end
