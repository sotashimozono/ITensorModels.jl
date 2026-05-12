using ITensorModels
using ITensors
using ITensors: SiteType
using ITensorMPS
using Random
using Test

@testset "SSH1D: construction" begin
    m = SSH1D()
    @test m isa AbstractLatticeModel
    @test m.t1 == 0.8
    @test m.t2 == 1.0
    @test site_type(m) == SiteType("Fermion")
end

@testset "SSH1D: pairwise / onsite all empty" begin
    m = SSH1D(; t1=0.8, t2=1.2)
    @test length(collect(ITensors.terms(bond_term(m, 1, 2)))) == 0
    @test length(collect(ITensors.terms(bond_coupling_term(m, 1, 2)))) == 0
    @test length(collect(ITensors.terms(onsite_term(m, 1)))) == 0
end

@testset "SSH1D: local_ham_terms alternates t1/t2" begin
    N = 5
    m = SSH1D(; t1=0.7, t2=1.3)
    terms = local_ham_terms(m, 1:N; boundary=:full)
    @test length(terms) == N - 1
    for (i, t) in enumerate(terms)
        coefs = [ITensors.coefficient(x) for x in ITensors.terms(t)]
        expected = isodd(i) ? -m.t1 : -m.t2
        @test all(c ≈ expected for c in coefs)
    end
end

@testset "SSH1D: build_opsum term count" begin
    N = 6
    m = SSH1D(; t1=0.5, t2=1.0)
    sites = siteinds("Fermion", N; conserve_nf=false)
    H = build_opsum(m, sites; phys_sites=1:N, boundary=:full)
    @test length(collect(ITensors.terms(H))) == 2 * (N - 1)
end

@testset "SSH1D: onsite_observable_op" begin
    m = SSH1D()
    @test onsite_observable_op(m, :n) == "N"
    @test_throws ErrorException onsite_observable_op(m, :bogus)
end

@testset "SSH1D: topological DMRG smoke (half-filling, t2 > t1)" begin
    N = 6
    m = SSH1D(; t1=0.5, t2=1.0)
    sites = siteinds("Fermion", N; conserve_nf=true)
    state = [isodd(i) ? "Occ" : "Emp" for i in 1:N]
    psi0 = MPS(sites, state)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)
    sweeps = Sweeps(15)
    maxdim!(sweeps, 20, 50, 100, 150, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200)
    cutoff!(sweeps, 1e-12)
    E, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test isfinite(E)
    @test E < 0
end

@testset "SSH1D: trivial DMRG smoke (t1 > t2)" begin
    N = 6
    m = SSH1D(; t1=1.0, t2=0.5)
    sites = siteinds("Fermion", N; conserve_nf=true)
    state = [isodd(i) ? "Occ" : "Emp" for i in 1:N]
    psi0 = MPS(sites, state)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)
    sweeps = Sweeps(15)
    maxdim!(sweeps, 20, 50, 100, 150, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200)
    cutoff!(sweeps, 1e-12)
    E, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test isfinite(E)
    @test E < 0
end
