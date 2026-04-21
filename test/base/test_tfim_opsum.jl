using ITensorModels
using ITensorModels: TFIM, build_opsum, bond_term, boundary_patch,
    local_ham_terms, site_type
using ITensors, ITensorMPS
using ITensors: SiteType
using ITensorSiteKit: PhysInds, EnvInds, PhysSite
using Random

N = 6

@testset "TFIM struct" begin
    m = TFIM()
    @test m.J == 1.0 && m.h == 1.0 && m.site === SiteType("S=1/2")
    m2 = TFIM(; J=0.7, h=1.3, site=SiteType("Qubit"))
    @test site_type(m2) === SiteType("Qubit")
end

@testset "build_opsum: plain S=1/2 chain, :full matches direct OpSum" begin
    sites = PhysInds(N; SiteType="S=1/2")
    m = TFIM(; J=1.0, h=0.5)
    H = MPO(build_opsum(m, sites; boundary=:full), sites)

    direct = OpSum()
    for n in 1:(N - 1)
        direct += -m.J, "Sz", n, "Sz", n + 1
    end
    for n in 1:N
        direct += -m.h, "Sx", n
    end
    H_direct = MPO(direct, sites)

    rng = MersenneTwister(0)
    ψ = random_mps(rng, sites; linkdims=4)
    @test inner(ψ', H, ψ) ≈ inner(ψ', H_direct, ψ) rtol = 1e-10

    ψup = MPS(sites, "Up")
    @test inner(ψup', H, ψup) ≈ -m.J * (N - 1) / 4 rtol = 1e-12
end

@testset "build_opsum: :bulk_half_edge differs by half-Sx boundary patches" begin
    sites = PhysInds(N; SiteType="S=1/2")
    m = TFIM(; J=1.0, h=1.0)
    H_half = MPO(build_opsum(m, sites; boundary=:bulk_half_edge), sites)
    H_full = MPO(build_opsum(m, sites; boundary=:full), sites)

    diff = OpSum()
    diff += -m.h / 2, "Sx", 1
    diff += -m.h / 2, "Sx", N
    H_diff = MPO(diff, sites)

    rng = MersenneTwister(0)
    ψ = random_mps(rng, sites; linkdims=4)
    @test (inner(ψ', H_full, ψ) - inner(ψ', H_half, ψ)) ≈
        inner(ψ', H_diff, ψ) rtol = 1e-10
end

@testset "local_ham_terms sums to build_opsum (bond decomposition)" begin
    sites = PhysInds(N; SiteType="S=1/2")
    m = TFIM(; J=1.0, h=0.5)
    for boundary in (:bulk_half_edge, :full)
        terms = local_ham_terms(m, 1:N; boundary)
        H_sum = sum(MPO(t, sites) for t in terms)
        H_full = MPO(build_opsum(m, sites; boundary), sites)
        rng = MersenneTwister(1)
        ψ = random_mps(rng, sites; linkdims=4)
        @test inner(ψ', H_sum, ψ) ≈ inner(ψ', H_full, ψ) rtol = 1e-10
    end
end

@testset "bond_term as local energy density" begin
    sites = PhysInds(N; SiteType="S=1/2")
    m = TFIM(; J=1.0, h=0.5)
    ψup = MPS(sites, "Up")
    for k in 1:(N - 1)
        e_k = inner(ψup', MPO(bond_term(m, k, k + 1), sites), ψup)
        @test e_k ≈ -m.J / 4 rtol = 1e-12
    end
end

@testset "build_opsum on env-padded chain (auto tag lookup)" begin
    phys = PhysInds(N; SiteType="S=1/2")
    envs = EnvInds(4; SiteType="Environment")
    sites = [envs[1], phys..., envs[2]]
    @test findall(i -> hastags(i, PhysSite), sites) == collect(2:(N + 1))

    m = TFIM(; J=1.0, h=0.5)
    opsum = build_opsum(m, sites)
    @test opsum isa OpSum
    @test length(opsum) > 0
end

@testset "build_opsum: non-adjacent phys_sites (purification-style layout)" begin
    Nph = 4
    all_sites = siteinds("S=1/2", 2 * Nph)
    phys_positions = collect(1:2:(2 * Nph))
    m = TFIM(; J=1.0, h=0.0)
    opsum = build_opsum(m, all_sites;
        phys_sites=phys_positions, boundary=:full)
    H = MPO(opsum, all_sites)
    ψup = MPS(all_sites, "Up")
    @test inner(ψup', H, ψup) ≈ -m.J * (Nph - 1) / 4 rtol = 1e-12
end

@testset "Qubit site emits Pauli operators" begin
    sites = siteinds("Qubit", N)
    mq = TFIM(; J=1.0, h=0.5, site=SiteType("Qubit"))
    opsum = build_opsum(mq, sites; phys_sites=1:N, boundary=:full)
    H = MPO(opsum, sites)
    ψ0 = MPS(sites, "0")
    @test inner(ψ0', H, ψ0) ≈ -mq.J * (N - 1) rtol = 1e-12
end

@testset "unknown boundary errors" begin
    sites = PhysInds(N; SiteType="S=1/2")
    @test_throws ErrorException build_opsum(TFIM(), sites; boundary=:bogus)
end
