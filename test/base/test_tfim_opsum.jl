using ITensorModels
using ITensorModels: TFIM, build_opsum
using ITensors, ITensorMPS
using ITensorSiteKit: PhysInds, EnvInds, PhysSite
using Random

N = 6

@testset "TFIM struct" begin
    m = TFIM()
    @test m.J == 1.0
    @test m.h == 1.0
    @test m.convention === :spin_half

    m2 = TFIM(; J=0.7, h=1.3, convention=:pauli)
    @test m2.J == 0.7
    @test m2.h == 1.3
    @test m2.convention === :pauli

    @test ITensorModels.site_type(m) == "S=1/2"
end

@testset "build_opsum: plain chain, :full boundary" begin
    sites = PhysInds(N; SiteType="S=1/2")
    m = TFIM(; J=1.0, h=0.5)
    opsum = build_opsum(m, sites; boundary=:full)
    H = MPO(opsum, sites)

    # H |↑…↑⟩ diagonal: -J Σ SzSz with Sz=+1/2 → -J (N-1)/4.
    ψup = MPS(sites, "Up")
    @test inner(ψup', H, ψup) ≈ -1.0 * (N - 1) / 4 rtol = 1e-12

    # Number of emitted terms: (N-1) ZZ + N Sx
    @test length(opsum) == (N - 1) + N
end

@testset "build_opsum: bulk-half-edge boundary" begin
    sites = PhysInds(N; SiteType="S=1/2")
    m = TFIM(; J=1.0, h=1.0)
    opsum_half = build_opsum(m, sites; boundary=:bulk_half_edge)
    # (N-1) ZZ + (N-2) interior Sx + 2 half-edge Sx = (N-1) + N
    @test length(opsum_half) == (N - 1) + N

    H_full = MPO(build_opsum(m, sites; boundary=:full), sites)
    H_half = MPO(opsum_half, sites)

    diff_expected = OpSum()
    diff_expected += -0.5, "Sx", 1
    diff_expected += -0.5, "Sx", N
    H_diff = MPO(diff_expected, sites)

    rng = MersenneTwister(0)
    ψ = random_mps(rng, sites; linkdims=4)
    @test (inner(ψ', H_full, ψ) - inner(ψ', H_half, ψ)) ≈ inner(ψ', H_diff, ψ) rtol = 1e-10
end

@testset "build_opsum: phys sites with env padding (auto tag lookup)" begin
    phys = PhysInds(N; SiteType="S=1/2")
    envs = EnvInds(4; SiteType="Environment")
    sites = [envs[1], phys..., envs[2]]

    bulk_positions = findall(i -> hastags(i, PhysSite), sites)
    @test bulk_positions == collect(2:(N + 1))

    m = TFIM(; J=1.0, h=0.5)
    opsum = build_opsum(m, sites)
    # Same term count as plain chain with bulk_half_edge default.
    @test length(opsum) == (N - 1) + N
end

@testset "convention equivalence :spin_half ↔ :pauli" begin
    sites = PhysInds(N; SiteType="S=1/2")
    m_s = TFIM(; J=1.0, h=0.5, convention=:spin_half)
    m_p = TFIM(; J=0.25, h=0.25, convention=:pauli)  # same H
    Hs = MPO(build_opsum(m_s, sites; boundary=:full), sites)
    Hp = MPO(build_opsum(m_p, sites; boundary=:full), sites)

    rng = MersenneTwister(1)
    ψ = random_mps(rng, sites; linkdims=4)
    @test inner(ψ', Hs, ψ) ≈ inner(ψ', Hp, ψ) rtol = 1e-12
end

@testset "unknown convention errors" begin
    @test_throws ErrorException build_opsum(
        TFIM(; convention=:bogus), PhysInds(N); boundary=:full
    )
end
