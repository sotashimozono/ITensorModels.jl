using ITensorModels
using ITensorModels: TFIM, build_opsum, site_type
using ITensors, ITensorMPS
using ITensors: SiteType
using ITensorSiteKit: PhysInds, EnvInds, PhysSite
using Random

N = 6

@testset "TFIM struct" begin
    m = TFIM()
    @test m.J == 1.0
    @test m.h == 1.0
    @test m.site === SiteType("S=1/2")

    m2 = TFIM(; J=0.7, h=1.3, site=SiteType("Qubit"))
    @test m2.site === SiteType("Qubit")
    @test site_type(m2) === SiteType("Qubit")
end

@testset "build_opsum: plain S=1/2 chain, :full boundary" begin
    sites = PhysInds(N; SiteType="S=1/2")
    m = TFIM(; J=1.0, h=0.5)
    opsum = build_opsum(m, sites; boundary=:full)
    H = MPO(opsum, sites)

    # H |↑…↑⟩ diagonal: -J Σ SzSz with Sz=+1/2 → -J (N-1)/4.
    ψup = MPS(sites, "Up")
    @test inner(ψup', H, ψup) ≈ -1.0 * (N - 1) / 4 rtol = 1e-12

    # (N-1) ZZ + N Sx terms
    @test length(opsum) == (N - 1) + N
end

@testset "build_opsum: bulk-half-edge boundary on S=1/2" begin
    sites = PhysInds(N; SiteType="S=1/2")
    m = TFIM(; J=1.0, h=1.0)
    opsum_half = build_opsum(m, sites; boundary=:bulk_half_edge)
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

@testset "build_opsum: env-padded chain (auto tag lookup)" begin
    phys = PhysInds(N; SiteType="S=1/2")
    envs = EnvInds(4; SiteType="Environment")
    sites = [envs[1], phys..., envs[2]]

    bulk_positions = findall(i -> hastags(i, PhysSite), sites)
    @test bulk_positions == collect(2:(N + 1))

    m = TFIM(; J=1.0, h=0.5)
    opsum = build_opsum(m, sites)
    @test length(opsum) == (N - 1) + N
end

@testset "build_opsum: non-adjacent phys_sites (purification-style layout)" begin
    # 2N-site chain: odd = phys, even = ancilla.
    Nph = 4
    all_sites = siteinds("S=1/2", 2 * Nph)
    phys_positions = collect(1:2:(2 * Nph))         # [1, 3, 5, 7]
    m = TFIM(; J=1.0, h=0.0)
    opsum = build_opsum(m, all_sites; phys_sites=phys_positions, boundary=:full)
    # Nph-1 ZZ bonds between consecutive entries of phys_positions,
    # Nph Sx terms (boundary=:full). h = 0 → Sx coeffs vanish, so
    # only ZZ terms survive in OpSum length.
    @test length(opsum) == (Nph - 1) + Nph

    # Spot-check: MPO on |↑↑…↑⟩ gives -J (Nph - 1) / 4 (only ZZ, all Sz=+1/2).
    H = MPO(opsum, all_sites)
    ψup = MPS(all_sites, "Up")
    @test inner(ψup', H, ψup) ≈ -1.0 * (Nph - 1) / 4 rtol = 1e-12
end

@testset "build_opsum: Qubit site uses Pauli operators" begin
    sites = siteinds("Qubit", N)
    mq = TFIM(; J=1.0, h=0.5, site=SiteType("Qubit"))
    # Same H form; on |0…0⟩ we have Z=+1, so -J (N-1) (not /4).
    opsum = build_opsum(mq, sites; phys_sites=collect(1:N), boundary=:full)
    H = MPO(opsum, sites)
    ψ0 = MPS(sites, "0")
    @test inner(ψ0', H, ψ0) ≈ -1.0 * (N - 1) rtol = 1e-12
end

@testset "unknown boundary errors" begin
    sites = PhysInds(N; SiteType="S=1/2")
    @test_throws ErrorException build_opsum(TFIM(), sites; boundary=:bogus)
end
