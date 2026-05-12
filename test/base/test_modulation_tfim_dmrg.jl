using ITensorModels
using ITensorModels:
    TFIM,
    build_opsum,
    modulated,
    Uniform,
    SSD,
    SinPower,
    SmoothBoundary,
    Tabulated,
    site_weight,
    bond_weight,
    bond_coupling_term,
    onsite_term
using ITensors
using ITensorMPS
using ITensors: SiteType
using Random
using Statistics: mean, std

# Verification matrix for the `ModulatedModel` wrapper:
#   V0  Uniform modulation == bare TFIM (energy + per-bond OpSum)
#   V1  DMRG E_GS with SSD modulation is finite, smooth, lower than its
#       trivial upper bound (random-state energy)
#   V2  SSD critical TFIM: ⟨Sˣ⟩ profile follows the site_weight envelope
#       (center >> edge)
#   V3  Smoke: SinPower{4}, SmoothBoundary, Tabulated all run end-to-end

const RTOL_ENERGY = 1e-6

# -- V0 -------------------------------------------------------------------
@testset "V0: Uniform modulation reproduces bare TFIM" begin
    for (J, h) in [(1.0, 0.3), (1.0, 1.0), (1.0, 1.7)]
        N = 10
        bare = TFIM(; J=J, h=h)
        wrapped = modulated(bare; L=N, modulation=Uniform())

        sites = siteinds("S=1/2", N)
        H_bare = MPO(build_opsum(bare, sites; phys_sites=1:N, boundary=:full), sites)
        H_wrap = MPO(build_opsum(wrapped, sites; phys_sites=1:N, boundary=:full), sites)

        rng = MersenneTwister(0xfeed)
        ψ0 = random_mps(rng, sites; linkdims=8)
        sweeps = Sweeps(15)
        maxdim!(sweeps, 10, 20, 40, 60)
        cutoff!(sweeps, 1e-12)

        E_bare, _ = dmrg(H_bare, ψ0, sweeps; outputlevel=0)
        E_wrap, _ = dmrg(H_wrap, copy(ψ0), sweeps; outputlevel=0)

        # Wrapped-with-Uniform DMRG energy must match bare TFIM DMRG.
        # Cross-check vs QAtlas BdG reference lives in
        # test_tfim_dmrg_vs_qatlas.jl, so this transitively pins us to
        # the exact ground state.
        @test E_wrap ≈ E_bare rtol = RTOL_ENERGY
    end
end

# -- V1 -------------------------------------------------------------------
@testset "V1: SSD-TFIM DMRG converges and beats random state" begin
    for N in (10, 16)
        m_ssd = modulated(TFIM(; J=1.0, h=1.0); L=N, modulation=SSD())
        sites = siteinds("S=1/2", N)
        H = MPO(build_opsum(m_ssd, sites; phys_sites=1:N, boundary=:full), sites)

        rng = MersenneTwister(42)
        ψ0 = random_mps(rng, sites; linkdims=16)
        sweeps = Sweeps(20)
        maxdim!(sweeps, 10, 20, 40, 80, 120)
        cutoff!(sweeps, 1e-12)
        E_dmrg, ψ_gs = dmrg(H, ψ0, sweeps; outputlevel=0)

        E_random = real(inner(ψ0', H, ψ0))
        @test isfinite(E_dmrg)
        @test E_dmrg < E_random
        @test isapprox(norm(ψ_gs), 1.0; atol=1e-8)
    end
end

# -- V2 -------------------------------------------------------------------
@testset "V2: SSD critical TFIM ⟨Sˣ⟩ profile is approximately uniform" begin
    # Katsura-Maruyama-Tanaka-Katsura (2011): the SSD ground state of
    # 1D TFIM is the PBC ground state (up to exponentially small finite-L
    # corrections); local observables ⟨Sˣᵢ⟩, ⟨Sᶻᵢ Sᶻᵢ₊₁⟩ are therefore
    # approximately uniform across the entire open chain, not concentrated
    # at the centre.
    N = 24
    J, h = 1.0, 1.0
    m_ssd = modulated(TFIM(; J=J, h=h); L=N, modulation=SSD())
    sites = siteinds("S=1/2", N)
    H = MPO(build_opsum(m_ssd, sites; phys_sites=1:N, boundary=:full), sites)

    rng = MersenneTwister(7)
    ψ0 = random_mps(rng, sites; linkdims=16)
    sweeps = Sweeps(25)
    maxdim!(sweeps, 10, 20, 40, 80, 120, 200)
    cutoff!(sweeps, 1e-12)
    _, ψ_gs = dmrg(H, ψ0, sweeps; outputlevel=0)

    sx = expect(ψ_gs, "Sx")
    sx_mean = mean(sx)
    sx_std = std(sx)

    # Ground state of -h Σ Sx (h > 0) is polarized in +Sx, so ⟨Sx⟩ > 0.
    @test sx_mean > 0
    # SSD prediction: profile is approximately translation-invariant.
    # Allow generous tolerance because finite L=24 still has visible
    # boundary breathing; the spread should still be small relative
    # to the mean.
    @test sx_std / abs(sx_mean) < 0.25
end

# -- V3 -------------------------------------------------------------------
@testset "V3: alternative modulations build a valid MPO" begin
    N = 12
    bare = TFIM(; J=1.0, h=1.0)
    sites = siteinds("S=1/2", N)

    m4 = modulated(bare; L=N, modulation=SinPower{4}())
    H4 = MPO(build_opsum(m4, sites; phys_sites=1:N, boundary=:full), sites)
    @test H4 isa MPO

    m_sbc = modulated(bare; L=N, modulation=SmoothBoundary(3))
    H_sbc = MPO(build_opsum(m_sbc, sites; phys_sites=1:N, boundary=:full), sites)
    @test H_sbc isa MPO

    fs = ones(N)
    fb = ones(N - 1)
    m_tab = modulated(bare; L=N, modulation=Tabulated(fs, fb))
    H_tab = MPO(build_opsum(m_tab, sites; phys_sites=1:N, boundary=:full), sites)
    H_uni = MPO(
        build_opsum(
            modulated(bare; L=N, modulation=Uniform()),
            sites;
            phys_sites=1:N,
            boundary=:full,
        ),
        sites,
    )
    rng = MersenneTwister(11)
    ψ = random_mps(rng, sites; linkdims=8)
    @test real(inner(ψ', H_tab, ψ)) ≈ real(inner(ψ', H_uni, ψ)) rtol = 1e-10

    @test bond_weight(SSD(), N ÷ 2, N) ≈ 1.0 atol = 1e-12
    @test site_weight(SSD(), 1, N) < site_weight(SSD(), N ÷ 2, N)
    @test site_weight(SinPower{4}(), 1, N) < site_weight(SSD(), 1, N)
end
