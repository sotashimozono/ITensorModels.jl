using ITensorModels
using ITensorModels: TFIM, TFIML, XXZ1D, Heisenberg1D,
    build_opsum, modulated, Uniform, SSD,
    bond_coupling_term, onsite_term
using ITensors
using ITensorMPS
using Random
using Statistics: mean, std

# Multi-model verification of the split-protocol + ModulatedModel pipeline.
# Per model:
#   V0  modulated(M; Uniform()) DMRG E_GS == bare M DMRG E_GS
#   V1  modulated(M; SSD()) DMRG converges
#   V2  (gapless models only) SSD ⟨Sᶻᵢ Sᶻᵢ₊₁⟩ profile is approximately
#       uniform — the original Katsura-Maruyama-Tanaka-Katsura claim
#       extended to XXZ / Heisenberg.

const RTOL_ENERGY = 1e-6

function _dmrg_gs(H, sites; seed::Int=42, sweeps_n::Int=20, maxd=200, init_dim=16)
    rng = MersenneTwister(seed)
    ψ0 = random_mps(rng, sites; linkdims=init_dim)
    sweeps = Sweeps(sweeps_n)
    maxdim!(sweeps, 10, 20, 40, 80, 120, maxd)
    cutoff!(sweeps, 1e-12)
    return dmrg(H, ψ0, sweeps; outputlevel=0)
end

function _bond_zz(ψ, sites, N)
    return [
        let
            os = OpSum()
            os += 1.0, "Sz", i, "Sz", i + 1
            real(inner(ψ', MPO(os, sites), ψ))
        end for i in 1:(N - 1)
    ]
end

# ============== TFIML ===================================================
@testset "Modulated TFIML" begin
    N = 10
    bare = TFIML(; J=1.0, h_x=1.0, h_z=0.05)
    sites = siteinds("S=1/2", N)

    H_bare = MPO(build_opsum(bare, sites; phys_sites=1:N, boundary=:full), sites)
    H_uni  = MPO(
        build_opsum(modulated(bare; L=N, modulation=Uniform()),
                    sites; phys_sites=1:N, boundary=:full),
        sites,
    )
    E_bare, _ = _dmrg_gs(H_bare, sites)
    E_uni, _  = _dmrg_gs(H_uni, sites)
    @test E_uni ≈ E_bare rtol = RTOL_ENERGY

    H_ssd = MPO(
        build_opsum(modulated(bare; L=N, modulation=SSD()),
                    sites; phys_sites=1:N, boundary=:full),
        sites,
    )
    E_ssd, ψ_ssd = _dmrg_gs(H_ssd, sites)
    @test isfinite(E_ssd)
    @test isapprox(norm(ψ_ssd), 1.0; atol=1e-8)
end

# ============== XXZ1D ===================================================
@testset "Modulated XXZ1D" begin
    @test length(onsite_term(XXZ1D(), 3)) == 0
    @test length(bond_coupling_term(XXZ1D(; J=1.0, Δ=0.5), 3, 4)) == 3

    for Δ in (0.0, 0.5, 1.0, 1.5)
        N = 10
        bare = XXZ1D(; J=1.0, Δ=Δ)
        sites = siteinds("S=1/2", N)
        H_bare = MPO(build_opsum(bare, sites; phys_sites=1:N, boundary=:full), sites)
        H_uni  = MPO(
            build_opsum(modulated(bare; L=N, modulation=Uniform()),
                        sites; phys_sites=1:N, boundary=:full),
            sites,
        )
        E_bare, _ = _dmrg_gs(H_bare, sites; seed=Int(round(100 * (1 + Δ))))
        E_uni, _  = _dmrg_gs(H_uni, sites; seed=Int(round(100 * (1 + Δ))))
        @test E_uni ≈ E_bare rtol = RTOL_ENERGY
    end

    # Gapless XXZ (Δ=0.5) under SSD: bulk ⟨Sᶻᵢ Sᶻᵢ₊₁⟩ should be near
    # uniform (CFT vacuum extended to OBC chain).
    N = 32
    bare = XXZ1D(; J=1.0, Δ=0.5)
    sites = siteinds("S=1/2", N)
    H_ssd = MPO(
        build_opsum(modulated(bare; L=N, modulation=SSD()),
                    sites; phys_sites=1:N, boundary=:full),
        sites,
    )
    _, ψ_ssd = _dmrg_gs(H_ssd, sites; sweeps_n=30, maxd=300)
    zz = _bond_zz(ψ_ssd, sites, N)
    central = zz[6:(N - 6)]
    @test std(central) / abs(mean(central)) < 0.1
end

# ============== Heisenberg1D ============================================
@testset "Modulated Heisenberg1D" begin
    @test length(onsite_term(Heisenberg1D(), 3)) == 0

    N = 10
    bare = Heisenberg1D(; J=1.0)
    sites = siteinds("S=1/2", N)

    H_bare = MPO(build_opsum(bare, sites; phys_sites=1:N, boundary=:full), sites)
    H_uni  = MPO(
        build_opsum(modulated(bare; L=N, modulation=Uniform()),
                    sites; phys_sites=1:N, boundary=:full),
        sites,
    )
    E_bare, _ = _dmrg_gs(H_bare, sites)
    E_uni, _  = _dmrg_gs(H_uni, sites)
    @test E_uni ≈ E_bare rtol = RTOL_ENERGY

    # Heisenberg AF is gapless → SSD GS bulk ≈ PBC GS bulk.
    N2 = 32
    sites2 = siteinds("S=1/2", N2)
    H_ssd = MPO(
        build_opsum(modulated(Heisenberg1D(; J=1.0); L=N2, modulation=SSD()),
                    sites2; phys_sites=1:N2, boundary=:full),
        sites2,
    )
    _, ψ_ssd = _dmrg_gs(H_ssd, sites2; sweeps_n=30, maxd=300)
    zz = _bond_zz(ψ_ssd, sites2, N2)
    central = zz[6:(N2 - 6)]
    @test mean(central) < 0
    @test std(central) / abs(mean(central)) < 0.1
end
