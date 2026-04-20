using ITensorModels: TFIM, build_opsum
using ITensors, ITensorMPS
using ITensors: SiteType
import QAtlas
using QAtlas: Energy, OBC, Infinite
using Random

# End-to-end validation: build_opsum must produce the actual TFIM
# Hamiltonian.  We run DMRG on the MPO and compare the converged ground
# state energy to the exact BdG result from QAtlas.

@testset "S=1/2 DMRG ground state matches QAtlas.OBC" begin
    for (J, h) in [(1.0, 0.3), (1.0, 0.7), (1.0, 1.5)]
        N = 10
        m = TFIM(; J=J, h=h)                   # S=1/2 site
        sites = siteinds("S=1/2", N)
        H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)

        rng = MersenneTwister(42)
        ψ0 = random_mps(rng, sites; linkdims=16)

        sweeps = Sweeps(20)
        maxdim!(sweeps, 10, 20, 40, 60, 80)
        cutoff!(sweeps, 1e-12)

        E_dmrg, _ = dmrg(H, ψ0, sweeps; outputlevel=0)
        E_qatlas = QAtlas.fetch(m, Energy(), OBC(N))   # ground state (no beta)

        @test E_dmrg ≈ E_qatlas rtol = 1e-6
    end
end

@testset "Qubit DMRG ground state matches QAtlas (Pauli units)" begin
    J, h = 1.0, 0.5
    N = 10
    m = TFIM(; J=J, h=h, site=SiteType("Qubit"))
    sites = siteinds("Qubit", N)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)

    rng = MersenneTwister(0)
    ψ0 = random_mps(rng, sites; linkdims=16)

    sweeps = Sweeps(20)
    maxdim!(sweeps, 10, 20, 40, 60, 80)
    cutoff!(sweeps, 1e-12)
    E_dmrg, _ = dmrg(H, ψ0, sweeps; outputlevel=0)

    E_qatlas = QAtlas.fetch(m, Energy(), OBC(N))   # forwarder routes via to_qatlas
    @test E_dmrg ≈ E_qatlas rtol = 1e-6
end
