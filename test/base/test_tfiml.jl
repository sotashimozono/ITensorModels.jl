using ITensorModels: TFIML, build_opsum, bond_term, local_ham_terms
using ITensors, ITensorMPS
using ITensors: SiteType
using ITensorSiteKit: PhysInds
using Random

N = 6

@testset "TFIML diagonal on |↑…↑⟩ picks up ZZ and Sz but not Sx" begin
    sites = PhysInds(N; SiteType="S=1/2")
    m = TFIML(; J=1.0, h_x=0.5, h_z=0.1)
    H = MPO(build_opsum(m, sites; boundary=:full), sites)

    ψup = MPS(sites, "Up")
    # Sz=+1/2 on all sites → ZZ contributes -J(N-1)/4, -h_z contributes -h_z*N/2.
    expected = -m.J * (N - 1) / 4 - m.h_z * N / 2
    @test inner(ψup', H, ψup) ≈ expected rtol = 1e-12
end

@testset "TFIML local_ham_terms sums to build_opsum" begin
    sites = PhysInds(N; SiteType="S=1/2")
    m = TFIML(; J=1.0, h_x=0.7, h_z=0.2)
    for boundary in (:bulk_half_edge, :full)
        terms = local_ham_terms(m, 1:N; boundary)
        H_sum = sum(MPO(t, sites) for t in terms)
        H = MPO(build_opsum(m, sites; boundary), sites)
        rng = MersenneTwister(3)
        ψ = random_mps(rng, sites; linkdims=4)
        @test inner(ψ', H_sum, ψ) ≈ inner(ψ', H, ψ) rtol = 1e-10
    end
end

@testset "TFIML reduces to TFIM at h_z = 0" begin
    sites = PhysInds(N; SiteType="S=1/2")
    m_tfiml = TFIML(; J=1.0, h_x=0.5, h_z=0.0)
    m_tfim = ITensorModels.TFIM(; J=1.0, h=0.5)

    H_tfiml = MPO(build_opsum(m_tfiml, sites; boundary=:full), sites)
    H_tfim = MPO(build_opsum(m_tfim, sites; boundary=:full), sites)

    rng = MersenneTwister(7)
    ψ = random_mps(rng, sites; linkdims=4)
    @test inner(ψ', H_tfiml, ψ) ≈ inner(ψ', H_tfim, ψ) rtol = 1e-10
end
