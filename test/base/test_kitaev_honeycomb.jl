using ITensorModels: KitaevBond, LatticeModel, build_opsum, local_ham_terms
using ITensors, ITensorMPS
using ITensors: SiteType
using LatticeCore: num_sites, bonds
using Lattice2D: Honeycomb as L2DHoneycomb, build_lattice
using Lattice2D: PeriodicAxis, OpenAxis, LatticeBoundary, UniformLayout, IsingSite
using QAtlas: QAtlas, Energy, OBC
using Random: MersenneTwister

# End-to-end integration: LatticeModel + KitaevBond + Lattice2D.Honeycomb
# constructs the Kitaev honeycomb Hamiltonian; ITensorMPS DMRG then
# yields a variational ground state that must converge to the exact
# value supplied by QAtlas.KitaevHoneycomb (whose OBC branch is
# validated against ED in the QAtlas test suite). Passing this means
# the LatticeCoreExt bond iteration + KitaevBond axis-dispatch are
# physically correct.

function _make_kitaev_honeycomb(Lx::Int, Ly::Int; K=1.0)
    lat = build_lattice(L2DHoneycomb, Lx, Ly;
        boundary=OpenAxis(), layout=UniformLayout(IsingSite()))
    model = LatticeModel(; lattice=lat,
        bond_models=Dict(
            :type_1 => KitaevBond(; K=K, axis=:z, site=SiteType("Qubit")),
            :type_2 => KitaevBond(; K=K, axis=:x, site=SiteType("Qubit")),
            :type_3 => KitaevBond(; K=K, axis=:y, site=SiteType("Qubit")),
        ))
    return lat, model
end

@testset "LatticeModel + KitaevBond on Honeycomb: bond book-keeping" begin
    Lx, Ly = 2, 2
    lat, model = _make_kitaev_honeycomb(Lx, Ly)

    n_bond = length(collect(bonds(lat)))
    n_term = length(local_ham_terms(model, 1:num_sites(lat); boundary=:full))
    @test n_term == n_bond      # one OpSum per lattice bond
    @test n_term > 0
end

@testset "Kitaev honeycomb DMRG matches QAtlas.KitaevHoneycomb OBC (isotropic)" begin
    # DMRG on the 8-site 2×2 OBC cluster. QAtlas's `fetch(KitaevHoneycomb,
    # Energy(), OBC; Lx, Ly)` returns per-site GS energy from the
    # flux-free-sector SVD — verified against dense ED in the QAtlas
    # test suite. At Lx = Ly = 2 the reference is ε₀ ≈ -0.5832336.
    Lx, Ly = 2, 2
    lat, model = _make_kitaev_honeycomb(Lx, Ly; K=1.0)
    N = num_sites(lat)
    sites = siteinds("Qubit", N)
    H = MPO(build_opsum(model, sites), sites)

    rng = MersenneTwister(42)
    ψ0 = random_mps(rng, sites; linkdims=16)
    sweeps = Sweeps(20)
    maxdim!(sweeps, 10, 20, 40, 80, 120)
    cutoff!(sweeps, 1e-12)
    E_dmrg, _ = dmrg(H, ψ0, sweeps; outputlevel=0)

    m_qatlas = QAtlas.KitaevHoneycomb(; Kx=1.0, Ky=1.0, Kz=1.0)
    ε_ref = QAtlas.fetch(m_qatlas, Energy(), OBC(0); Lx=Lx, Ly=Ly)
    E_ref = ε_ref * N   # per-site → total

    @info "Kitaev DMRG vs QAtlas (Lx=Ly=2 OBC isotropic)" N E_dmrg E_ref ε_dmrg =
        E_dmrg / N ε_ref rel_err = abs(E_dmrg - E_ref) / abs(E_ref)
    @test E_dmrg ≈ E_ref rtol = 1e-4
end

@testset "Kitaev honeycomb DMRG matches QAtlas anisotropic OBC" begin
    # Anisotropic points — check the KitaevBond axis dispatch is wired
    # to the right σˣ / σʸ / σᶻ pauli ops. Ax-phase pushes Kx dominant.
    Lx, Ly = 2, 2
    cases = [
        (1.0, 1.0, 1.0),     # B-phase (isotropic, covered above but
        #                                include as sanity)
        (0.3, 0.7, 1.0),     # generic anisotropic
        (2.0, 0.5, 0.5),     # Ax-phase (gapped)
    ]
    for (Kx, Ky, Kz) in cases
        lat = build_lattice(L2DHoneycomb, Lx, Ly;
            boundary=OpenAxis(), layout=UniformLayout(IsingSite()))
        model = LatticeModel(; lattice=lat,
            bond_models=Dict(
                :type_1 => KitaevBond(; K=Kz, axis=:z, site=SiteType("Qubit")),
                :type_2 => KitaevBond(; K=Kx, axis=:x, site=SiteType("Qubit")),
                :type_3 => KitaevBond(; K=Ky, axis=:y, site=SiteType("Qubit")),
            ))
        N = num_sites(lat)
        sites = siteinds("Qubit", N)
        H = MPO(build_opsum(model, sites), sites)

        rng = MersenneTwister(hash((Kx, Ky, Kz)) & 0xffff)
        ψ0 = random_mps(rng, sites; linkdims=16)
        sweeps = Sweeps(25)
        maxdim!(sweeps, 10, 20, 40, 80, 120)
        cutoff!(sweeps, 1e-12)
        E_dmrg, _ = dmrg(H, ψ0, sweeps; outputlevel=0)

        m_qatlas = QAtlas.KitaevHoneycomb(; Kx=Kx, Ky=Ky, Kz=Kz)
        E_ref = QAtlas.fetch(m_qatlas, Energy(), OBC(0); Lx=Lx, Ly=Ly) * N
        @info "Kitaev DMRG vs QAtlas (anisotropic)" Kx Ky Kz E_dmrg E_ref rel_err =
            abs(E_dmrg - E_ref) / abs(E_ref)
        @test E_dmrg ≈ E_ref rtol = 1e-3
    end
end
