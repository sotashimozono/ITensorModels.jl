using ITensorModels: KitaevBond, LatticeModel, build_opsum, local_ham_terms
using ITensors, ITensorMPS
using ITensors: SiteType
using LatticeCore: num_sites, bonds
using Lattice2D: Honeycomb, build_lattice
using Lattice2D: PeriodicAxis, OpenAxis, LatticeBoundary, UniformLayout,
    IsingSite

# Smoke test: does LatticeModel + KitaevBond produce a sensible
# Hamiltonian on Lattice2D's Honeycomb? We check the three-axis Kitaev
# Hamiltonian has the expected bond count and yields a negative ground
# state energy via DMRG at the isotropic point K_x = K_y = K_z = 1.

function _make_kitaev_honeycomb(Lx::Int, Ly::Int; K=1.0)
    lat = build_lattice(Honeycomb, Lx, Ly;
        boundary=OpenAxis(),
        layout=UniformLayout(IsingSite()))
    model = LatticeModel(;
        lattice=lat,
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

@testset "KitaevHoneycomb isotropic DMRG ground energy is negative" begin
    # Tiny Honeycomb for CI — 2×2 cells = 8 sites.
    Lx, Ly = 2, 2
    lat, model = _make_kitaev_honeycomb(Lx, Ly; K=1.0)
    N = num_sites(lat)

    sites = siteinds("Qubit", N)
    H = MPO(build_opsum(model, sites), sites)

    ψ0 = random_mps(sites; linkdims=8)
    sweeps = Sweeps(12)
    maxdim!(sweeps, 10, 20, 40, 60)
    cutoff!(sweeps, 1e-10)
    E, _ = dmrg(H, ψ0, sweeps; outputlevel=0)

    @info "Kitaev honeycomb DMRG" Lx Ly N K = 1.0 E E_per_site = E / N
    # Isotropic Kitaev gs energy per site in the TL is roughly -0.39 (in
    # units where K = 1). Small OBC systems deviate, but per-site energy
    # should clearly be in the range (-0.6, -0.1) — i.e. negative and
    # finite.
    @test E / N < -0.1
    @test E / N > -0.6
end
