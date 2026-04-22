using ITensors: SiteType, @SiteType_str

"""
    KitaevBond(; K=1.0, axis=:z, site=SiteType("S=1/2"))

Directional Ising coupling on a single bond — the building block for
the Kitaev honeycomb model

    H_Kitaev = -K_x Σ_{(i,j) ∈ x-bonds} σ^x_i σ^x_j
              -K_y Σ_{(i,j) ∈ y-bonds} σ^y_i σ^y_j
              -K_z Σ_{(i,j) ∈ z-bonds} σ^z_i σ^z_j

`bond_term(m::KitaevBond, i, j)` emits `-K · σ^axis_i σ^axis_j` using
the local operator names from the chosen `site` SiteType (spin-½ "Sα"
vs Qubit "α"). To build the full Kitaev honeycomb Hamiltonian, feed
three `KitaevBond`s to a `LatticeModel`:

```julia
using ITensorModels, Lattice2D, LatticeCore
lat = build_lattice(Honeycomb, Lx, Ly; ...)
m = LatticeModel(; lattice=lat, bond_models=Dict(
    :type_1 => KitaevBond(; K=Kz, axis=:z),
    :type_2 => KitaevBond(; K=Kx, axis=:x),
    :type_3 => KitaevBond(; K=Ky, axis=:y)))
```

Hyperparameters

- `K::Float64 = 1.0` — coupling strength. Sign is the `-K` convention.
- `axis::Symbol` — `:x`, `:y`, or `:z`; selects which spin component
  the bond couples on.
- `site::SiteType` — local Hilbert space. Currently `S=1/2`, `Qubit`,
  `S=1`, `S=3/2` are supported via the shared `kitaev_op` dispatch
  below.
"""
Base.@kwdef struct KitaevBond <: AbstractLatticeModel
    K::Float64 = 1.0
    axis::Symbol = :z
    site::SiteType = SiteType("S=1/2")
end

site_type(m::KitaevBond) = m.site

# Per-site-type operator names for the three Kitaev axes. Extend with
# more SiteType methods to cover additional local Hilbert spaces.
kitaev_op(::SiteType"S=1/2", ::Val{:x}) = "Sx"
kitaev_op(::SiteType"S=1/2", ::Val{:y}) = "Sy"
kitaev_op(::SiteType"S=1/2", ::Val{:z}) = "Sz"
kitaev_op(::SiteType"S=1",   ::Val{:x}) = "Sx"
kitaev_op(::SiteType"S=1",   ::Val{:y}) = "Sy"
kitaev_op(::SiteType"S=1",   ::Val{:z}) = "Sz"
kitaev_op(::SiteType"S=3/2", ::Val{:x}) = "Sx"
kitaev_op(::SiteType"S=3/2", ::Val{:y}) = "Sy"
kitaev_op(::SiteType"S=3/2", ::Val{:z}) = "Sz"
kitaev_op(::SiteType"Qubit", ::Val{:x}) = "X"
kitaev_op(::SiteType"Qubit", ::Val{:y}) = "Y"
kitaev_op(::SiteType"Qubit", ::Val{:z}) = "Z"

"""
    _kitaev_op(site, axis) -> String

Look up the ITensors operator name for the given Kitaev `axis`
(`:x` / `:y` / `:z`) on the given `site`. Errors on unsupported
combinations so that dispatch misses are loud rather than silent.
"""
function _kitaev_op(site::SiteType, axis::Symbol)
    axis ∈ (:x, :y, :z) ||
        error("KitaevBond: unknown axis $axis (expected :x, :y, :z).")
    return kitaev_op(site, Val(axis))
end

function bond_term(m::KitaevBond, i::Int, j::Int)
    op = _kitaev_op(m.site, m.axis)
    opsum = OpSum()
    opsum += -m.K, op, i, op, j
    return opsum
end
