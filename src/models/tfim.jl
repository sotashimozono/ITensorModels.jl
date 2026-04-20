using ITensors: SiteType, @SiteType_str

"""
    TFIM(; J = 1.0, h = 1.0, site = SiteType("S=1/2"))

1D transverse-field Ising model

    H = -J Σᵢ Zᵢ Zᵢ₊₁ - h Σᵢ Xᵢ

where `Z`, `X` are the natural local operators of `site`:

- `SiteType("S=1/2")` → `Z = Sᶻ`, `X = Sˣ` (spin-½, ITensors `"Sz"`/`"Sx"`)
- `SiteType("Qubit")` → `Z = σᶻ`, `X = σˣ` (Pauli, ITensors `"Z"`/`"X"`)

Callers choose the `SiteType` to match the physical units they want
`J` / `h` to carry; no separate convention flag is needed. Conversion to
QAtlas (which is Pauli-based) happens in `ext/QAtlasExt.jl` via
`to_qatlas`, which dispatches on the site type.
"""
Base.@kwdef struct TFIM <: AbstractLatticeModel
    J::Float64 = 1.0
    h::Float64 = 1.0
    site::SiteType = SiteType("S=1/2")
end

site_type(m::TFIM) = m.site

# ---------------------------------------------------------------------
# Per-site operator names (SiteType dispatch).
# Add more SiteType methods to extend TFIM coverage.
# ---------------------------------------------------------------------

ising_z_op(::SiteType"S=1/2") = "Sz"
ising_x_op(::SiteType"S=1/2") = "Sx"
ising_z_op(::SiteType"Qubit") = "Z"
ising_x_op(::SiteType"Qubit") = "X"

"""
    build_opsum(model::TFIM, sites; phys_sites, boundary=:bulk_half_edge) -> OpSum

Emit the TFIM Hamiltonian as an `OpSum` on `sites`, placing operators on
the positions listed in `phys_sites`.

- `phys_sites` (default: positions carrying the `PhysSite` tag) is the
  ordered list of chain positions that host a physical spin. ZZ
  couplings are emitted between **every pair of consecutive entries in
  `phys_sites`** regardless of chain gap — this lets a purification-style
  `[phys, anc, phys, anc, …]` layout pass `phys_sites = 1:2:N` to couple
  `(1,3), (3,5), …`, and a plain chain pass `phys_sites = 1:N` to couple
  every neighbour, and an env-split chain use the default tag-based lookup.
- `boundary = :bulk_half_edge` (default): half-weight `h/2 Σ X` on the
  two boundary phys sites so the X count matches the ZZ bond count
  (matches the REB Env-embedded bulk convention).
- `boundary = :full`: full-weight `h Σ X` on every phys site.

Operator names (`Sz`/`Sx` vs `Z`/`X`) are selected by
[`ising_z_op`](@ref) / [`ising_x_op`](@ref) dispatching on
`model.site`. Register new `SiteType` methods there to support
additional local Hilbert spaces.
"""
function build_opsum(
    m::TFIM,
    sites;
    phys_sites=findall(i -> hastags(i, PhysSite), sites),
    boundary::Symbol=:bulk_half_edge,
)
    zop = ising_z_op(m.site)
    xop = ising_x_op(m.site)
    phys = collect(phys_sites)
    N = length(phys)
    opsum = OpSum()
    N == 0 && return opsum

    for k in 1:(N - 1)
        opsum += -m.J, zop, phys[k], zop, phys[k + 1]
    end

    if boundary === :bulk_half_edge && N >= 2
        opsum += -m.h / 2, xop, phys[1]
        for n in phys[2:(end - 1)]
            opsum += -m.h, xop, n
        end
        opsum += -m.h / 2, xop, phys[end]
    elseif boundary === :full || N == 1
        for n in phys
            opsum += -m.h, xop, n
        end
    else
        error("TFIM build_opsum: unknown boundary $boundary")
    end
    return opsum
end
