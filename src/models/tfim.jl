"""
    TFIM(; J = 1.0, h = 1.0, convention = :spin_half)

1D transverse-field Ising model.

- `:spin_half` (default): `H = -J Σ Sᶻᵢ Sᶻᵢ₊₁ - h Σ Sˣᵢ` with
  `Sᶻ, Sˣ` the spin-1/2 operators (ITensors `"S=1/2"` site type).
- `:pauli`: `H = -J Σ σᶻᵢ σᶻᵢ₊₁ - h Σ σˣᵢ` using Pauli matrices. In
  ITensors land we still emit `"Sz"` / `"Sx"` but multiply by `4`/`2`
  respectively. This convention matches `QAtlas.TFIM`.

Mapping between conventions:
`TFIM(J, h; convention=:spin_half)` and `TFIM(J/4, h/2; convention=:pauli)`
describe the same Hamiltonian.
"""
Base.@kwdef struct TFIM <: AbstractLatticeModel
    J::Float64 = 1.0
    h::Float64 = 1.0
    convention::Symbol = :spin_half
end

site_type(::TFIM) = "S=1/2"

"""
Resolve `(zz_coef, x_coef)` such that the emitted OpSum — which uses
ITensors `"Sz"` / `"Sx"` (spin-1/2) operators — implements
`-J Σ ... - h Σ ...` in the requested convention.
"""
function _tfim_coefs(m::TFIM)
    if m.convention === :spin_half
        return (-m.J, -m.h)
    elseif m.convention === :pauli
        # σᶻ = 2 Sᶻ ⇒ σᶻσᶻ = 4 SᶻSᶻ;  σˣ = 2 Sˣ
        return (-4 * m.J, -2 * m.h)
    else
        error("TFIM: unknown convention $(m.convention). Use :spin_half or :pauli.")
    end
end

"""
    build_opsum(model::TFIM, sites; phys_sites, boundary=:bulk_half_edge) -> OpSum

Emit a TFIM OpSum acting only on `phys_sites` (default: those carrying
the `"PhysSite"` tag from ITensorSiteKit).

`boundary`:
- `:bulk_half_edge` (default): half-weight σˣ on the two bulk endpoints
  so the σˣ count matches the ZZ bond count — matches the OBC-chunk
  embedding used by REB.
- `:full`: full-weight σˣ on every physical site (plain OBC chain).

ZZ couplings are emitted only between *adjacent* physical positions
(`phys_sites[k+1] == phys_sites[k] + 1`); non-adjacent pairs are
skipped, which is the correct behaviour when env/aux sites split the
bulk into disconnected chunks.
"""
function build_opsum(
    m::TFIM,
    sites;
    phys_sites=findall(i -> hastags(i, PhysSite), sites),
    boundary::Symbol=:bulk_half_edge,
)
    zz_coef, x_coef = _tfim_coefs(m)
    phys = collect(phys_sites)
    N = length(phys)
    opsum = OpSum()
    N == 0 && return opsum

    for k in 1:(N - 1)
        n, n2 = phys[k], phys[k + 1]
        if n2 == n + 1
            opsum += zz_coef, "Sz", n, "Sz", n2
        end
    end

    if boundary === :bulk_half_edge && N >= 2
        opsum += x_coef / 2, "Sx", phys[1]
        for n in phys[2:(end - 1)]
            opsum += x_coef, "Sx", n
        end
        opsum += x_coef / 2, "Sx", phys[end]
    elseif boundary === :full || N == 1
        for n in phys
            opsum += x_coef, "Sx", n
        end
    else
        error("TFIM build_opsum: unknown boundary $boundary")
    end
    return opsum
end
