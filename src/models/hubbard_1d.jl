using ITensors: SiteType, @SiteType_str

"""
    Hubbard1D(; t=1.0, U=4.0, μ=0.0, site=SiteType("Electron"))

1D single-band Fermi-Hubbard model

    H = -t Σ_{i,σ} (c†_{i,σ} c_{i+1,σ} + h.c.)
      + U Σ_i n_{i↑} n_{i↓}
      + μ Σ_i (n_{i↑} + n_{i↓})

on `SiteType("Electron")` sites (4-dim local Hilbert space:
|0⟩, |↑⟩, |↓⟩, |↑↓⟩). Jordan-Wigner strings between non-adjacent
fermion operators are inserted automatically by ITensors' OpSum →
MPO conversion.

Strong-coupling limit `U/t → ∞` at half-filling reduces to the
Heisenberg spin chain (super-exchange J = 4t²/U); weak-coupling
limit is two decoupled free-fermion chains. The 1D Hubbard model is
exactly solvable by Bethe ansatz (Lieb–Wu 1968).
"""
Base.@kwdef struct Hubbard1D <: AbstractLatticeModel
    t::Float64 = 1.0
    U::Float64 = 4.0
    μ::Float64 = 0.0
    site::SiteType = SiteType("Electron")
end

site_type(m::Hubbard1D) = m.site

function bond_term(m::Hubbard1D, i::Int, j::Int)
    H = OpSum()
    H += -m.t, "Cdagup", i, "Cup", j
    H += -m.t, "Cdagup", j, "Cup", i
    H += -m.t, "Cdagdn", i, "Cdn", j
    H += -m.t, "Cdagdn", j, "Cdn", i
    for k in (i, j)
        H += (m.U / 2), "Nupdn", k
        H += (m.μ / 2), "Nup", k
        H += (m.μ / 2), "Ndn", k
    end
    return H
end

function boundary_patch(m::Hubbard1D, k::Int)
    H = OpSum()
    H += (m.U / 2), "Nupdn", k
    H += (m.μ / 2), "Nup", k
    H += (m.μ / 2), "Ndn", k
    return H
end

function onsite_observable_op(m::Hubbard1D, name::Symbol)
    name === :nup && return "Nup"
    name === :ndn && return "Ndn"
    name === :n && return "Ntot"
    name === :nupdn && return "Nupdn"
    name === :sz && return "Sz"
    return error("Hubbard1D: unsupported onsite observable $name")
end

function bond_coupling_term(m::Hubbard1D, i::Int, j::Int)
    H = OpSum()
    H += -m.t, "Cdagup", i, "Cup", j
    H += -m.t, "Cdagup", j, "Cup", i
    H += -m.t, "Cdagdn", i, "Cdn", j
    H += -m.t, "Cdagdn", j, "Cdn", i
    return H
end

function onsite_term(m::Hubbard1D, k::Int)
    H = OpSum()
    H += m.U, "Nupdn", k
    H += m.μ, "Nup", k
    H += m.μ, "Ndn", k
    return H
end
