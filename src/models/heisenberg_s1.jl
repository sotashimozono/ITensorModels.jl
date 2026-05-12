using ITensors: SiteType, @SiteType_str

"""
    S1Heisenberg1D(; J=1.0, site=SiteType("S=1"))

1D spin-1 isotropic Heisenberg chain

    H = J Σ [ Sˣᵢ Sˣᵢ₊₁ + Sʸᵢ Sʸᵢ₊₁ + Sᶻᵢ Sᶻᵢ₊₁ ]

on a three-dimensional local Hilbert space. Matches
`QAtlas.S1Heisenberg1D`.

The Haldane gap (≈ 0.41048J at the infinite-chain limit) and the
symmetry-protected-topological (SPT) ground state on open boundaries
are the key physics distinguishing this model from its spin-½
counterpart [`Heisenberg1D`](@ref).

Structurally this is a thin alias around [`XXZ1D`](@ref) at `Δ = 1`
with `site = SiteType("S=1")` — the spin-1 Sx/Sy/Sz operators on
`SiteType("S=1")` carry the same operator-norm convention as
`SiteType("S=1/2")`, so the existing `XXZ1D` machinery applies
verbatim.
"""
Base.@kwdef struct S1Heisenberg1D <: AbstractLatticeModel
    J::Float64 = 1.0
    site::SiteType = SiteType("S=1")
end

site_type(m::S1Heisenberg1D) = m.site

function bond_term(m::S1Heisenberg1D, i::Int, j::Int)
    return bond_term(XXZ1D(; J=m.J, Δ=1.0, site=m.site), i, j)
end

function onsite_observable_op(m::S1Heisenberg1D, name::Symbol)
    return onsite_observable_op(XXZ1D(; J=m.J, Δ=1.0, site=m.site), name)
end

# ---------------------------------------------------------------------
# Split protocol: delegate to XXZ1D at Δ=1 — same alias pattern as
# Heisenberg1D's split protocol.
# ---------------------------------------------------------------------

function bond_coupling_term(m::S1Heisenberg1D, i::Int, j::Int)
    return bond_coupling_term(XXZ1D(; J=m.J, Δ=1.0, site=m.site), i, j)
end

onsite_term(::S1Heisenberg1D, ::Int) = OpSum()
