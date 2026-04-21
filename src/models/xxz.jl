using ITensors: SiteType, @SiteType_str

"""
    XXZ1D(; J=1.0, Δ=1.0, site=SiteType("S=1/2"))

1D spin-½ XXZ chain in the spin-½ (Takahashi) convention

    H = J Σ [ Sˣᵢ Sˣᵢ₊₁  +  Sʸᵢ Sʸᵢ₊₁  +  Δ Sᶻᵢ Sᶻᵢ₊₁ ]

matching `QAtlas.XXZ1D`. `Δ = 1` is the isotropic Heisenberg point,
`Δ = 0` is the XX / free-fermion point.

On `SiteType("Qubit")` the same abstract Hamiltonian is emitted using
the Pauli operators `X`, `Y`, `Z` (a factor-of-4 rescale is absorbed
inside `bond_term`).
"""
Base.@kwdef struct XXZ1D <: AbstractLatticeModel
    J::Float64 = 1.0
    Δ::Float64 = 1.0
    site::SiteType = SiteType("S=1/2")
end

site_type(m::XXZ1D) = m.site

# Operator names on S=1/2 vs Qubit sites, plus the scale factor that
# turns QAtlas' spin-½ coefficients into the operator norms used here.
_xxz_ops(::SiteType"S=1/2") = ("Sx", "Sy", "Sz", 1.0)
_xxz_ops(::SiteType"Qubit") = ("X", "Y", "Z", 0.25)

function bond_term(m::XXZ1D, i::Int, j::Int)
    xop, yop, zop, scale = _xxz_ops(m.site)
    J = m.J * scale
    opsum = OpSum()
    opsum += J, xop, i, xop, j
    opsum += J, yop, i, yop, j
    opsum += J * m.Δ, zop, i, zop, j
    return opsum
end

# XXZ1D has no on-site terms → no boundary patch needed.
