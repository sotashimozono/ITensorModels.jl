using ITensors: SiteType, @SiteType_str

"""
    DMIHeisenberg1D(; J=1.0, D=(0.0, 0.0, 1.0), site=SiteType("S=1/2"))

1D spin-½ Heisenberg chain with Dzyaloshinskii-Moriya interaction:

    H = J Σ_i Sᵢ · Sᵢ₊₁ + Σ_i 𝐃 · (Sᵢ × Sᵢ₊₁)

where `D = (Dˣ, Dʸ, Dᶻ)` is the DMI vector (uniform along the chain).
Underlying physics: chiral magnetism, spin-spiral ground states,
skyrmion lattices. Leading symmetry-allowed bond perturbation for
systems with broken inversion symmetry (Dzyaloshinskii 1958;
Moriya 1960).
"""
Base.@kwdef struct DMIHeisenberg1D <: AbstractLatticeModel
    J::Float64 = 1.0
    D::NTuple{3,Float64} = (0.0, 0.0, 1.0)
    site::SiteType = SiteType("S=1/2")
end

site_type(m::DMIHeisenberg1D) = m.site

function _dmi_heisenberg_bond(m::DMIHeisenberg1D, i::Int, j::Int)
    Dx, Dy, Dz = m.D
    H = OpSum()
    H += m.J, "Sx", i, "Sx", j
    H += m.J, "Sy", i, "Sy", j
    H += m.J, "Sz", i, "Sz", j
    H += Dx, "Sy", i, "Sz", j
    H += -Dx, "Sz", i, "Sy", j
    H += Dy, "Sz", i, "Sx", j
    H += -Dy, "Sx", i, "Sz", j
    H += Dz, "Sx", i, "Sy", j
    H += -Dz, "Sy", i, "Sx", j
    return H
end

bond_term(m::DMIHeisenberg1D, i::Int, j::Int) = _dmi_heisenberg_bond(m, i, j)
bond_coupling_term(m::DMIHeisenberg1D, i::Int, j::Int) = _dmi_heisenberg_bond(m, i, j)
onsite_term(::DMIHeisenberg1D, ::Int) = OpSum()

function onsite_observable_op(::DMIHeisenberg1D, name::Symbol)
    name === :sx && return "Sx"
    name === :sy && return "Sy"
    name === :sz && return "Sz"
    return error("DMIHeisenberg1D: unsupported onsite observable $name")
end
