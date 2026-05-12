using ITensors: SiteType, @SiteType_str

"""
    S1XXZ1D(; J=1.0, Delta=1.0, site=SiteType("S=1"))

S=1 XXZ Heisenberg chain:

    H = J sum_i [Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Delta Sz_i Sz_{i+1}]

Phase diagram (Schulz 1986):
- : isotropic S=1 Heisenberg; gapped Haldane phase (SPT).
- Large : easy-axis Neel phase (Ising-like ordering).
- : pure XY chain on S=1; gapless.
- : easy-plane / ferromagnetic tendency.

The Haldane phase is robust for approximately -1 < Delta < infinity
(exact bounds depend on additional terms). Generalises S1Heisenberg1D
to tunable exchange anisotropy.
"""
Base.@kwdef struct S1XXZ1D <: AbstractLatticeModel
    J::Float64 = 1.0
    Delta::Float64 = 1.0
    site::SiteType = SiteType("S=1")
end

site_type(m::S1XXZ1D) = m.site

function bond_term(m::S1XXZ1D, i::Int, j::Int)
    H = OpSum()
    H += m.J, "Sx", i, "Sx", j
    H += m.J, "Sy", i, "Sy", j
    H += m.J * m.Delta, "Sz", i, "Sz", j
    return H
end

boundary_patch(::S1XXZ1D, ::Int) = OpSum()

function bond_coupling_term(m::S1XXZ1D, i::Int, j::Int)
    return bond_term(m, i, j)
end

onsite_term(::S1XXZ1D, ::Int) = OpSum()

function onsite_observable_op(::S1XXZ1D, name::Symbol)
    name === :sx && return "Sx"
    name === :sy && return "Sy"
    name === :sz && return "Sz"
    return error("S1XXZ1D: unsupported onsite observable $name")
end
