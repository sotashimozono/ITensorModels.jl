using ITensors: SiteType, @SiteType_str

"""
    PXP1D(; Ω=1.0, site=SiteType("Qubit"))

1D PXP model (Rydberg blockade):

    H = Ω Σ_i P_{i-1} X_i P_{i+1}

with `P = (I - Z)/2` the projector onto the "blockaded" state. The
Hamiltonian flips a qubit only when both neighbors are projected —
direct kinetic model for chains of Rydberg atoms in the strong-
blockade regime. Famously hosts quantum many-body scars (Bernien
et al., Nature 2017; Turner et al., Nat. Phys. 2018).

Overrides [`local_ham_terms`](@ref) because the elementary
interaction is 3-site. OBC: end sites carry the projector on the
single existing neighbor only.
"""
Base.@kwdef struct PXP1D <: AbstractLatticeModel
    Ω::Float64 = 1.0
    site::SiteType = SiteType("Qubit")
end

site_type(m::PXP1D) = m.site

bond_term(::PXP1D, ::Int, ::Int) = OpSum()
bond_coupling_term(::PXP1D, ::Int, ::Int) = OpSum()
onsite_term(::PXP1D, ::Int) = OpSum()

function onsite_observable_op(::PXP1D, name::Symbol)
    name === :x && return "X"
    name === :z && return "Z"
    name === :n && return "Proj1"
    return error("PXP1D: unsupported onsite observable $name")
end

function local_ham_terms(m::PXP1D, phys_sites; boundary::Symbol=:bulk_half_edge)
    phys = collect(phys_sites)
    N = length(phys)
    terms = OpSum[]
    Ω = m.Ω
    for k in 1:N
        H = OpSum()
        has_left = k > 1
        has_right = k < N
        if has_left && has_right
            H += Ω / 4, "X", phys[k]
            H += -Ω / 4, "Z", phys[k - 1], "X", phys[k]
            H += -Ω / 4, "X", phys[k], "Z", phys[k + 1]
            H += Ω / 4, "Z", phys[k - 1], "X", phys[k], "Z", phys[k + 1]
        elseif has_right
            H += Ω / 2, "X", phys[k]
            H += -Ω / 2, "X", phys[k], "Z", phys[k + 1]
        elseif has_left
            H += Ω / 2, "X", phys[k]
            H += -Ω / 2, "Z", phys[k - 1], "X", phys[k]
        else
            H += Ω, "X", phys[k]
        end
        push!(terms, H)
    end
    return terms
end
