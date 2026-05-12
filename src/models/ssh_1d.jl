using ITensors: SiteType, @SiteType_str

"""
    SSH1D(; t1=0.8, t2=1.0, site=SiteType("Fermion"))

Su-Schrieffer-Heeger (SSH) chain for spinless fermions:

    H = -t1 sum_{i odd}  (c+_i c_{i+1} + h.c.)
        -t2 sum_{i even} (c+_i c_{i+1} + h.c.)

Paradigmatic 1D topological insulator (symmetry class BDI).
- Topological phase: |t2| > |t1| -- zero-energy edge modes at OBC.
- Trivial phase:     |t2| < |t1| -- fully gapped, no edge modes.
- Critical point:    |t2| = |t1| -- gapless (massless Dirac fermion).

Overrides local_ham_terms because the hopping amplitude alternates
with bond parity (odd bonds carry t1, even bonds carry t2).
"""
Base.@kwdef struct SSH1D <: AbstractLatticeModel
    t1::Float64 = 0.8
    t2::Float64 = 1.0
    site::SiteType = SiteType("Fermion")
end

site_type(m::SSH1D) = m.site

bond_term(::SSH1D, ::Int, ::Int) = OpSum()
bond_coupling_term(::SSH1D, ::Int, ::Int) = OpSum()
onsite_term(::SSH1D, ::Int) = OpSum()

function onsite_observable_op(::SSH1D, name::Symbol)
    name === :n && return "N"
    return error("SSH1D: unsupported onsite observable $name")
end

function local_ham_terms(m::SSH1D, phys_sites; boundary::Symbol=:bulk_half_edge)
    phys = collect(phys_sites)
    N = length(phys)
    terms = OpSum[]
    for i in 1:(N - 1)
        H = OpSum()
        t = isodd(i) ? m.t1 : m.t2
        H += -t, "Cdag", phys[i], "C", phys[i + 1]
        H += -t, "Cdag", phys[i + 1], "C", phys[i]
        push!(terms, H)
    end
    return terms
end
