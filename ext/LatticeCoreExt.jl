module LatticeCoreExt

using ITensors: OpSum, hastags
using ITensorModels
using ITensorModels: AbstractLatticeModel, LatticeModel, bond_term,
    local_ham_terms, boundary_patch
using ITensorSiteKit: PhysSite
using LatticeCore: AbstractLattice, bonds, num_sites

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

"""
    _ordering(m::LatticeModel)

Resolve the lattice-site → MPS-position map. An empty `m.ordering`
selects the natural order `1:num_sites(m.lattice)`.
"""
function _ordering(m::LatticeModel)
    isempty(m.ordering) ? collect(1:num_sites(m.lattice)) : m.ordering
end

"""
    _lookup_bond_model(m::LatticeModel, bt::Symbol)

Fetch the sub-model for `Bond.type == bt`, falling back to `:nearest`
if the specific tag is not registered. This lets a uniform-couplings
dict `Dict(:nearest => Heisenberg1D(...))` cover lattices whose bond
types are labelled differently (`:type_1`, `:type_2`, etc.) as long as
the user passes a dict keyed on the actual labels they want to match.
"""
function _lookup_bond_model(m::LatticeModel, bt::Symbol)
    haskey(m.bond_models, bt) && return m.bond_models[bt]
    haskey(m.bond_models, :nearest) && return m.bond_models[:nearest]
    return error(
        "LatticeModel: no bond_model registered for Bond.type = $bt. " *
        "Known keys: $(collect(keys(m.bond_models))).",
    )
end

# ---------------------------------------------------------------------
# local_ham_terms / build_opsum
# ---------------------------------------------------------------------

"""
    local_ham_terms(m::LatticeModel{<:AbstractLattice}, phys_sites; boundary)

Iterate the lattice's bonds; for each bond emit the corresponding
sub-model's `bond_term` on the MPS positions obtained from `_ordering`.
Single-site boundary patches are **not** emitted — 2D / graph lattices
don't have a canonical boundary in the `:bulk_half_edge` sense, so each
bond term carries the full on-site weight already (this matches the
behaviour of `Heisenberg1D` / `XXZ1D` whose on-site fields are zero).

`phys_sites` is ignored: the lattice itself selects which positions
host physical DOF via its bond connectivity. `boundary` is accepted
for interface compatibility but currently must be `:full`.
"""
function ITensorModels.local_ham_terms(
    m::LatticeModel{<:AbstractLattice}, phys_sites;
    boundary::Symbol=:full,
)
    boundary === :full || error(
        "LatticeModel currently supports only boundary = :full " *
        "(got $boundary). Open a `:bulk_half_edge` variant when a " *
        "ChainConfig that needs it lands.",
    )
    ord = _ordering(m)
    terms = OpSum[]
    for b in bonds(m.lattice)
        submodel = _lookup_bond_model(m, b.type)
        push!(terms, bond_term(submodel, ord[b.i], ord[b.j]))
    end
    return terms
end

"""
    build_opsum(m::LatticeModel{<:AbstractLattice}, sites;
                phys_sites=nothing, boundary=:full)

Compose the full `OpSum` by summing every bond term produced by
[`local_ham_terms`](@ref). `phys_sites` is ignored (the lattice graph
provides connectivity).
"""
function ITensorModels.build_opsum(
    m::LatticeModel{<:AbstractLattice}, sites;
    phys_sites=nothing, boundary::Symbol=:full,
)
    opsum = OpSum()
    for t in ITensorModels.local_ham_terms(m, phys_sites; boundary)
        opsum += t
    end
    return opsum
end

end # module LatticeCoreExt
