module ITensorModels

using ITensors
using ITensorMPS
using ITensorSiteKit: PhysSite

export AbstractLatticeModel, site_type
export bond_term, boundary_patch, local_ham_terms, build_opsum
export TFIM, TFIML, XXZ1D, Heisenberg1D, KitaevBond, LatticeModel
export to_qatlas, from_qatlas

"""
    AbstractLatticeModel

Root type for Hamiltonian specifications. Concrete subtypes implement
[`bond_term`](@ref) (and optionally [`boundary_patch`](@ref)); the
generic [`local_ham_terms`](@ref) / [`build_opsum`](@ref) machinery
lifts those into a full MPO OpSum for any site layout.
"""
abstract type AbstractLatticeModel end

"""
    site_type(model) -> ITensors.SiteType

ITensors `SiteType` used when building physical indices for `model`.
"""
function site_type end

"""
    to_qatlas(model)

Translate an `ITensorModels` model to the matching `QAtlas` model,
applying the unit conversion implied by `model.site`. Implemented in
`ext/QAtlasExt.jl` when `QAtlas` is loaded.
"""
function to_qatlas end

"""
    from_qatlas(qmodel)

Inverse of [`to_qatlas`](@ref). Implemented in `ext/QAtlasExt.jl`.
"""
function from_qatlas end

include("core/interface.jl")

include("models/tfim.jl")
include("models/tfiml.jl")
include("models/xxz.jl")
include("models/heisenberg.jl")
include("models/kitaev_bond.jl")
include("models/lattice_model.jl")

end # module ITensorModels
