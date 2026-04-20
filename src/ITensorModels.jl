module ITensorModels

using ITensors
using ITensorMPS
using ITensorSiteKit: PhysSite

export AbstractLatticeModel, site_type, build_opsum
export TFIM
export to_qatlas, from_qatlas

"""
    AbstractLatticeModel

Root type for Hamiltonian specifications. Subtypes carry the coupling
constants and convention choice of a physical model; concrete OpSum /
gate construction lives in method tables hanging off this type.
"""
abstract type AbstractLatticeModel end

"""
    site_type(model) -> String

ITensors `SiteType` tag used to generate physical indices for `model`
(e.g. `"S=1/2"`).
"""
function site_type end

"""
    build_opsum(model, sites; phys_sites, boundary) -> OpSum

Construct the Hamiltonian `OpSum` for `model` on `sites`, placing
operators only on `phys_sites`. `boundary` selects the OBC weighting
convention (`:bulk_half_edge` or `:full`).
"""
function build_opsum end

"""
    to_qatlas(model)

Translate an `ITensorModels` model to the matching `QAtlas` model,
applying the unit conversion implied by `model.site`. Implemented in
`ext/QAtlasExt.jl` when `QAtlas` is loaded.
"""
function to_qatlas end

"""
    from_qatlas(qmodel)

Inverse of [`to_qatlas`](@ref). Implemented in QAtlasExt.
"""
function from_qatlas end

include("models/tfim.jl")

end # module ITensorModels
