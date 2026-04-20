module ITensorModels

using ITensors
using ITensorMPS
using ITensorSiteKit: PhysSite

export AbstractLatticeModel, site_type, build_opsum
export TFIM
export to_qatlas, from_qatlas
export thermal_energy, thermal_magnetization_x, thermal_specific_heat

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

# ---------------------------------------------------------------------
# QAtlas bridge: individual thermal quantities, implemented in
# ext/QAtlasExt.jl when QAtlas is loaded.
# ---------------------------------------------------------------------

"""
    thermal_energy(model, geom; beta) -> Float64 or Vector{Float64}

Exact thermal energy of `model` on the requested geometry
(`QAtlas.OBC(N)` / `QAtlas.Infinite()`). Requires `QAtlas` to be loaded.
"""
function thermal_energy end

"""
    thermal_magnetization_x(model, geom; beta)

Thermal transverse magnetization per site. QAtlas-loaded only.
"""
function thermal_magnetization_x end

"""
    thermal_specific_heat(model, geom; beta)

Thermal specific heat per site. QAtlas-loaded only.
"""
function thermal_specific_heat end

"""
    to_qatlas(model)

Translate an `ITensorModels` model to the matching `QAtlas` model
applying the Pauli-convention conversion. Implemented in QAtlasExt.
"""
function to_qatlas end

"""
    from_qatlas(qmodel)

Inverse of [`to_qatlas`](@ref). Implemented in QAtlasExt.
"""
function from_qatlas end

include("models/tfim.jl")

end # module ITensorModels
