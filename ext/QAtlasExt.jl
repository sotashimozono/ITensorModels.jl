module QAtlasExt

using ITensorModels
using ITensors: SiteType, @SiteType_str
using QAtlas

# --- Model bridge -------------------------------------------------------

"""
    to_qatlas(m::ITensorModels.TFIM) -> QAtlas.TFIM

Translate to the QAtlas (Pauli-convention) struct, applying the local
operator convention coming from `m.site`. Register a
`to_qatlas(m, ::SiteType"...")` method to extend support.
"""
ITensorModels.to_qatlas(m::ITensorModels.TFIM) = ITensorModels.to_qatlas(m, m.site)

# Spin-½ site: our H = -J Σ SzSz - h Σ Sx = -(J/4) Σ σzσz - (h/2) Σ σx.
function ITensorModels.to_qatlas(m::ITensorModels.TFIM, ::SiteType"S=1/2")
    return QAtlas.TFIM(; J=m.J / 4, h=m.h / 2)
end

# Qubit site: operators already Pauli, no rescaling.
function ITensorModels.to_qatlas(m::ITensorModels.TFIM, ::SiteType"Qubit")
    return QAtlas.TFIM(; J=m.J, h=m.h)
end

"""
    from_qatlas(qm::QAtlas.TFIM) -> ITensorModels.TFIM

Inverse of [`to_qatlas`](@ref). Defaults to a `Qubit`-site struct so
`J`/`h` carry the same Pauli units as `qm`.
"""
function ITensorModels.from_qatlas(qm::QAtlas.TFIM)
    return ITensorModels.TFIM(; J=qm.J, h=qm.h, site=SiteType("Qubit"))
end

# --- fetch forwarder ----------------------------------------------------
#
# Route `QAtlas.fetch` calls that take an ITensorModels model through
# `to_qatlas`.  Every concrete-struct method
# `fetch(::QAtlas.Model, ::Quantity, ::BC; ...)` registered inside QAtlas
# becomes available for free on any `AbstractLatticeModel` whose site
# has a `to_qatlas` implementation.  Adding a method to `QAtlas.fetch`
# with an ITensorModels-owned argument type is not type piracy.

function QAtlas.fetch(
    m::ITensorModels.AbstractLatticeModel,
    q::QAtlas.AbstractQuantity,
    bc::QAtlas.BoundaryCondition;
    kwargs...,
)
    return QAtlas.fetch(ITensorModels.to_qatlas(m), q, bc; kwargs...)
end

end # module QAtlasExt
