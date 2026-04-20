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

# --- Thermal quantities -------------------------------------------------

function ITensorModels.thermal_energy(m::ITensorModels.TFIM, bc; beta)
    return QAtlas.fetch(
        ITensorModels.to_qatlas(m), QAtlas.Energy(), bc; beta=Float64(beta)
    )
end

end # module QAtlasExt
