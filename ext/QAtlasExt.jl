module QAtlasExt

using ITensorModels
using QAtlas

# --- Model bridge -------------------------------------------------------

"""
    to_qatlas(m::ITensorModels.TFIM) -> QAtlas.TFIM

Translate to the QAtlas Pauli-convention struct, applying the
spin-half → Pauli coefficient map when necessary.
"""
function ITensorModels.to_qatlas(m::ITensorModels.TFIM)
    if m.convention === :pauli
        return QAtlas.TFIM(; J=m.J, h=m.h)
    elseif m.convention === :spin_half
        # spin-1/2: H = -J Σ SzSz - h Σ Sx = -(J/4) Σ σzσz - (h/2) Σ σx
        return QAtlas.TFIM(; J=(m.J / 4), h=(m.h / 2))
    else
        error("to_qatlas: unknown convention $(m.convention)")
    end
end

"""
    from_qatlas(qm::QAtlas.TFIM) -> ITensorModels.TFIM

Return the Pauli-convention ITensorModels counterpart.
"""
function ITensorModels.from_qatlas(qm::QAtlas.TFIM)
    return ITensorModels.TFIM(; J=qm.J, h=qm.h, convention=:pauli)
end

# --- Thermal quantities -------------------------------------------------

function ITensorModels.thermal_energy(m::ITensorModels.TFIM, bc; beta)
    return QAtlas.fetch(to_qatlas(m), QAtlas.Energy(), bc; beta=Float64(beta))
end

end # module QAtlasExt
