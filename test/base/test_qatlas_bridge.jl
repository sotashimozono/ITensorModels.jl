using ITensorModels: TFIM, to_qatlas, from_qatlas, thermal_energy
using QAtlas: QAtlas

@testset "to_qatlas / from_qatlas" begin
    m_p = TFIM(; J=0.4, h=0.6, convention=:pauli)
    qm = to_qatlas(m_p)
    @test qm isa QAtlas.TFIM
    @test qm.J == 0.4
    @test qm.h == 0.6

    m_s = TFIM(; J=1.0, h=0.5, convention=:spin_half)
    qm2 = to_qatlas(m_s)
    @test qm2.J ≈ 0.25
    @test qm2.h ≈ 0.25

    back = from_qatlas(qm)
    @test back.J == 0.4
    @test back.h == 0.6
    @test back.convention === :pauli
end

@testset "thermal_energy via QAtlas.Infinite" begin
    m = TFIM(; J=1.0, h=1.0, convention=:pauli)      # Pauli-critical point
    ε = thermal_energy(m, QAtlas.Infinite(); beta=1.0)
    # Spot-check: finite, negative (ferromagnetic GS + thermal).
    @test isfinite(ε)
    @test ε < 0

    # :spin_half at the equivalent Pauli point (J/4=0.25, h/2=0.25) should
    # give the same QAtlas result.
    m_s = TFIM(; J=1.0, h=0.5, convention=:spin_half)
    m_p = TFIM(; J=0.25, h=0.25, convention=:pauli)
    @test thermal_energy(m_s, QAtlas.Infinite(); beta=2.0) ≈
        thermal_energy(m_p, QAtlas.Infinite(); beta=2.0) rtol = 1e-12
end
