using ITensorModels: TFIM, to_qatlas, from_qatlas, thermal_energy
using ITensors: SiteType
import QAtlas

@testset "to_qatlas / from_qatlas" begin
    # Qubit site → direct Pauli mapping (no rescale).
    m_q = TFIM(; J=0.4, h=0.6, site=SiteType("Qubit"))
    qm = to_qatlas(m_q)
    @test qm isa QAtlas.TFIM
    @test qm.J == 0.4
    @test qm.h == 0.6

    # S=1/2 site → J/4, h/2 rescale.
    m_s = TFIM(; J=1.0, h=0.5)
    qm2 = to_qatlas(m_s)
    @test qm2.J ≈ 0.25
    @test qm2.h ≈ 0.25

    back = from_qatlas(qm)
    @test back.J == 0.4
    @test back.h == 0.6
    @test back.site === SiteType("Qubit")
end

@testset "thermal_energy equivalence across SiteTypes" begin
    # TFIM(J=1, h=0.5; S=1/2) ≡ TFIM(J=0.25, h=0.25; Qubit) in QAtlas units.
    m_s = TFIM(; J=1.0, h=0.5)
    m_q = TFIM(; J=0.25, h=0.25, site=SiteType("Qubit"))
    @test thermal_energy(m_s, QAtlas.Infinite(); beta=2.0) ≈
        thermal_energy(m_q, QAtlas.Infinite(); beta=2.0) rtol = 1e-12

    ε = thermal_energy(m_q, QAtlas.Infinite(); beta=1.0)
    @test isfinite(ε)
    @test ε < 0
end

@testset "to_qatlas undefined for unsupported SiteType" begin
    m_s1 = TFIM(; site=SiteType("S=1"))
    @test_throws MethodError to_qatlas(m_s1)
end
