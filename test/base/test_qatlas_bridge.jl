using ITensorModels: TFIM, to_qatlas, from_qatlas
using ITensors: SiteType
import QAtlas
using QAtlas: Energy, MassGap, Infinite, OBC

# These tests are deliberately non-tautological: they validate the
# QAtlas bridge against (a) analytic TFIM results and (b) agreement
# between the ITensorModels + ext forwarder and a direct QAtlas call on
# a hand-written struct.

@testset "to_qatlas round-trip" begin
    m_q = TFIM(; J=0.4, h=0.6, site=SiteType("Qubit"))
    @test to_qatlas(m_q).J == 0.4 && to_qatlas(m_q).h == 0.6

    back = from_qatlas(to_qatlas(m_q))
    @test back.J == 0.4 && back.h == 0.6 && back.site === SiteType("Qubit")
end

@testset "MassGap matches analytic 2|h-J|" begin
    for (J, h) in [(1.0, 0.3), (1.0, 1.7), (0.8, 0.5)]
        m = TFIM(; J=J, h=h, site=SiteType("Qubit"))
        Δ = QAtlas.fetch(m, MassGap(), Infinite())
        @test Δ ≈ 2 * abs(h - J) rtol = 1e-12
    end
end

@testset "SiteType unit conversion: S=1/2 ≡ Qubit(J/4, h/2)" begin
    # Same physical Hamiltonian expressed in two conventions.
    J, h = 1.0, 0.5
    m_s = TFIM(; J=J, h=h)                                 # S=1/2 default
    m_q = TFIM(; J=J / 4, h=h / 2, site=SiteType("Qubit"))  # equivalent Pauli

    @test QAtlas.fetch(m_s, Energy(), Infinite(); beta=2.0) ≈
        QAtlas.fetch(m_q, Energy(), Infinite(); beta=2.0) rtol = 1e-12
    @test QAtlas.fetch(m_s, MassGap(), Infinite()) ≈
        QAtlas.fetch(m_q, MassGap(), Infinite()) rtol = 1e-12
end

@testset "Forwarder parity with direct QAtlas call" begin
    # Hand-written QAtlas struct with the Pauli couplings that match a
    # :S=1/2 ITensorModels.TFIM(J=1, h=0.5). The forwarder's output must
    # equal a direct fetch on the Pauli struct.
    m = TFIM(; J=1.0, h=0.5)
    qm_direct = QAtlas.TFIM(; J=0.25, h=0.25)

    @test QAtlas.fetch(m, Energy(), OBC(12); beta=3.0) ≈
        QAtlas.fetch(qm_direct, Energy(), OBC(12); beta=3.0) rtol = 1e-12
    @test QAtlas.fetch(m, MassGap(), OBC(12)) ≈
        QAtlas.fetch(qm_direct, MassGap(), OBC(12)) rtol = 1e-12
end

@testset "to_qatlas undefined for unsupported SiteType" begin
    m_s1 = TFIM(; site=SiteType("S=1"))
    @test_throws MethodError to_qatlas(m_s1)
end
