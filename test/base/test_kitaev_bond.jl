using ITensorModels: KitaevBond, bond_term, site_type
using ITensors, ITensorMPS
using ITensors: SiteType

@testset "KitaevBond struct + site_type" begin
    m = KitaevBond()
    @test m.K == 1.0
    @test m.axis === :z
    @test m.site === SiteType("S=1/2")
    @test site_type(m) === SiteType("S=1/2")

    mq = KitaevBond(; K=0.5, axis=:x, site=SiteType("Qubit"))
    @test site_type(mq) === SiteType("Qubit")
end

@testset "KitaevBond emits single -K axis-axis term" begin
    for ax in (:x, :y, :z)
        m = KitaevBond(; K=1.3, axis=ax)
        op_sum = bond_term(m, 1, 2)
        # One OpSum term with operator name "Sx"/"Sy"/"Sz" on sites 1, 2.
        @test length(op_sum) == 1
    end
end

@testset "KitaevBond on |↑↑⟩ (spin-½) gives -K/4 for :z" begin
    sites = siteinds("S=1/2", 2)
    m = KitaevBond(; K=1.2, axis=:z)
    H = MPO(bond_term(m, 1, 2), sites)
    ψ = MPS(sites, ["Up", "Up"])
    # On |↑↑⟩, Sz = +1/2 each → SzSz = 1/4, so -K SzSz = -K/4.
    @test inner(ψ', H, ψ) ≈ -1.2 / 4 rtol = 1e-12
end

@testset "KitaevBond on Qubit |00⟩ gives -K for :z" begin
    sites = siteinds("Qubit", 2)
    m = KitaevBond(; K=0.7, axis=:z, site=SiteType("Qubit"))
    H = MPO(bond_term(m, 1, 2), sites)
    ψ = MPS(sites, ["0", "0"])
    # On |00⟩ with Qubit convention, Z = +1 each → ZZ = 1, so -K·1 = -K.
    @test inner(ψ', H, ψ) ≈ -0.7 rtol = 1e-12
end

@testset "KitaevBond unknown axis errors" begin
    m = KitaevBond(; K=1.0, axis=:bogus)
    @test_throws ErrorException bond_term(m, 1, 2)
end
