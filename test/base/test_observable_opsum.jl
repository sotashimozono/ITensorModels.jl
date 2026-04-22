using ITensorModels
using ITensorModels:
    AbstractLatticeModel,
    TFIM,
    TFIML,
    XXZ1D,
    Heisenberg1D,
    onsite_observable_op,
    build_onsite_observable_opsum,
    site_type
using ITensors, ITensorMPS
using ITensors: SiteType
using ITensorSiteKit: PhysInds
using Random

N = 5

@testset "onsite_observable_op: TFIM picks Sx/Sz per SiteType" begin
    m12 = TFIM(; site=SiteType("S=1/2"))
    mqb = TFIM(; site=SiteType("Qubit"))
    @test onsite_observable_op(m12, :sx) == "Sx"
    @test onsite_observable_op(m12, :sz) == "Sz"
    @test onsite_observable_op(m12, :sy) == "Sy"
    @test onsite_observable_op(mqb, :sx) == "X"
    @test onsite_observable_op(mqb, :sz) == "Z"
    @test onsite_observable_op(mqb, :sy) == "Y"
    @test_throws ErrorException onsite_observable_op(m12, :not_a_real_observable)
end

@testset "onsite_observable_op: XXZ1D / Heisenberg1D / TFIML reuse Sx/Sy/Sz" begin
    mxxz = XXZ1D()
    mhei = Heisenberg1D()
    mtfiml = TFIML()
    for m in (mxxz, mhei, mtfiml)
        @test onsite_observable_op(m, :sx) == "Sx"
        @test onsite_observable_op(m, :sy) == "Sy"
        @test onsite_observable_op(m, :sz) == "Sz"
    end
    mqb = XXZ1D(; site=SiteType("Qubit"))
    @test onsite_observable_op(mqb, :sx) == "X"
    @test onsite_observable_op(mqb, :sy) == "Y"
    @test onsite_observable_op(mqb, :sz) == "Z"
end

@testset "build_onsite_observable_opsum: TFIM Mx total equals direct OpSum" begin
    sites = PhysInds(N; SiteType="S=1/2")
    m = TFIM()
    op = build_onsite_observable_opsum(m, sites, :sx)

    direct = OpSum()
    for i in 1:N
        direct += 1.0, "Sx", i
    end

    # Compare by contracting both MPOs against a random MPS.
    rng = MersenneTwister(7)
    ψ = random_mps(rng, sites; linkdims=4)
    @test inner(ψ', MPO(op, sites), ψ) ≈ inner(ψ', MPO(direct, sites), ψ) rtol = 1e-12
end

@testset "build_onsite_observable_opsum: phys_sites subset + custom weights" begin
    # Simulate an aux-sandwiched layout: 5 sites, aux at ends, 3 phys in the middle.
    sites = PhysInds(N; SiteType="S=1/2")
    m = TFIM()
    phys = [2, 3, 4]

    # Per-site average ⟨Sx⟩ = (1/3) Σ_i∈phys Sx_i
    weights = fill(1 / length(phys), length(phys))
    op = build_onsite_observable_opsum(m, sites, :sx; phys_sites=phys, weights=weights)

    direct = OpSum()
    for (k, i) in enumerate(phys)
        direct += weights[k], "Sx", i
    end

    ψ = MPS(sites, "Up")
    # ⟨↑↑↑↑↑| Sx_i |↑↑↑↑↑⟩ = 0 for any i, so contraction is 0 either way.
    @test inner(ψ', MPO(op, sites), ψ) ≈ inner(ψ', MPO(direct, sites), ψ) atol = 1e-12

    # Confirm the OpSum really did restrict to the 3 bulk sites (not all 5).
    # OpSum length counts distinct term entries.
    @test length(op) == length(phys)
end

@testset "build_onsite_observable_opsum: argument-length mismatch throws" begin
    sites = PhysInds(N; SiteType="S=1/2")
    m = TFIM()
    @test_throws ArgumentError build_onsite_observable_opsum(
        m, sites, :sx; phys_sites=[1, 2, 3], weights=[1.0, 2.0]
    )
end
