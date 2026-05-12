"""
    AbstractModulation

Root type for spatial envelopes applied to a bond-and-onsite-separable
`AbstractLatticeModel`. Modulations expose a per-site weight
`site_weight(mod, i, L)` and a per-bond weight `bond_weight(mod, i, L)`
(bond `i` connects sites `i` and `i+1`). Site / bond indices are
1-indexed.

Use [`ModulatedModel`](@ref) to wrap a model with a modulation; the
wrapper consumes the split protocol ([`bond_coupling_term`](@ref),
[`onsite_term`](@ref)) and re-emits `bond_term` / `boundary_patch`
that fit the existing `build_opsum` pipeline.
"""
abstract type AbstractModulation end

"""
    site_weight(mod::AbstractModulation, i::Int, L::Int) -> Float64

Envelope value at site `i` of a chain of length `L`. Returns 1.0 in
the unmodulated bulk.
"""
function site_weight end

"""
    bond_weight(mod::AbstractModulation, i::Int, L::Int) -> Float64

Envelope value on the bond connecting sites `i` and `i+1` of a chain
of length `L`. Returns 1.0 in the unmodulated bulk.
"""
function bond_weight end

# ---------------------------------------------------------------------
# Uniform — no modulation (sanity / degeneracy testing)
# ---------------------------------------------------------------------

"""
    Uniform()

Identity envelope (`site_weight = bond_weight ≡ 1`). Used to verify
that `ModulatedModel(base, L, Uniform())` reproduces the bare model;
this is the minimal correctness check for the split protocol.
"""
struct Uniform <: AbstractModulation end

site_weight(::Uniform, ::Int, ::Int) = 1.0
bond_weight(::Uniform, ::Int, ::Int) = 1.0

# ---------------------------------------------------------------------
# SSD — Sine-Square Deformation (Gendiar–Krcmar–Nishino, 2009)
# ---------------------------------------------------------------------

"""
    SSD()

Standard Sine-Square Deformation envelope:

- `site_weight(SSD(), i, L) = sin²(π (i - 1/2) / L)`
- `bond_weight(SSD(), i, L) = sin²(π i / L)`

Ground-state expectation values of local operators in the central
region coincide with those of the periodic chain of the same length
(Katsura, Maruyama, Tanaka, Katsura, 2011 for TFIM / free-fermion;
Gendiar–Krcmar–Nishino, 2009 for the original CFT-vacuum argument).
"""
struct SSD <: AbstractModulation end

site_weight(::SSD, i::Int, L::Int) = sin(pi * (i - 0.5) / L)^2
bond_weight(::SSD, i::Int, L::Int) = sin(pi * i / L)^2

# ---------------------------------------------------------------------
# SinPower{N} — generalized sin^N envelope
# ---------------------------------------------------------------------

"""
    SinPower{N}()

Generalized sine-power envelope: `site_weight ∝ sin^N(π (i - 1/2) / L)`,
`bond_weight ∝ sin^N(π i / L)`. `SinPower{2}()` is equivalent to
[`SSD()`](@ref); higher `N` (e.g. `SinPower{4}`, Hotta–Shibata 2012)
yields faster decay toward the boundary.
"""
struct SinPower{N} <: AbstractModulation end

site_weight(::SinPower{N}, i::Int, L::Int) where {N} =
    sin(pi * (i - 0.5) / L)^N
bond_weight(::SinPower{N}, i::Int, L::Int) where {N} =
    sin(pi * i / L)^N

# ---------------------------------------------------------------------
# SmoothBoundary — Vekic–White (1993) smooth boundary conditions
# ---------------------------------------------------------------------

"""
    SmoothBoundary(edge::Int)

Vekic–White (1993) smooth boundary envelope. The first and last
`edge` sites carry a half-cosine ramp from 0 to 1; bulk sites carry
weight 1. Used to suppress edge artefacts while keeping the bulk
Hamiltonian uniform.
"""
struct SmoothBoundary <: AbstractModulation
    edge::Int
end

function _smooth_ramp(d::Real, edge::Int)
    # d is the distance from the left/right boundary, measured at
    # half-integer offsets so that d = 1/2 sits at the outermost
    # site / bond. Returns 0 at d = 0 and 1 at d = edge.
    d >= edge && return 1.0
    d <= 0 && return 0.0
    return (1 - cos(pi * d / edge)) / 2
end

function site_weight(m::SmoothBoundary, i::Int, L::Int)
    e = m.edge
    e <= 0 && return 1.0
    d_left  = i - 0.5
    d_right = L - i + 0.5
    return _smooth_ramp(min(d_left, d_right), e)
end

function bond_weight(m::SmoothBoundary, i::Int, L::Int)
    e = m.edge
    e <= 0 && return 1.0
    # bond i sits between sites i and i+1, i.e. at position i (integer).
    d_left  = i
    d_right = L - i
    return _smooth_ramp(min(d_left, d_right), e)
end

# ---------------------------------------------------------------------
# Tabulated — escape hatch for arbitrary envelopes
# ---------------------------------------------------------------------

"""
    Tabulated(f_site::Vector{Float64}, f_bond::Vector{Float64})

Explicit per-site / per-bond weights, length `L` and `L-1`
respectively. Use this to script custom envelopes (smoothed
quasi-periodic, dipole, learned weights from QMC, …) without
defining a new `AbstractModulation` subtype.
"""
struct Tabulated <: AbstractModulation
    f_site::Vector{Float64}
    f_bond::Vector{Float64}
end

function site_weight(m::Tabulated, i::Int, L::Int)
    length(m.f_site) == L || error(
        "Tabulated: f_site length $(length(m.f_site)) != chain length L=$L",
    )
    return m.f_site[i]
end

function bond_weight(m::Tabulated, i::Int, L::Int)
    length(m.f_bond) == L - 1 || error(
        "Tabulated: f_bond length $(length(m.f_bond)) != L-1=$(L-1)",
    )
    return m.f_bond[i]
end
