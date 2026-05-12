# User-facing factories for common ND envelopes. The generic
# declarations live here; concrete methods that read the lattice
# geometry are implemented in `ext/LatticeCoreExt.jl`.

"""
    spherical_ssd(lat; radius=:inscribed, N=2) -> RadialEnvelope

Spherical SSD envelope on `lat`: BoundingBoxCenter + EuclideanDistance
+ SinPowerProfile{N}(R). `radius` selects the envelope radius from the
lattice's bounding box:

* `:inscribed` — `R = minimum_d (extent_d / 2)`. The envelope reaches
  zero on the inscribed disk; corners are clipped to zero.
* `:circumscribed` — `R = ‖extent / 2‖₂`. The envelope only reaches
  zero at the corners (the extreme points); edge midpoints carry a
  finite weight.

`N = 2` returns a [`SinSquareProfile`](@ref); `N > 2` widens the bulk
plateau (the radial Hotta–Shibata sin^N profile).
"""
function spherical_ssd end

"""
    cylindrical_ssd(lat; axis=1, N=2) -> RadialEnvelope

Cylindrical SSD envelope on `lat`: BoundingBoxCenter +
AxialDistance(axis) + SinPowerProfile{N}(R), where `R = extent[axis] / 2`.
The envelope is uniform in every direction other than `axis` — natural
for cylinders that are periodic in the perpendicular direction(s) and
SSD-modulated along `axis`.
"""
function cylindrical_ssd end

"""
    rectangular_ssd(lat; N=2) -> RadialEnvelope

Rectangular (axis-product) SSD envelope on `lat`: BoundingBoxCenter +
AxisProductDistance + SinPowerProfile{N}(SVector(R_1, …, R_D)), with
`R_d = extent_d / 2`. Equivalent to the LatticeCore hypercubic SSD
`∏_d sin²(π (c_d − 1/2) / L_d)`.
"""
function rectangular_ssd end
