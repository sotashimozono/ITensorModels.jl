# Modulation envelope examples

Standalone scripts that visualise the modulation envelopes provided by
`ITensorModels`. Each script writes its output PNGs to `examples/out/`.

## Setup

```bash
cd examples
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The `Project.toml` here `develop`s the parent `ITensorModels` checkout
so script changes are picked up immediately.

## Scripts

| Script | Visualises |
|--------|------------|
| `plot_modulation_1d.jl` | 1D `AbstractModulation` envelopes (`SSD`, `SinPower{N}`, `SmoothBoundary`) — `site_weight` and `bond_weight` on a length-60 chain. |
| `plot_modulation_2d_continuous.jl` | 2D `RadialEnvelope` factory output evaluated on a continuous grid (heatmaps): rectangular / cylindrical / spherical / sin-power / smooth-ramp. |
| `plot_modulation_honeycomb_scatter.jl` | The same envelopes sampled on a real `honeycomb(L, L)` lattice with `OpenAxis()` — scatter plot coloured by `site_weight`. |

## Running

```bash
julia --project=. plot_modulation_1d.jl
julia --project=. plot_modulation_2d_continuous.jl
julia --project=. plot_modulation_honeycomb_scatter.jl
```

All scripts run headless (no display required) and write PNGs into
`out/` next to the script.
