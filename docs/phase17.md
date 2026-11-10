# Phase 17 – Discretization accuracy: relaxation-independent SIMPLE, corrected Gauss, LUST

## Relaxation-independent steady solutions (Majumdar correction)
- The face flux of SIMPLE contained the interpolated relaxation part (1 − α_u) u_prev of
  HbyA, so the converged steady solution depended on α_u (Rhie–Chow pitfall)
- Now replaced by (1 − α_u) φ_prev: the converged face flux is
  φ = interp(H/a)·S − interp(V/a) ∇p_f·S, independent of α_u
- Sphere Re 20 (37k cells, least squares): before α_u 0.9 → Cd 2.577, 0.5 → < 2.54;
  after both → 2.5835

## Skewness-corrected Gauss gradients
- `solver.gradient: "gauss_corrected"`: Green-Gauss with face values
  φ_f = w φ_P + (1 − w) φ_N + ∇φ_LSQ,f · (x_f − x_ip), x_ip on the line P–N
- Exact for linear fields on any cell shape (test on a skewed mesh; plain Gauss is not) and
  conservative (face sums telescope)
- Also on the GPU kernels (least-squares pass + corrected face sums; test: device = CPU for all
  three schemes)

## LUST convection (momentum)
- `solver.convection: "lust"`: 75 % linear interpolation + 25 % linear upwind, low numerical
  dissipation for vortex shedding / LES; CPU loop and GPU kernel (test: skewed cavity GPU = CPU)

## Sphere Re 20: gradient schemes and mesh refinement
| mesh | gradient | Cd | Cp | Cf |
|---|---|---|---|---|
| 37k (near 0.06) | Gauss | 2.945 | 1.143 | 1.802 |
| 37k | Gauss corrected | 2.888 | 1.116 | 1.771 |
| 37k | least squares | 2.583 | 0.937 | 1.646 |
| reference (Magnaudet / Tabata & Itakura) | | 2.707 / 2.724 | | |

## Sphere Re 300 with LUST
- Pending
