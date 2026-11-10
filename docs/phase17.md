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
| 96k (near 0.03) | Gauss corrected | 2.887 | 1.110 | 1.776 |
| 96k (near 0.03) | least squares | 2.529 | 0.911 | 1.618 |
| 111k (refine box 0.25 around sphere and wake) | Gauss corrected | 2.870 | 1.106 | 1.764 |
| 111k (refine box 0.25) | least squares | 2.635 | 0.965 | 1.670 |
| reference (Magnaudet / Tabata & Itakura) | | 2.707 / 2.724 | | |

(α_u = 0.9, Majumdar correction; Gauss row: α_u 0.7 without correction)

- The reference lies between the Gauss-type (conservative) and least-squares results; refining
  near the sphere does not close the gap, refining the region around the sphere and wake
  moves both towards the reference (the coarse cells of 0.5–1 D at 1–3 D distance matter at
  Re 20, where the disturbance reaches far)
- Conclusion: the deficit is discretisation error of these 40–110k-cell meshes (the
  gradient scheme alone changes Cd by ±6–9 %), not a modelling error; accurate low-Re drag
  needs a finer mesh in the whole near field (size ≲ 0.25 D up to several D)

## Sphere Re 300 with LUST
- Same mesh as Phase 16 (98.8k cells, wake box size 0.2), restart at t = 40 from the steady
  oblique wake, BDF2 + PISO, dt 0.08, 4 MPI processes (1.4 s per step)
- A transient burst at t ≈ 90–100 (lift up to 0.3, half period ≈ 3.6 → St ≈ 0.14, close to
  Johnson & Patel's 0.137), then the wake settles again: t > 105 mean Cd 0.679, lift 0.046,
  decaying fluctuations – no sustained shedding on this wake mesh with any of the schemes
  (Barth-Jespersen, Venkatakrishnan, LUST); unlimited linear upwind diverges
