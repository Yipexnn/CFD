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
- Finer wake (box size 0.15, 173.7k cells, LUST, from rest, 4 processes, 4 s per step):
  the wake becomes **unsteady**; t = 30–87: mean Cd 0.678 (Johnson & Patel 0.656, +3 %),
  mean lift 0.061 (0.069), lift oscillation amplitude 0.017, irregular periods 4.6–13
  (St ≈ 0.12 vs 0.137); the run then diverges at t ≈ 88 (LUST is unbounded on tetrahedra)
- `solver.convection: "limited_lust"` (new): LUST with the limiter applied to the whole
  higher-order correction (CPU loop + GPU kernel; test: skewed cavity GPU = CPU). Same mesh,
  Venkatakrishnan K = 0.1, t = 0–110: **stable, but exactly steady** – Cd 0.6785,
  lift 0.0606, no oscillation
- Interpretation: the irregular unsteadiness of unlimited LUST was most likely the growing
  numerical instability, not physical shedding. Re 300 is only slightly above the onset of
  shedding (Re ≈ 270–280), so the weak physical instability is damped by the remaining
  numerical dissipation of these tetrahedral wake meshes (0.15–0.2 D)
- **Re 500** (same 174k mesh, `limited_lust`, Venkatakrishnan K = 0.1, t = 0–110, 94 min on 4
  processes): the wake is **unsteady** from t ≈ 30 on; t > 40: mean Cd 0.591 (correlation
  ≈ 0.555, +6.5 %), lift 0.084 ± 0.022, dominant St ≈ 0.10 (literature ≈ 0.17) – the solver
  captures the unsteady wake, but on 0.15 D tetrahedral wake cells the shedding is too weak and
  too slow, consistent with the excess dissipation seen at Re 300
- **Re 500, finer wake** (refine boxes 0.1 D in x 0.3–6 D, 0.2 D in 6–10 D, 254k cells, same
  schemes, 7.75 s per step on 4 processes, stopped by the time limit at t = 68): t > 30 mean
  Cd 0.571 (correlation ≈ 0.555, +2.9 %), mean |lift| 0.090, lift oscillation amplitude 0.064
  (3× the 174k mesh) – the finer wake strengthens the shedding and moves Cd towards the
  reference; the window holds only ≈ 2 periods (St 0.05–0.07, not converged – needs t ≳ 200)
- Open: sustained shedding at Re 300 (needs a finer / hex-dominant wake mesh or a higher
  Reynolds number test, e.g. Re 500–1000, where the instability is strong)
