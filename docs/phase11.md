# Phase 11 – Implicit / turbulent compressible, multiphase turbulence, contact angles, GPU

## Implicit compressible solver
- `physics.compressible.implicit`: steady pseudo-time iterations (V/Δt − ∂R/∂Q) ΔQ = R with
  local Δt from `cfl`
- **LU-SGS** (Jameson & Yoon, unstructured form): Rusanov-type flux-Jacobian splitting,
  one forward + one backward Gauss–Seidel sweep, matrix-free (linearised flux differences);
  process boundaries explicit (ΔQ of ghost cells = 0), so MPI = serial at convergence
- **Jacobian-free Newton–Krylov** (`krylov_dim` > 0): right-preconditioned FGMRES with the
  exact operator by finite differences of R, LU-SGS as preconditioner, scaled inner products,
  inexact solve (`krylov_tolerance`). Plain LU-SGS behaves like Gauss–Seidel for wall-normal
  diffusion and stalls on y+ ≈ 1 grids; the Krylov solve removes this.
- **Venkatakrishnan limiter** (`limiter`): smooth; Barth–Jespersen limiter "chatter" stalls
  steady residuals at ~1e-2
- Positivity safeguard on the update; density residual relative to the largest of the first
  five iterations

| Supersonic ramp (7200 cells, residual 1e-7) | iterations | time |
|---|---|---|
| explicit RK3, Barth–Jespersen | stalls at 3e-2 | – |
| explicit RK3, Venkatakrishnan | 2545 | 24 s |
| LU-SGS CFL 20 / 200, Venkatakrishnan | 353 / 297 | 1.7 s / 1.1 s |

Shock: p2/p1 2.4674 vs theory 2.4675, angle 36.98° vs 36.94°.

## Compressible k-ω SST
- Density-weighted SST (Menter 2003): ρk, ρω with mass flux, capacity ρ (transient),
  μ + σ μ_t; solved segregated (implicit, upwind convection, under-relaxed when steady) after
  each mean-flow update; μ_t in the viscous stresses and the heat flux (Pr_t)
- Low-Re walls: ω in wall cells from the viscous/log blend, k zero gradient, μ_t = 0 on walls
- Validation `validation/flat_plate_turbulent` (NASA TMR 2DZP setup: M 0.2, Re 5e6/m):

| grid | iterations (1e-8) | Cf(x=0.97) | TMR ref. | 1/5-power law |
|---|---|---|---|---|
| 80×64 | 1727 | 0.00260 | ~0.0027 | 0.00272 |
| 160×96 | 1834 | 0.00261 | | |

  Laminar check (same program, `laminar`): Blasius Cf 0.00058/0.00030/0.00025 at
  x = 0.25/0.97/1.5 matched.
- Free-stream turbulence matters: with I = 0.001, μ_t/μ = 0.1 the solution can stay on the
  laminar branch (seen on CI); examples/tests use I = 0.01, μ_t/μ = 1.
- Example `examples/flat_plate_turbulent.json` (2304 cells, converges in ~630 steps, ~35 s)

## Multiphase turbulence
- `KOmegaSst` now works on a generic flow state (`SstFlowState`: velocity, volumetric flux,
  velocity gradient, boundary velocities, per-cell laminar ν); FlowSolver unchanged
- VOF: mixture SST (as interFoam): ν = μ_mix/ρ_mix, μ_eff = μ_mix + ρ ν_t (wall-function ν_t
  on walls); enabled with `physics.turbulence: "kOmegaSST"`
- Test: VOF channel filled with one phase = single-phase SIMPLE + SST (same upwind scheme)

## Static contact angle
- `boundaries.<wall>.contact_angle` [deg, through phase 1]; at walls the interface normal is
  turned to n_w·n = cos θ keeping its tangential direction (as OpenFOAM's
  constantAlphaContactAngle), used for the consistent wall α value in the gradient and to
  close the curvature divergence (without it: 90°)
- Validation `validation/droplet_contact_angle` (2D droplet relaxing from a semicircle;
  angle from the cap height and area):

| prescribed | 45° | 60° | 90° | 120° |
|---|---|---|---|---|
| 96×40, t = 0.12 s | 45.7° | 60.3° | 88.4° | 116.8° |

  The contact line moves only through numerical slip (no-slip walls), so finer grids need
  longer to reach equilibrium.

## GPU linear solvers
- Library `src/gpu` (`cfd::gpu::solve`): CSR matrix, Jacobi-preconditioned CG and BiCGStab;
  all vectors stay on the device during a solve (matrix and right-hand side copied once)
- Backends (`CFD_GPU_BACKEND`):
  - `cuda`: own kernels (CSR SpMV, fused vector updates, block-reduced dot products with double
    `atomicAdd`, compute capability ≥ 6.0); used when a CUDA device is present
  - `openmp`: OpenMP `target teams distribute parallel for` kernels inside one `target data`
    region (GCC nvptx / Clang offload flags via `CFD_OMP_OFFLOAD_FLAGS`)
  - without a device (or `none`): the same kernels with host OpenMP threads
- Use: `solver.device: "gpu"` → pressure `GpuCg`, momentum `GpuBiCgStab` (FlowSolver, VOF);
  decomposed (MPI) systems fall back to the distributed CPU solvers; `cfd --info` prints the
  backend and device; GUI: "Linear solvers" CPU / GPU
- Trade-off: Jacobi-PCG needs many more iterations than AMG-PCG for the pressure (O(N^½) vs
  ~constant), so the GPU pays off for large meshes (≳ 10⁵ cells); on the CPU keep AMG.
  Cavity 128², 3000 SIMPLE iterations, 4 cores: CPU AMG 29 s, GPU solvers on host threads
  (no device) 67 s
- CI (`gpu-backends` job): CUDA 12.6 container build (tests use the host fallback — no GPU on
  the runners) and GCC 13 with real nvptx offload code; local: same lid-driven cavity result
  with GPU and CPU solvers (1e-5)
- Manual test on the user's NVIDIA GPU: `docs/windows_test.md` §5

## Definition of Done
- [x] implicit compressible (LU-SGS, Newton–Krylov), Venkatakrishnan limiter
- [x] compressible SST, turbulent flat plate validation, case files, checkpoint, GUI fields, MPI
- [x] multiphase turbulence (mixture SST), case files, checkpoint, MPI
- [x] static contact angle, droplet validation
- [x] GPU linear solvers (CUDA + OpenMP offload, host fallback), CI build of both backends
- [ ] GPU tested on real hardware (manual, Windows + NVIDIA)
- [ ] later: GPU AMG preconditioner, GPU assembly
- [ ] GUI: compressible / contact-angle settings in the case editor (values are kept from the
      case file)
