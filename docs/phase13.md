# Phase 13 – GUI settings, sphere study, radiation + solids, parallel mesh, GPU AMG

## GUI settings
- Physics tab (scrollable): ν(T) / α(T) polynomial coefficients (other laws from the case
  file are kept: "keep: …"), P1 radiation (absorption, wall emissivity), compressible group
  (γ, R, μ, Sutherland, CFL, implicit, Krylov dimension, limiter)
- Boundary table: compressible inlet temperature, multiphase contact angle
- Test `cfd_case_editor_tests` (headless Qt): editor round trip of all new settings; settings
  without widgets (per-patch emissivity) are kept

## Sphere drag study (Re = 100, `examples/sphere_3d.json`)

| mesh | cells | Cd |
|---|---|---|
| size 1.0, size_near 0.1 | 22 696 | 1.182 |
| size 1.0, size_near 0.06 | 37 181 | 1.187 |
| size 0.5, size_near 0.1, refine 4 | 105 345 | 1.161 |
| reference (Johnson & Patel 1999; Schiller–Naumann) | | 1.087 / 1.09 |

Refining near the body does not change Cd; refining the wake / far field lowers it (−2 %). The
remaining +6.5 % is consistent with the small domain (symmetry box 10 D wide, 5 D upstream:
blockage) and the coarse surface (≈ 30 cells around the circumference). A larger domain with
a finer surface is the next check (long run, not done).

## Radiation with solid regions
- P1 is solved in the fluid zone; fluid/solid interfaces are opaque walls (Marshak, default
  emissivity) at the conductivity-weighted surface temperature; the net radiative flux is a
  surface heat sink of the solid cell behind each interface face
- Radiative source scaled by `physics.energy.rho_cp` (fluid ρc_p)
- Test: conducting plate (k = 50 W/(m K)) radiating across a transparent gap to a 500 K wall:
  surface temperature and heat balance match k (T0 − Ti) = σ (Ti⁴ − Tw⁴) (≤ 1–2 %)

## Parallel mesh distribution
- Only the root process reads / generates the global mesh; `parallel::distribute` partitions
  it (RCB), builds every local mesh and sends it (`pack_local_mesh` / `unpack_local_mesh`, sizes
  first, then the meshes); the other processes never hold the global mesh
- Runner: global counts broadcast, solid zones selected on the local mesh (cell zones are
  decomposed too), parabolic inlet extents via global min/max, gathered output / checkpoints
  unchanged (written by the root)
- MPI test: distributed local meshes = locally decomposed ones (2 and 3 processes); all MPI
  tests pass on 2 and 4 processes
- Remaining limit: the root still needs memory for the global mesh (a fully parallel reader
  would split the file)

## GPU AMG
- `fv::AmgPreconditioner` exports its hierarchy (operators, transfer operators, dense coarse
  inverse); `gpu::solve_amg_cg`: CG with one V-cycle per iteration (damped Jacobi ×2 pre/post,
  dense coarse solve) on CUDA or OpenMP offload (host threads without a device)
- `solver.device: "gpu"` → pressure: AMG-PCG on the accelerator (hierarchy built and cached
  on the host, rebuilt every 20 solves); momentum: Jacobi BiCGStab
- Poisson 40² / 80²: < 40 AMG-PCG iterations, > 3× fewer than Jacobi-PCG; cavity test
  0.7 s (was 2.2 s with Jacobi-PCG); cavity 128² 3000 iterations: CPU AMG 26 s, GPU path on
  host threads 35 s (no device here)
- Matrix assembly stays on the CPU (FV coefficients are assembled in face loops on the host;
  moving them is a larger refactor)

## Definition of Done
- [x] GUI settings + test
- [x] sphere study (partly explained, larger-domain run open)
- [x] radiation with solids + test
- [x] parallel mesh distribution + MPI tests
- [x] GPU AMG-PCG (CUDA / offload / host) + tests
- [ ] GPU assembly; GPU run on real hardware (manual, `docs/windows_test.md` §5)
