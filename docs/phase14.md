# Phase 14 – Compressible far field, decomposed meshes, GPU assembly, sphere study

## Compressible characteristic far field
- Boundary type `far_field` (compressible): free stream from `velocity`, `pressure`,
  `temperature`; boundary state from the Riemann invariants (outgoing from the interior,
  incoming from the free stream), entropy and tangential velocity from the upwind side;
  supersonic in-/outflow handled; works for inflow and outflow patches
- Tests: an acoustic pulse leaving a subsonic tube (M ≈ 0.25) through both ends leaves
  0.008 % of its amplitude in the domain (fixed-state inlet + fixed-pressure outlet: 40 %);
  uniform subsonic flow with far field on all sides stays exact; case-file syntax / errors

## Decomposed meshes (each process reads only its part)
- `cfd decompose <case.json> <N>` (one process): builds the global mesh once, partitions it
  (RCB) and writes `<output>/processors<N>/processor<r>.bin` (local mesh with halo, global
  ids, global counts)
- `"decomposed_mesh": true`: `mpiexec -n N cfd run case.json` — every process reads only its
  own file; no process holds the global mesh; results are written as `<name>.pvtu` with one
  piece per process (owned cells); checkpoints stay in global order (restart with the same
  or — after re-decomposition — another process count)
- Without the flag the root still reads the mesh and distributes it (Phase 13)
- MPI test: decomposed run = distributed run (forces to 1e-12), pieces and .pvtu written,
  restart from its checkpoint
- Fix found by the test: since the Phase 13 distribution the non-root processes have no
  global mesh, so output paths must depend on the decomposed-run flag, not on that

## GPU assembly
- `gpu::DeviceSolver`: persistent per equation (kept in the solver's `AmgCache`): the CSR
  pattern, assembly positions, work vectors and the AMG coarse levels stay on the device; per
  solve only the LDU coefficients, right-hand side and initial guess are copied and the CSR
  values are assembled on the device (CUDA kernels); the host CSR is only built when the AMG
  hierarchy is rebuilt (every 20 solves)
- Without a CUDA device (OpenMP offload / host): assembly on the host, same solvers
- The finite-volume coefficients themselves are still computed on the CPU (face loops of
  the flow / scalar equations)
- Test: repeated solves with changing coefficients on one pattern (AMG-CG and BiCGStab) =
  CPU AMG-CG; CUDA build in CI (no GPU on the runners)

## Sphere drag study (Re = 100)

| mesh | cells | scheme | Cd |
|---|---|---|---|
| domain 20×10×10 D, size 1.0, near 0.06 | 37 181 | limited linear upwind | 1.187 |
| same | 37 181 | linear upwind (converged in 269 it.) | 1.182 |
| domain 20×10×10 D, size 0.5 | 105 345 | limited | 1.161 |
| domain 40×20×20 D, size 1.0, near 0.06, refine 5 | 123 350 | limited | 1.161 |
| reference (Johnson & Patel 1999) | | | 1.087 |

Ruled out: domain blockage (2× larger domain: same Cd), near-body resolution, the limiter.
Wake refinement helps (−2 %). The remaining +6.8 % most likely comes from the tetrahedral
discretisation away from the prism layers (non-orthogonality up to ~66°, skewness) or the
first-order wall-shear evaluation; next checks: more non-orthogonal correctors, a hex-dominant
or finer tetrahedral wake, comparison of pressure / friction drag separately.

## Definition of Done
- [x] compressible far-field BC + tests
- [x] decomposed meshes (decompose command, per-process reading, .pvtu output) + MPI test
- [x] GPU-side assembly / persistent device system + test, CUDA CI build
- [x] sphere study continued (blockage and limiter ruled out; cause still open)
- [ ] GPU run on real hardware (manual, `docs/windows_test.md` §5)
