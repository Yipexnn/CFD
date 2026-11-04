# Phase 7 – MPI parallelisation (plan)

Goal: run cases on N processes (`mpiexec -n N cfd run case.json`) with results identical in
meaning to serial runs; serial build and behaviour unchanged.

## Design
- **Optional dependency**: `CFD_ENABLE_MPI` (default OFF). `cfd::par::Comm` wraps MPI
  (rank, size, allreduce, halo exchange); without MPI it is a serial stub (size 1).
  - Windows: MS-MPI (SDK via vcpkg `msmpi`, runtime installer from Microsoft for `mpiexec`)
  - Linux: OpenMPI (apt)
- **Decomposition** (`src/parallel`):
  - every rank builds/reads the full mesh (Gmsh runs on rank 0 only), then partitions it
    deterministically: recursive coordinate bisection (RCB) on cell centres (no extra dependency)
  - local mesh = owned cells + one layer of **ghost cells** with all their faces; faces of ghost
    cells towards non-local cells form a `processor` halo patch (zero flux)
  - all face loops (assembly, gradients, limiter, Rhie–Chow) stay unchanged; owned rows are
    complete because every face of an owned cell is local
- **Halo exchange** after every update of cell fields and gradients (U, p, k, ω, ν_t, wall
  distance, gradients); owned values sent to the neighbouring ranks' ghosts
- **Linear solvers**: own distributed CG / BiCGStab (owned rows only, matvec after ghost
  exchange, dot products with allreduce); AMG used as block-Jacobi preconditioner on the owned
  block of each rank
- **Global reductions**: residual normalisation, continuity error, mass-balance check, forces,
  convergence test
- **Output**: fields gathered to rank 0 → same single VTU / CSV / checkpoint as serial (GUI and
  restart unchanged, also restart with a different process count)
- **CLI**: `mpiexec -n 4 cfd run case.json`; only rank 0 logs
- **GUI**: "Processes" setting; N > 1 starts `mpiexec -n N cfd run` as a subprocess (QProcess),
  parses its log for the live residual plot, loads the final VTU; Stop terminates the process

## Validation / tests
- unit tests for partitioning (balance, ghost layer, face consistency)
- serial vs. 2/4-process results: cavity, channel SST, cylinder (Cd) agree to solver tolerance
- CI: Linux job with OpenMPI running the MPI tests with `mpiexec -n 2/4`
- benchmark: speed-up for cavity 256² and cylinder on 1/2/4 processes

## Definition of Done
- [ ] serial behaviour unchanged (all existing tests)
- [ ] parallel results match serial (laminar, SST, Gmsh meshes)
- [ ] speed-up measured and documented
- [ ] Windows build with MS-MPI in CI
