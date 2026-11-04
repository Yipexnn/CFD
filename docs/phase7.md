# Phase 7 – MPI parallelisation

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
  exchange, dot products with allreduce)
  - pressure: CG with overlapping additive Schwarz (local AMG on owned + ghost cells, ghost rows
    with their full diagonal, corrections added back to the owners) plus an additive coarse
    correction with 16 aggregates per process
  - momentum / turbulence: Jacobi-preconditioned BiCGStab
- **Global reductions**: residual normalisation, continuity error, mass-balance check, forces,
  convergence test
- **Output**: fields gathered to rank 0 → same single VTU / CSV / checkpoint as serial (GUI and
  restart unchanged, also restart with a different process count)
- **CLI**: `mpiexec -n 4 cfd run case.json`; only rank 0 logs
- **GUI**: "Processes" setting; N > 1 starts `mpiexec -n N cfd run` as a subprocess (QProcess),
  parses its log for the live residual plot, loads the final VTU; Stop terminates the process

## Results
- Parallel = serial to solver tolerance: cavity (laminar), SST channel with wall functions,
  Schäfer–Turek cylinder (Cd 5.5876062 on 1 and 4 processes, same 387 iterations).
- Restart works with any process count (checkpoint in global order).
- Speed-up (4-core VM, 1 thread per process):

| Case | 1 process | 2 | 4 |
|---|---|---|---|
| Cavity 256² (65k cells), 100 SIMPLE iterations | 8.8 s | 6.3 s | 3.8 s (2.3×) |
| Cylinder 2D-1 (5.3k cells, incl. Gmsh) | 3.8 s | 4.3 s | 2.8 s |

- Limitation: the pressure CG needs ~2.5× the serial iterations (domain decomposition
  preconditioner instead of a global AMG). A true parallel AMG would raise the speed-up further.

## Usage
- CLI: `mpiexec -n 4 cfd run case.json` (needs a build with `CFD_ENABLE_MPI=ON`)
- GUI: toolbar "Processes" > 1 → runs `mpiexec -n N cfd run` in the background; residuals are
  plotted live, the fields are loaded at the end. `CFD_MPIEXEC` / `CFD_MPIEXEC_FLAGS` override the
  launcher and its options.

## Validation / tests
- unit tests for partitioning (balance, ghost layer, face consistency)
- serial vs. 2/4-process results: cavity, channel SST, cylinder (Cd) agree to solver tolerance
- CI: Linux job with OpenMPI running the MPI tests with `mpiexec -n 2/4`
- benchmark: speed-up for cavity 256² and cylinder on 1/2/4 processes

## Definition of Done
- [x] serial behaviour unchanged (all existing tests)
- [x] parallel results match serial (laminar, SST, Gmsh meshes)
- [x] speed-up measured and documented
- [x] GUI parallel runs
- [ ] Windows build with MS-MPI in CI (GUI workflow)
- [ ] User tested on Windows (manual)
- [ ] later: parallel AMG, parallel mesh reading (each process reads the full mesh today)
