# Phase 4 – Case I/O

Goal: run complete cases from JSON files, with VTU output, residual logs and exact restart.

## Delivered
- `cfd::cases` library (`src/case`)
  - `parse_case` / `load_case`: JSON case files (comments allowed), strict validation with path-annotated errors. Format: [case_format.md](case_format.md).
  - `CaseRunner`: builds the mesh (box or Gmsh), FlowSolver and optional k-ω SST, then runs steady (SIMPLE) or transient (PISO) cases. A progress callback can stop the run (for the GUI).
  - Output per case: `<name>.vtu` (U, p, + k, ω, ν_t, y+), optional `<name>_<step>.vtu`, `residuals.csv`, `checkpoint.bin`.
  - Checkpoint/restart: binary file with U, p, face flux, k, ω, ν_t, step and time. Restart is bit-exact for laminar and SST cases.
- CLI: `cfd run <case.json>`
- Examples: `examples/cavity.json`, `examples/pipe_flow.json`, `examples/channel_sst.json`

## Deferred
- Faster pressure solver (AMG) and the backward-facing step validation: separate performance phase.

## Definition of Done
- [x] Case parsing tests (valid, defaults, errors)
- [x] Runner tests: outputs, intermediate writes, transient end time, stop callback, exact restart (laminar + SST), mesh-mismatch error
- [x] CI green
