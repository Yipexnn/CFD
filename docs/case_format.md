# Case file format

A case is a JSON file (`//` line comments allowed). It is parsed by
`src/case/src/case_config.cpp`. Unknown keys are rejected at every level, and
invalid values throw with the key path in the message. Relative paths are
resolved against the case file's directory.

Top-level keys: `name`, `mesh` (required), `physics`, `boundaries`, `initial`,
`solver`, `output`, `restart`. Every section and key except `mesh` is optional.

## Annotated example

`examples/cavity.json`: lid-driven cavity, Re 100.

```json
{
  "name": "cavity",                          // default: file stem
  "mesh": {
    "type": "box",                           // "box" or "gmsh"
    "cells": [32, 32, 1],                    // cells per direction
    "min": [0, 0, 0],                        // box corner
    "max": [1, 1, 0.03125],                  // opposite corner
    "patch_types": { "zmin": "symmetry", "zmax": "symmetry" }
  },
  "physics": { "nu": 0.01, "turbulence": "laminar" },   // Re = U L / nu = 100
  "boundaries": {
    "ymax": { "type": "wall", "velocity": [1, 0, 0] }   // moving lid
  },                                         // other walls: no-slip
  "solver": { "mode": "steady", "max_iterations": 3000, "tolerance": 1e-6 },
  "output": { "directory": "results/cavity", "write_interval": 0, "log_interval": 100 }
}
```

## Mesh

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | string | `"box"` | `"box"` (structured hex mesh) or `"gmsh"` (Gmsh 4.1 file). |
| `file` | string | required for gmsh | Gmsh `.msh` file. Only valid with `type: "gmsh"`. |
| `cells` | [int, int, int] | `[1, 1, 1]` | Cells per direction, each >= 1. Box only. |
| `min` | [num, num, num] | `[0, 0, 0]` | Box corner. Box only. |
| `max` | [num, num, num] | `[1, 1, 1]` | Opposite corner, must exceed `min` per direction. Box only. |
| `grading` | [num, num, num] | `[1, 1, 1]` | Last cell size / first cell size per direction (1 = uniform, >1 grows towards `max`). Box only. |
| `patch_types` | object | all `"wall"` | Keys `xmin`, `xmax`, `ymin`, `ymax`, `zmin`, `zmax`. Values: `wall`, `inlet`, `outlet`, `symmetry`, `patch`. Box only. |

Gmsh meshes accept only `type` and `file`; box meshes accept only the keys above.

## Physics

| Key | Type | Default | Description |
|---|---|---|---|
| `nu` | number | `0.001` | Kinematic viscosity [m^2/s]. Must be > 0. |
| `body_force` | [num, num, num] | `[0, 0, 0]` | Acceleration [m/s^2], e.g. a driving pressure gradient. |
| `turbulence` | string | `"laminar"` | `"laminar"` or `"kOmegaSST"`. |

## Boundaries

Object keyed by patch name (`xmin` ... `zmax`). Each entry is an object with:

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | string | required | `velocity_inlet`, `wall`, `pressure_outlet`, `symmetry` or `empty`. |
| `velocity` | [num, num, num] | `[0, 0, 0]` | Inlet velocity (required for `velocity_inlet`) or wall velocity. |
| `pressure` | number | `0` | Outlet kinematic pressure (p / rho). |
| `turbulence_intensity` | number | `0.05` | Inlet turbulence: k = 1.5 (I \|U\|)^2. |
| `viscosity_ratio` | number | `10` | Inlet turbulence: nu_t / nu. |

Patches without an entry get a default from their patch type:

| Patch type | Default condition |
|---|---|
| `wall` | No-slip wall |
| `symmetry` | Symmetry |
| `outlet` | Pressure outlet, p = 0 |
| `inlet`, `patch` | No default: an entry is required |

`empty` is valid only for directions with one cell layer (2D/1D cases).

## Initial

| Key | Type | Default | Description |
|---|---|---|---|
| `velocity` | [num, num, num] | `[0, 0, 0]` | Initial velocity field. |
| `k` | number | `-1` | Initial turbulent kinetic energy. <= 0: derived by the model. |
| `omega` | number | `-1` | Initial specific dissipation rate. <= 0: derived by the model. |

## Solver

| Key | Type | Default | Description |
|---|---|---|---|
| `mode` | string | `"steady"` | `"steady"` (SIMPLE) or `"transient"` (PISO). |
| `max_iterations` | int | `1000` | Steady: maximum SIMPLE iterations, must be >= 1. |
| `tolerance` | number | `1e-6` | Steady: convergence on the maximum initial residual. |
| `dt` | number | `1e-3` | Transient: time step, must be > 0. |
| `end_time` | number | `1.0` | Transient: end time, must be > 0. |
| `alpha_u` | number | `0.7` | Velocity under-relaxation, 0 < alpha <= 1. |
| `alpha_p` | number | `0.3` | Pressure under-relaxation, 0 < alpha <= 1. |
| `convection` | string | `"linear_upwind"` | `"linear_upwind"` or `"upwind"`. |
| `non_orthogonal_correctors` | int | `0` | Must be >= 0. |
| `piso_correctors` | int | `2` | Must be >= 1. |

`dt`, `end_time`, `alpha_*` and the corrector counts are validated in both modes.

## Output

| Key | Type | Default | Description |
|---|---|---|---|
| `directory` | string | `"results"` | Output directory, created if needed. |
| `write_interval` | int | `0` | Iterations / time steps between intermediate VTU writes. `0` = final only. |
| `log_interval` | int | `10` | Iterations / time steps between console log lines. |
| `checkpoint_interval` | int | `0` | Iterations / time steps between checkpoints. `0` = final only. |

### Output files

Written to `output.directory`:

- `<name>_<step>.vtu`: intermediate results (`write_interval > 0`).
- `<name>.vtu`: final result (cell data U, p and, with SST, k, omega, nut, yPlus).
- `residuals.csv`: step, time, Ux, Uy, Uz, p, k, omega, continuity.
- `checkpoint.bin`: restart file (fields, face flux, step and time).

## Restart

| Key | Type | Default | Description |
|---|---|---|---|
| `restart` | string (top level) | none | Checkpoint file to start from. Fields, step and time are restored; the mesh must match. |

## Running

```
cfd run examples/cavity.json
```

Other examples: `examples/channel_sst.json`, `examples/pipe_flow.json`.
