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
| `type` | string | `"box"` | `"box"` (structured hex mesh), `"gmsh"` (Gmsh 4.1 file) or `"geometry"` (auto meshing with Gmsh, see below). |
| `file` | string | required for gmsh | Gmsh `.msh` file. Only valid with `type: "gmsh"`. |
| `cells` | [int, int, int] | `[1, 1, 1]` | Cells per direction, each >= 1. Box only. |
| `min` | [num, num, num] | `[0, 0, 0]` | Box corner. Box only. |
| `max` | [num, num, num] | `[1, 1, 1]` | Opposite corner, must exceed `min` per direction. Box only. |
| `grading` | [num, num, num] | `[1, 1, 1]` | Last cell size / first cell size per direction (1 = uniform, >1 grows towards `max`). Box only. |
| `patch_types` | object | all `"wall"` | Keys `xmin`, `xmax`, `ymin`, `ymax`, `zmin`, `zmax`. Values: `wall`, `inlet`, `outlet`, `symmetry`, `patch`. Box only. |

Gmsh meshes accept only `type` and `file`; box meshes accept only the keys above.

### Geometry (auto meshing)

`"type": "geometry"` writes a Gmsh script, runs the Gmsh executable (`CFD_GMSH` or `gmsh` on
`PATH`) in `<output directory>/mesh` and reads the result.

| Key | Type | Default | Description |
|---|---|---|---|
| `min`, `max` | [num, num, num] | `[0,0,0]`, `[1,1,1]` | Flow domain box (external flow). |
| `bodies` | array | `[]` | Solid bodies subtracted from the box (see below). |
| `internal` | bool | `false` | Internal flow: the fluid is the single STEP `file` body; its surfaces are `walls` unless a `patches` selector claims them. |
| `two_d` | bool | `false` | 2D: geometry in the plane z = `min.z`, extruded by one cell layer to `max.z` (use `empty` on the z patches). Box and cylinder bodies only. |
| `size` | number | `0.05` | Maximum cell size. |
| `size_near` | number | `0` | Cell size at body surfaces (<= 0: `size`). |
| `refine_distance` | number | `0` | Distance over which `size_near` grows to `size`. |
| `boundary_layer` | object | none | `layers`, `first` (first cell height), `ratio` (growth). Quad layers around bodies, 2D only. |
| `quads` | bool | `true` | 2D: recombine into quads (hex cells after extrusion). |
| `domain_patches` | object | face names | Patch names of the box faces `xmin` ... `zmax`. Equal names are merged. |
| `patches` | array | `[]` | `{"name", "min", "max"}`: every boundary surface inside the box gets this patch name. |

Bodies:

| `type` | Keys |
|---|---|
| `box` | `min`, `max` |
| `cylinder` | `base` (centre of the base), `axis` (vector, its length = height; default `[0,0,1]`; 2D: disk at `base`), `radius` |
| `sphere` | `centre`, `radius` |
| `file` | `file`: `.stl` (closed triangulation) or `.step`/`.stp`, relative to the case file |

Every body has `patch` (default `"obstacle"`); body patches are walls by default. Patch types of
other patches follow their names (`inlet*`, `outlet*`, `wall*`, `sym*`), otherwise an entry in
`boundaries` is required.

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
| `profile` | string | `"uniform"` | Velocity inlet: `"parabolic"` = `velocity` * prod 4 s (1 - s) over the patch extent (peak = `velocity`). |
| `profile_axes` | string | `"y"` | Axes of the parabolic profile, e.g. `"y"` (2D channel) or `"yz"` (rectangular duct). |

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
| `convection` | string | `"linear_upwind"` | `"linear_upwind"`, `"limited_linear_upwind"` (bounded, recommended for high Reynolds numbers / coarse meshes) or `"upwind"` (first order). |
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
- `forces.csv`: step, time and per force patch Fx, Fy, Fz, Cx, Cy, Cz.
- `mesh/`: Gmsh script, mesh and log (geometry meshes).
- `checkpoint.bin`: restart file (fields, face flux, step and time).

## Forces

Optional top-level object; forces on the listed patches are written to `forces.csv`
(every `log_interval` steps and at the end).

| Key | Type | Default | Description |
|---|---|---|---|
| `patches` | [string] | required | Patches to integrate (pressure + viscous). |
| `density` | number | `1` | Force = density * kinematic force. |
| `reference_velocity` | number | `1` | Coefficients C = F / (0.5 rho U^2 A). |
| `reference_area` | number | `1` | Reference area A (2D: diameter * depth). |

## Restart

| Key | Type | Default | Description |
|---|---|---|---|
| `restart` | string (top level) | none | Checkpoint file to start from. Fields, step and time are restored; the mesh must match. |

## Running

```
cfd run examples/cavity.json
```

Other examples: `examples/channel_sst.json`, `examples/pipe_flow.json`.
