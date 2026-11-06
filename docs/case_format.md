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

Every body has `patch` (default `"obstacle"`); body patches are walls by default.
`"solid": {"conductivity": k, "rho_cp": c}` meshes the body as a solid region named `patch`
(conjugate heat transfer, needs `physics.energy`; not with STL bodies or internal flow): it is
kept as its own cell zone instead of being cut out, and its fluid-side surfaces become the wall
patch `patch` of the flow mesh. Patch types of
other patches follow their names (`inlet*`, `outlet*`, `wall*`, `sym*`), otherwise an entry in
`boundaries` is required.

## Physics

| Key | Type | Default | Description |
|---|---|---|---|
| `nu` | number | `0.001` | Kinematic viscosity [m^2/s]. Must be > 0. |
| `body_force` | [num, num, num] | `[0, 0, 0]` | Acceleration [m/s^2], e.g. a driving pressure gradient. |
| `turbulence` | string | `"laminar"` | `"laminar"` or `"kOmegaSST"`. |

### Compressible (`physics.compressible`)

Density-based solver for an ideal gas (HLLC fluxes, MUSCL reconstruction, explicit SSP-RK3;
steady mode uses local time stepping, explicit or implicit). With `physics.turbulence`
`"kOmegaSST"` the k-ω SST model is solved (density-weighted, wall-resolved: first cell
y+ ≲ 1). Not combinable with energy, solids or multiphase.

| Key | Type | Default | Description |
|---|---|---|---|
| `gamma` | number | `1.4` | Ratio of specific heats. |
| `r_gas` | number | `287` | Specific gas constant [J/(kg K)]. |
| `mu` | number | `0` | Dynamic viscosity [Pa s]; 0 = inviscid (Euler). |
| `prandtl` | number | `0.72` | Heat conduction k = mu c_p / Pr. |
| `cfl` | number | `0.5` | Courant number of the explicit time step. |
| `second_order` | bool | `true` | MUSCL reconstruction. |
| `limiter` | string | `"barthJespersen"` | `"barthJespersen"` (strict; transient shocks) or `"venkatakrishnan"` (smooth; steady runs converge to machine accuracy). |
| `venkat_k` | number | `0.1` | Venkatakrishnan threshold relative to the global range of each variable. |
| `implicit` | bool | `false` | Steady runs: implicit pseudo-time iterations (LU-SGS) instead of explicit RK3; `cfl` typically 50–1000. |
| `krylov_dim` | integer | `0` | Implicit: > 0 solves each step with Jacobian-free FGMRES (this Krylov dimension, LU-SGS preconditioned) – needed on stretched wall grids (viscous / turbulent). |
| `krylov_tolerance` | number | `0.1` | Relative linear tolerance of the FGMRES solve. |
| `prandtl_t` | number | `0.9` | Turbulent Prandtl number (heat flux c_p μ_t / Pr_t). |

Boundaries: `velocity_inlet` = fixed state (`velocity`, `pressure`, `temperature`);
`pressure_outlet` = static `pressure` if subsonic (`pressure` <= 0: extrapolate);
`wall` = no-slip adiabatic (slip if `mu` = 0); `symmetry` = slip; `empty`.
Initial: `initial.velocity`, `initial.pressure`, `initial.temperature` and
`initial.regions: [{"min", "max", "velocity", "pressure", "temperature"}]`.
Transient runs use dt = min(`solver.dt`, CFL limit); steady runs stop when the density
residual (relative to the largest of the first five iterations) and the k / ω residuals drop
below `solver.tolerance`. Turbulent inlets: `turbulence_intensity` and `viscosity_ratio`
(μ_t / μ); use at least I ≈ 0.01, μ_t/μ ≈ 1 – with almost no free-stream turbulence SST can
stay laminar.
Output fields: U, p, rho, T, Mach (+ k, omega, nut).

### Multiphase (`physics.multiphase`)

Two immiscible incompressible fluids (volume of fluid, transient only; not combinable with
energy or solids yet). `alpha` = volume fraction of phase 1. `physics.turbulence`
`"kOmegaSST"` solves the mixture k-ω SST model (volumetric flux, mixture viscosity; as
interFoam). Walls take a static contact angle (`boundaries.<wall>.contact_angle`).

| Key | Type | Default | Description |
|---|---|---|---|
| `rho1`, `nu1` | number | `1000`, `1e-6` | Phase 1 density [kg/m³] and kinematic viscosity [m²/s]. |
| `rho2`, `nu2` | number | `1`, `1.48e-5` | Phase 2. |
| `sigma` | number | `0` | Surface tension [N/m]. |
| `gravity` | [num, num, num] | `[0, 0, 0]` | [m/s²]. |
| `compression` | number | `1` | Interface compression coefficient. |

`initial.alpha_regions`: `[{"min", "max"}, ...]` boxes filled with phase 1 (cell centres).
`boundaries.<inlet>.alpha`: phase fraction entering through a velocity inlet (default 0).
Outlet `pressure` is the static pressure [Pa]; output fields: U, p, p_rgh, alpha, rho.

### Energy (`physics.energy`)

Enables the temperature equation with Boussinesq buoyancy −β (T − T_ref) g.

| Key | Type | Default | Description |
|---|---|---|---|
| `prandtl` / `alpha` | number | `prandtl: 0.71` | Prandtl number ν/α, or the thermal diffusivity α [m²/s] (not both). |
| `prandtl_t` | number | `0.85` | Turbulent Prandtl number (with kOmegaSST). |
| `beta` | number | `0` | Thermal expansion coefficient [1/K]. |
| `t_ref` | number | `0` | Reference temperature [K] (also inlet default and initial value). |
| `gravity` | [num, num, num] | `[0, 0, 0]` | Gravity [m/s²]. |
| `rho_cp` | number | `1` | Fluid volumetric heat capacity ρc_p (solid properties are relative to it). |

## Boundaries

Object keyed by patch name (`xmin` ... `zmax`). Each entry is an object with:

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | string | required | `velocity_inlet`, `wall`, `pressure_outlet`, `symmetry` or `empty`. |
| `velocity` | [num, num, num] | `[0, 0, 0]` | Inlet velocity (required for `velocity_inlet`) or wall velocity. |
| `pressure` | number | `0` | Outlet kinematic pressure (p / rho). |
| `turbulence_intensity` | number | `0.05` | Inlet turbulence: k = 1.5 (I \|U\|)^2. |
| `viscosity_ratio` | number | `10` | Inlet turbulence: nu_t / nu. |
| `contact_angle` | number | `90` | Multiphase walls: static contact angle [deg] between wall and interface, measured through phase 1. |
| `profile` | string | `"uniform"` | Velocity inlet: `"parabolic"` = `velocity` * prod 4 s (1 - s) over the patch extent (peak = `velocity`). |
| `thermal` | object | adiabatic (inlets: `t_ref`) | Energy equation: `{"type": "fixed" \| "heat_flux" \| "adiabatic", "value": T or q}`; q is kinematic (q/(ρ c_p), K·m/s, positive into the fluid). An entry with only `thermal` keeps the flow condition of the patch type. |
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
| `temperature` | number | `t_ref` | Initial temperature (energy equation). |

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
| `max_courant` | number | `0` | Transient multiphase: adapt the time step (at most `dt`) to this Courant number; `0` = fixed `dt`. |
| `device` | string | `"cpu"` | `"gpu"`: pressure and momentum linear systems with Jacobi-preconditioned CG / BiCGStab on the accelerator of the build (CUDA, OpenMP offload; host threads without a device). MPI runs use the CPU solvers. |
| `time_scheme` | string | `"euler"` | Transient: `"euler"` (1st order) or `"backward"` (BDF2, 2nd order). |

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
- `residuals.csv`: step, time, Ux, Uy, Uz, p, k, omega, continuity, T.
- `forces.csv`: step, time and per force patch Fx, Fy, Fz, Cx, Cy, Cz.
- `heat.csv`: step, time and per heat patch Q, Nu.
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

## Solids (conjugate heat transfer)

Optional top-level array (needs `physics.energy`). Each entry is a solid zone; the flow is
solved in the remaining (fluid) cells, the temperature everywhere. Fluid/solid faces become
wall patches named after the solid (usable in `boundaries`, `forces`).

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | string | required | Gmsh physical volume name, or the name of a box selection. |
| `min`, `max` | [num, num, num] | none | Box selection: cells whose centre lies inside. |
| `conductivity` | number | `1` | Thermal conductivity k [W/(m K)]. |
| `rho_cp` | number | `1` | Volumetric heat capacity ρc_p [J/(m³ K)]. |

## Heat

Optional top-level object (energy equation): heat flow into the fluid per patch, written to
`heat.csv` like `forces.csv`.

| Key | Type | Default | Description |
|---|---|---|---|
| `patches` | [string] | required | Patches to integrate, or solid names (heat flow solid → fluid). |
| `rho_cp` | number | `1` | Q = ρ c_p × kinematic heat flow [W]. |
| `reference_length`, `reference_delta_t`, `reference_area` | number | `1` | Nu = Q_kin L / (α ΔT A). |

## Restart

| Key | Type | Default | Description |
|---|---|---|---|
| `restart` | string (top level) | none | Checkpoint file to start from. Fields, step and time are restored; the mesh must match. |

## Running

```
cfd run examples/cavity.json
```

Other examples: `examples/channel_sst.json`, `examples/pipe_flow.json`.
