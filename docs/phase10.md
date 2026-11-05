# Phase 10 – Windows test guide, geometry solids, multiphase (VOF), compressible

Order of work (each merged separately when CI is green):
1. [x] Windows test guide (`docs/windows_test.md`)
2. [x] Solid bodies in the geometry module (`Body::solid`, Gmsh fragments → physical volumes,
       2D + 3D; case file `bodies[].solid`; GUI: solids table)
3. [x] Multiphase VOF: two immiscible incompressible fluids, MULES-like bounded interface
       capturing (or CICSAM), density/viscosity blending, surface tension (CSF); validation:
       dam break (Martin & Moyce), static droplet (pressure jump)
4. [ ] Compressible: pressure-based (rhoPimple-like) solver, ideal gas, total energy;
       validation: Sod shock tube, supersonic flow (wedge / forward step)

## 2. Solid bodies (done)
- `Body::solid`: OCC BooleanFragments with the fluid volume (3D) or surface (2D, then extruded);
  physical volume per solid (found by bounding box), interface surfaces get no patch
- case file: `bodies[].solid {conductivity, rho_cp}` (round-trips with the body); GUI bodies
  table columns "solid k" / "solid rho cp"
- example `examples/conjugate_block_2d.json` (serial = 3 MPI processes = GUI)

## 3. Multiphase VOF (done)
- `fv::MultiphaseSolver`: explicit alpha transport (upwind + central/compression flux limited
  with Zalesak FCT, sub-cycled to Co_alpha 0.25), mixture density/viscosity, momentum with the
  mass flux, PISO on p_rgh (gravity and CSF surface tension as face forces, reconstructed to
  cells), serial and MPI
- case file `physics.multiphase`, `initial.alpha_regions`, inlet `alpha`, `solver.max_courant`
  (adaptive dt); output alpha, p, p_rgh, rho; checkpoint/restart; GUI field alpha
- validation `validation/dam_break` (Martin & Moyce): mass conserved to 1e-14, alpha bounded
  to 1e-9; front speed matches the experiment (slope 1.30 vs 1.30 for T 1.4–3.0), constant
  offset +0.25 a (experimental gate opening delay, typical for VOF comparisons)

| T | 0.84 | 1.43 | 1.98 | 2.51 | 2.97 |
|---|---|---|---|---|---|
| Z simulation (120×60) | 1.38 | 1.92 | 2.54 | 3.25 | 3.92 |
| Z Martin & Moyce | 1.22 | 1.67 | 2.33 | 3.00 | 3.67 |

- tests: hydrostatic tank at rest, static drop Laplace pressure (±10 %), dam-break mass and
  bounds, case file + restart, MPI = serial
- example `examples/dam_break.json`
- later: momentum predictor / higher-order momentum convection, multiphase with turbulence,
  contact angle

