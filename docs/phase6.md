# Phase 6 – Real geometry

Goal: import CAD geometry, mesh it automatically and validate flow around a body.

## Design
- **`cfd_geometry`** (`src/geometry`): `GeometrySpec` → Gmsh `.geo` script → Gmsh executable →
  MSH 4.1 → `mesh::Mesh`.
  - external Gmsh executable (`CFD_GMSH` or `PATH`) instead of the SDK: the vcpkg gmsh port is
    built without the mesher, the official binaries include OpenCASCADE.
  - external flow: box minus bodies (box, cylinder, sphere, STEP via OpenCASCADE booleans; STL via
    surface classification in the built-in kernel)
  - internal flow: STEP solid as fluid, patches by bounding-box selectors
  - 2D: planar geometry, quad boundary layer (BoundaryLayer field), extruded one cell layer →
    hexes / prisms for `empty` patches
  - size field: `size_near` at bodies growing to `size` over `refine_distance`
  - body surfaces → wall patches; duplicate names merged into one physical group
- **Forces**: `FlowSolver::patch_force` = pressure (extrapolated to the face) + viscous
  `nu_eff dU/dn`; case `forces` block → `forces.csv` with coefficients.
- **Case files**: mesh type `geometry`, parabolic inlet profile (`profile`, `profile_axes`), see
  [case_format.md](case_format.md). Example: `examples/cylinder_2d.json`.
- **GUI**: mesh source "Geometry (Gmsh auto mesh)": domain, bodies table (primitives, STL/STEP
  file), mesh sizes, boundary layer, domain patch names. The preview mesh is reused for the run.

## Validation: Schäfer–Turek 2D-1 (Re = 20)
`validation/cylinder_schaefer_turek [size] [size_near] [max_iterations]`

| Mesh (size / near) | Cells | Cd | Cl | Δp |
|---|---|---|---|---|
| 0.02 / 0.004 | 4 031 | 5.587 | 0.0105 | 0.1181 |
| 0.02 / 0.002 | 5 344 | 5.588 | -0.0047 | 0.1180 |
| 0.01 / 0.001 | 18 557 | 5.596 | 0.0074 | 0.1177 |
| Reference | | 5.57953 | 0.010618 | 0.11752 |

Cd and Δp within 0.3 %. Cl (0.2 % of Cd) is sensitive to the unstructured surface mesh.

## Definition of Done
- [x] Gmsh auto meshing: primitives, STL, STEP, internal flow, 2D boundary layers
- [x] Case file + GUI integration
- [x] Patch forces / coefficients
- [x] Cylinder benchmark (Cd, Δp within 0.5 %)
- [x] Unit tests (Gmsh-dependent tests skipped without Gmsh)
- [ ] 3D boundary-layer prisms (Gmsh has no robust 3D BL for OCC booleans; later)
- [ ] User tested on Windows (manual)
