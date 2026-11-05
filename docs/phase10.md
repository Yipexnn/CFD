# Phase 10 – Windows test guide, geometry solids, multiphase (VOF), compressible

Order of work (each merged separately when CI is green):
1. [x] Windows test guide (`docs/windows_test.md`)
2. [x] Solid bodies in the geometry module (`Body::solid`, Gmsh fragments → physical volumes,
       2D + 3D; case file `bodies[].solid`; GUI: solids table)
3. [ ] Multiphase VOF: two immiscible incompressible fluids, MULES-like bounded interface
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

