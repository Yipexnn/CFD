# Phase 10 – Windows test guide, geometry solids, multiphase (VOF), compressible

Order of work (each merged separately when CI is green):
1. [x] Windows test guide (`docs/windows_test.md`)
2. [ ] Solid bodies in the geometry module (`Body::solid`, Gmsh fragments → physical volumes,
       2D + 3D; case file `bodies[].solid`; GUI: solids table)
3. [ ] Multiphase VOF: two immiscible incompressible fluids, MULES-like bounded interface
       capturing (or CICSAM), density/viscosity blending, surface tension (CSF); validation:
       dam break (Martin & Moyce), static droplet (pressure jump)
4. [ ] Compressible: pressure-based (rhoPimple-like) solver, ideal gas, total energy;
       validation: Sod shock tube, supersonic flow (wedge / forward step)
