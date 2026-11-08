# Phase 12 – Variable properties, radiation, 3D prism layers, (parallel mesh, GPU AMG)

## Variable properties
- `fv::PropertyLaw`: constant, polynomial Σ cᵢTⁱ, Sutherland, power law
- Incompressible + energy: ν(T) (`physics.energy.properties.nu`) updates a per-cell viscosity
  of the flow solver after each energy solve (`FlowSolver::set_cell_viscosity`; the
  div(ν (∇U)ᵀ) term is switched on for variable ν; SST uses the local ν); α(T) in the energy
  diffusivity
- Compressible: `physics.compressible.viscosity` μ(T) (e.g. Sutherland), conduction
  k = c_p μ(T)/Pr, also in SST and the implicit operator
- Tests (analytic): conduction slab with α = 1 + 2T (Kirchhoff transform, ≤ 2e-3), Couette
  flow with ν linear in T (u ∝ ln ν, ≤ 5e-3), Sutherland Couette = constant μ(300 K)

## P1 radiation
- `fv::P1Radiation`: div(Γ ∇G) − aG + 4aσT⁴ = 0, Γ = 1/(3(a + σₛ) − Cσₛ); Marshak conditions
  (new mixed/Robin `ScalarBoundary`) on walls, inlets, outlets with per-patch emissivity;
  energy source a(G − 4σT⁴) linearised (−16aσT³ implicit), divided by `heat.rho_cp`
- Case file `physics.energy.radiation`; `heat` flows include the radiative flux; field `G`
  (VTU, GUI)
- Validation: radiative equilibrium of a grey slab between grey walls — exact P1 result
  q/σ(T₁⁴ − T₂⁴) = 1/(3τ/4 + 1/ε₁ + 1/ε₂ − 1) reproduced within 0.5 % for τ = 0.5, 1, 5 and
  ε = 1 / 0.5 (wall fluxes balance to 0.2 %)

## 3D prism boundary layers
- Gmsh 4.12 has no 3D boundary-layer field, so two Gmsh runs:
  1. body surfaces (OpenCASCADE primitives / STEP, or STL triangles) meshed with `size_near`
     and exported as one STL solid per body
  2. the bodies are merged as discrete surfaces, extruded along their normals in
     `boundary_layer.layers` prism layers (`Extrude{Surface{..}; Layers{..}; Recombine;}`), and
     the box domain minus the layer tops is filled with tetrahedra (Distance/Threshold
     refinement from the layer tops)
- Same case syntax as 2D (`mesh.boundary_layer`); external flow with non-solid bodies
- MSH reader: bounding boxes of extruded/discrete entities (±DBL_MAX) are skipped as text
- Test: sphere in a box — prisms = 5 × wall faces, exact volume, first-layer distance,
  no negative cells
- Example `examples/sphere_3d.json` (sphere Re = 100, 6 prism layers):

| mesh | cells | Cd | reference |
|---|---|---|---|
| size_near 0.1 | 22 696 | 1.182 | 1.09 (Schiller–Naumann), 1.087 (Johnson & Patel) |

## Definition of Done
- [x] variable properties (incompressible + energy, compressible), tests
- [x] P1 radiation, Marshak walls, case files, validation
- [x] 3D prism layers for Gmsh geometries, 3D sphere example
- [ ] parallel mesh distribution (root reads/partitions, ranks receive only their part)
- [ ] GPU AMG preconditioner and GPU assembly
