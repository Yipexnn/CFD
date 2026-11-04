# Phase 8 – Heat transfer

Goal: temperature transport with buoyancy (Boussinesq), thermal boundary conditions and
heat-flux / Nusselt output; serial and MPI, case files and GUI.

## Design
- **Energy equation** (`fv::EnergyEquation`): ddt(T) + div(phi T) = div((α + ν_t/Pr_t) grad T)
  - reuses `solve_scalar` (bounded convection, linear upwind, non-orthogonal correction, halo)
  - new scalar boundary kind `FixedGradient` (heat flux)
- **Thermal BCs** per patch: `fixed` temperature, `heat_flux` q (kinematic, K·m/s = q/(ρ c_p)),
  `adiabatic` (default); velocity inlets default to the reference temperature
- **Boussinesq buoyancy**: per-cell acceleration −β (T − T_ref) g added to the momentum source
  (`FlowSolver::set_cell_acceleration`)
- **Coupling**: after each SIMPLE iteration / PISO step: solve T, update the buoyancy force;
  residual `T` in residuals.csv and the convergence test
- **Output**: T in VTU and checkpoint (optional block, old checkpoints still readable); heat flow
  per patch (`heat` block in the case file → heat.csv, incl. Nusselt number)
- **Case file**: `physics.energy {prandtl | alpha, prandtl_t, beta, t_ref, gravity}`,
  `boundaries.<patch>.thermal {type, value}`, `initial.temperature`
- **GUI**: energy settings (physics tab), thermal BC columns, field T

## Validation
Differentially heated square cavity (de Vahl Davis 1983), Pr = 0.71:

| Ra | 1e3 | 1e4 | 1e5 | 1e6 |
|---|---|---|---|---|
| mean Nu (reference) | 1.118 | 2.243 | 4.519 | 8.800 |

## Definition of Done
- [ ] energy equation, thermal BCs, buoyancy; serial = MPI
- [ ] de Vahl Davis Nu within 1 % (Ra 1e3–1e5) on fine meshes
- [ ] case file, output, restart
- [ ] GUI
- [ ] tests, docs
