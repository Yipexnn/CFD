# Phase 9 – Transient benchmark, parallel AMG, conjugate heat transfer

## 1. Vortex shedding (Schäfer–Turek 2D-2, Re = 100)
- new time scheme `backward` (BDF2, 2nd order; first step Euler), case key `solver.time_scheme`,
  GUI solver tab; unit test confirms 1st/2nd order
- program `validation/cylinder_schaefer_turek_2d2 [dt] [end_time] [size] [size_near]`,
  example `examples/cylinder_2d_re100.json`

| Mesh / dt | Cd_max | Cl_max | St |
|---|---|---|---|
| 6.3k cells, dt 0.004 | 3.276 | 1.084 | 0.275 |
| 6.3k cells, dt 0.001 | 3.243 | 1.018 | 0.292 |
| 11.4k cells, dt 0.0005 | 3.270 | 1.037 | 0.294 |
| Reference | 3.22–3.24 | 0.99–1.01 | 0.295–0.305 |

Within ~1 % (Cd, St) and ~3 % (Cl) of the reference ranges; a finer near-wake mesh would be
needed to land inside them.

## 2. Parallel AMG
- `fv::ParallelAmg`: distributed smoothed aggregation (see [phase7.md](phase7.md)); pressure CG
  iterations ~7.5 per solve on 2–4 processes (serial AMG ~4, previous Schwarz preconditioner ~11)
- cavity 256² on 4 processes: 3.0 s (was 3.8 s), 2.8× faster than 1 process

## 3. Conjugate heat transfer
- solid regions = cell zones: Gmsh physical volumes (`solids[].name` = volume name) or a box
  selection (`min`/`max`, cell centres)
- the flow is solved on the fluid zone (`parallel::extract_zone`; fluid/solid faces become wall
  patches named after the solid), the temperature on the full mesh with per-cell heat capacity
  and conductivity (harmonic interpolation at material jumps)
- solid properties `conductivity`, `rho_cp` (physical); fluid `physics.energy.rho_cp`
- `heat.patches` may name a solid: heat flow from that solid into the fluid
- output on the full mesh (U = 0 in solids), checkpoint/restart, MPI (zones decomposed with the
  mesh), GUI shows all fields on the full mesh
- tests: composite wall (exact), plate case energy balance and restart, MPI = serial,
  Gmsh physical volumes → zones
- example `examples/conjugate_plate.json` (heated plate under a laminar channel flow)

Limitations / later: solid bodies in the geometry module (Gmsh fragments), editing solids in the
GUI (kept from the case file), heat sources in solids.

## Definition of Done
- [x] BDF2 + 2D-2 validation
- [x] parallel AMG, faster than the Schwarz preconditioner
- [x] conjugate heat transfer (serial, MPI, case file, output, restart)
- [ ] user tested on Windows (manual)
