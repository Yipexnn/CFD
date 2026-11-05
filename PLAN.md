# CFD Software – Plan

Goal: realistic 3D CFD desktop application, running locally on Windows.

## Decisions
- **Physics (first):** incompressible + RANS turbulence (k-ω SST); more physics later
- **Dimension:** 3D, unstructured finite volume
- **Language:** C++20, CMake, vcpkg
- **Performance:** multi-core CPU (OpenMP)
- **GUI:** Qt6 + VTK
- **Geometry:** STL/STEP import + auto meshing (Gmsh SDK); built-in primitives for testing
- **OS:** Windows
- **License:** personal use (GPL libs OK)

## Stack
| Area | Library |
|---|---|
| Linear algebra | Eigen |
| Linear solvers | AMGCL |
| Parallelism | OpenMP |
| Meshing / CAD | Gmsh SDK (OpenCASCADE) |
| GUI / 3D view | Qt6, VTK |
| Case files | JSON (nlohmann/json) |
| Results | VTU (ParaView-compatible) |
| Tests / CI | GoogleTest, GitHub Actions (Windows) |

## Phases
0. ✅ **Setup** – repo structure, CMake/vcpkg, CI, tests → details: [docs/phase0.md](docs/phase0.md)
1. ✅ **Mesh core** – unstructured 3D FV mesh (cells/faces/owner-neighbour), box/channel generator, `.msh` import, quality metrics → details: [docs/phase1.md](docs/phase1.md)
2. ✅ **Laminar solver** – collocated FV, SIMPLE (steady) / PIMPLE (transient), Rhie-Chow, BCs (inlet, outlet, wall, symmetry), non-orthogonal correction → details: [docs/phase2.md](docs/phase2.md)
   - Validation: lid-driven cavity (Ghia), Poiseuille channel (incl. non-orthogonal mesh); cylinder deferred to Phase 6
3. ✅ **Turbulence** – k-ω SST, wall functions, y+ → details: [docs/phase3.md](docs/phase3.md)
   - Validation: channel Re_τ 395 (DNS) and 2000 (wall functions); backward-facing step deferred (needs AMG)
4. ✅ **Case I/O** – JSON case files, VTU output, checkpoint/restart → details: [docs/phase4.md](docs/phase4.md), format: [docs/case_format.md](docs/case_format.md)
4b. ✅ **Performance** – AMG-preconditioned CG pressure solver (~5× faster, mesh-independent iteration counts), limited linear upwind, laminar backward-facing step validation → details: [docs/performance.md](docs/performance.md)
5. ✅ **GUI** (manual Windows test pending) – geometry viewer, primitives (box/cylinder/sphere), mesh settings, BC assignment, run with live residuals, post-processing (slices, contours, streamlines, vectors) → details: [docs/phase5.md](docs/phase5.md)
6. ✅ **Real geometry** – STL/STEP import, Gmsh auto meshing, 2D boundary layers, forces, Schäfer–Turek cylinder validation (3D prism layers deferred) → details: [docs/phase6.md](docs/phase6.md)
7. ✅ **MPI** – domain decomposition (RCB, ghost cells), distributed solvers, parallel CLI and GUI runs → details: [docs/phase7.md](docs/phase7.md)
8. ✅ **Heat transfer** – energy equation, Boussinesq buoyancy, thermal BCs, Nusselt output, de Vahl Davis validation → details: [docs/phase8.md](docs/phase8.md)
9. ✅ **Transient benchmark, parallel AMG, conjugate heat transfer** – BDF2, Schäfer–Turek 2D-2, distributed AMG, solid zones → details: [docs/phase9.md](docs/phase9.md)
10. **Later** – compressible, multiphase, GPU, geometry-module solid bodies

## Notes
- Development environment is Linux; code stays portable. Windows build/test via CI; GUI tested on user's PC.
- Manual Windows test checklist: [docs/windows_test.md](docs/windows_test.md)
