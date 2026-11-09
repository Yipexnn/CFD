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
10. ✅ **Geometry solids, multiphase, compressible** – solid bodies in Gmsh meshing, VOF two-phase flow (dam break), density-based compressible solver (Sod, oblique shock), Windows test checklist → details: [docs/phase10.md](docs/phase10.md)
11. ✅ **Implicit/turbulent compressible, multiphase turbulence, contact angles, GPU** (GPU hardware test pending) – LU-SGS + Jacobian-free Newton–Krylov, Venkatakrishnan limiter, compressible k-ω SST (turbulent flat plate), mixture SST for VOF, static contact angle (droplet spreading), GPU Krylov solvers (CUDA, OpenMP offload) → details: [docs/phase11.md](docs/phase11.md)
12. ✅ **Variable properties, radiation, 3D prism layers** – PropertyLaw (ν(T), α(T), Sutherland μ(T)), P1 radiation (radiative equilibrium validation), 3D prism boundary layers (sphere Re 100) → details: [docs/phase12.md](docs/phase12.md)
13. ✅ **GUI settings, sphere study, radiation + solids, parallel mesh distribution, GPU AMG** → details: [docs/phase13.md](docs/phase13.md)
14. ✅ **Compressible far field, decomposed meshes, GPU assembly, sphere study** → details: [docs/phase14.md](docs/phase14.md)
15. ✅ **Sphere drag resolved (least-squares gradients), GPU face assembly** → details: [docs/phase15.md](docs/phase15.md)
16. ✅ **CPU/GPU auto selection, FV kernels on the GPU, refine boxes, sphere Re 20 / 300** → details: [docs/phase16.md](docs/phase16.md)
17. **Later** – GPU hardware test (manual), Rhie-Chow / turbulence terms on the GPU, cause of the sphere low-Re drag deficit (Re 20: −4…6 %)

## Notes
- Development environment is Linux; code stays portable. Windows build/test via CI; GUI tested on user's PC.
- Manual Windows test checklist: [docs/windows_test.md](docs/windows_test.md)
