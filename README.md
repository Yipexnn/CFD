# CFD

3D CFD desktop application in C++ (work in progress).

See [PLAN.md](PLAN.md) and [Phase 0 details](docs/phase0.md).

## Status

Phases 0–10 done: mesh core, laminar solver (SIMPLE/PISO), k-ω SST turbulence, JSON case files with VTU output and restart, AMG pressure solver, desktop GUI, geometry import and auto meshing (Gmsh) with force coefficients, MPI parallel runs, heat transfer with buoyancy and solid regions (conjugate), BDF2 transient benchmark, VOF two-phase flow, compressible flow (shocks).

## Windows setup (CLion)

1. Install Visual Studio 2022 Build Tools with workload "Desktop development with C++".
2. Install Git.
3. Install vcpkg:

   ```powershell
   git clone https://github.com/microsoft/vcpkg C:\dev\vcpkg
   C:\dev\vcpkg\bootstrap-vcpkg.bat -disableMetrics
   setx VCPKG_ROOT C:\dev\vcpkg
   ```

   Set the user environment variable `VCPKG_ROOT=C:\dev\vcpkg` (done by `setx` above), then restart CLion.
4. CLion → Settings → Build, Execution, Deployment → Toolchains: add **Visual Studio**, architecture `amd64`, and move it to the top (the default MinGW will NOT work).
5. CLion → Settings → Build, Execution, Deployment → CMake:
   - Enable presets `windows-msvc-debug` / `windows-msvc-release`.
   - Disable the default "Debug" profile.
   - Set the preset toolchain to Visual Studio.
6. Build target `cfd`. Run tests via the `cfd_tests` run configuration or "All CTest".

Note: the first configure builds vcpkg packages (a few minutes).

## Command-line build

Configure:

```sh
cmake --preset <preset>
```

Build:

```sh
cmake --build --preset <preset>
```

Test:

```sh
ctest --preset <preset>
```

- Replace `<preset>` with `windows-msvc-debug`, `windows-msvc-release`, `linux-gcc-debug` or `linux-gcc-release`.
- On Windows, run these commands in a "Developer PowerShell for VS 2022".
- On Linux, use the `linux-gcc-*` presets.
- `VCPKG_ROOT` must be set.

## Gmsh (geometry import / auto meshing)

Cases with `"mesh": {"type": "geometry", ...}` (primitives, STL, STEP) are meshed by the
Gmsh executable:
- Windows: download the Gmsh SDK/zip from https://gmsh.info, unpack it and either add the folder
  with `gmsh.exe` to `PATH` or set `CFD_GMSH=C:\path\to\gmsh.exe` (restart CLion / the GUI).
- Linux: `sudo apt install gmsh`.

## MPI (parallel runs)

- Windows: install the MS-MPI runtime **10.1.12498.52** (`msmpisetup.exe` from Microsoft,
  https://download.microsoft.com/download/7/2/7/72731ebb-b63c-4170-ade7-836966263a8f/msmpisetup.exe),
  then use a GUI preset (MPI enabled) or `windows-msvc-mpi-release`. vcpkg fetches the SDK.
- Linux: `sudo apt install libopenmpi-dev openmpi-bin` (the `linux-system-gui` preset enables MPI).
- Run: `mpiexec -n 4 cfd run case.json`, or in the GUI set "Processes" in the toolbar.
- Large meshes: `cfd decompose case.json 4` writes one mesh file per process; with
  `"decomposed_mesh": true` in the case each process reads only its part and results are
  written as `.pvtu` pieces (no process ever holds the whole mesh).
- Details and benchmarks: [docs/phase7.md](docs/phase7.md).

## GPU linear solvers (optional)

`CFD_GPU_BACKEND` = `none` (default) | `cuda` | `openmp`; enable per case with
`"solver": {"device": "gpu"}` or the GUI "Linear solvers" setting. `cfd --info` shows the backend.
- Windows + NVIDIA: install the CUDA Toolkit (12.x, with Visual Studio integration), then use
  the preset `windows-msvc-gui-cuda-release` (builds for the installed GPU: `native`).
- Linux CUDA: `-DCFD_GPU_BACKEND=cuda -DCMAKE_CUDA_ARCHITECTURES=native`.
- Linux OpenMP offload (GCC): `sudo apt install gcc-13-offload-nvptx`, then
  `-DCFD_GPU_BACKEND=openmp "-DCFD_OMP_OFFLOAD_FLAGS=-foffload=nvptx-none -fcf-protection=none -fno-stack-protector"`
  (MSVC has no offload: host threads).
- Without a device the same solvers run on host threads. Details: [docs/phase11.md](docs/phase11.md).

## Usage

```sh
cfd run examples/cavity.json        # run a case file (see docs/case_format.md)
cfd run examples/cylinder_2d.json   # cylinder benchmark (needs Gmsh), forces in forces.csv
cfd run examples/heated_cavity.json # natural convection (energy equation), Nusselt in heat.csv
cfd run examples/conjugate_plate.json  # conjugate heat transfer (fluid + solid plate)
cfd run examples/cylinder_2d_re100.json # vortex shedding (transient, needs Gmsh)
cfd run examples/dam_break.json     # two-phase VOF (water column collapse)
cfd run examples/sod_shock_tube.json # compressible shock tube
cfd run examples/forward_step_mach3.json # supersonic flow (needs Gmsh)
cfd run examples/conjugate_block_2d.json # meshed solid body (CHT, needs Gmsh)
cfd mesh-box 10 10 10               # create a box mesh and print statistics
cfd check-mesh mesh.msh             # check a Gmsh 4.1 mesh
cfd run-cavity 32 100               # quick lid-driven cavity demo
cfd --version | --info | --help
```

Results are written as `.vtu` files (open with ParaView) plus `residuals.csv` and a
`checkpoint.bin` for restarts (`"restart": "results/<case>/checkpoint.bin"`).

## Desktop GUI (`cfd_gui`)

Set up, run and post-process cases interactively (Qt + VTK):
- left: case editor (mesh, physics, boundaries, solver, output)
- centre: 3D view; right: post-processing (field, slice, contours, vectors, streamlines)
- bottom: live residual plot and log
- File → Open/Save case (JSON); Run → Generate mesh / Run / Stop

Build it with the GUI presets (vcpkg builds Qt 6 and VTK the first time — expect about 1–2 hours (release-only Qt/VTK, no Qt Quick) and
~20 GB of disk; later builds reuse the binaries):
- CLion: enable the preset `windows-msvc-gui-dev` (RelWithDebInfo, debuggable) or `windows-msvc-gui-release` (Settings → Build, Execution, Deployment → CMake), then build the `cfd_gui` target.
- Command line (Developer PowerShell): `cmake --preset windows-msvc-gui-release` then `cmake --build --preset windows-msvc-gui-release`.
- Linux with system packages: `sudo apt install libvtk9-dev libvtk9-qt-dev qtbase5-dev libeigen3-dev nlohmann-json3-dev libgtest-dev libopenmpi-dev openmpi-bin`, then `cmake --preset linux-system-gui && cmake --build --preset linux-system-gui`.

Scriptable options (also used by CI): `cfd_gui case.json --run --show slice,vectors,streamlines --field p --screenshot view.png --quit`.

## Project layout

- `src/core` – core library (`cfd::core`)
- `src/app` – CLI application (`cfd`)
- `src/gui` – desktop application (`cfd_gui`)
- `src/case` – case files and runner
- `src/fv` – finite-volume solver
- `src/mesh` – mesh, Gmsh reader, VTU writer
- `src/geometry` – geometry (primitives, STL/STEP) → Gmsh auto meshing
- `src/parallel` – MPI wrapper, mesh decomposition, halo exchange
- `src/gpu` – accelerated Krylov solvers (CUDA / OpenMP offload / host)
- `validation` – benchmark programs
- `tests` – unit tests
- `cmake` – CMake helper modules
- `docs` – documentation
