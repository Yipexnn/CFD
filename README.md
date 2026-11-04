# CFD

3D CFD desktop application in C++ (work in progress).

See [PLAN.md](PLAN.md) and [Phase 0 details](docs/phase0.md).

## Status

Phases 0–6 done: mesh core, laminar solver (SIMPLE/PISO), k-ω SST turbulence, JSON case files with VTU output and restart, AMG pressure solver, desktop GUI, geometry import and auto meshing (Gmsh) with force coefficients.

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

## Usage

```sh
cfd run examples/cavity.json        # run a case file (see docs/case_format.md)
cfd run examples/cylinder_2d.json   # cylinder benchmark (needs Gmsh), forces in forces.csv
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
- Linux with system packages: `sudo apt install libvtk9-dev libvtk9-qt-dev qtbase5-dev libeigen3-dev nlohmann-json3-dev libgtest-dev`, then `cmake --preset linux-system-gui && cmake --build --preset linux-system-gui`.

Scriptable options (also used by CI): `cfd_gui case.json --run --show slice,vectors,streamlines --field p --screenshot view.png --quit`.

## Project layout

- `src/core` – core library (`cfd::core`)
- `src/app` – CLI application (`cfd`)
- `src/gui` – desktop application (`cfd_gui`)
- `src/case` – case files and runner
- `src/fv` – finite-volume solver
- `src/mesh` – mesh, Gmsh reader, VTU writer
- `src/geometry` – geometry (primitives, STL/STEP) → Gmsh auto meshing
- `validation` – benchmark programs
- `tests` – unit tests
- `cmake` – CMake helper modules
- `docs` – documentation
