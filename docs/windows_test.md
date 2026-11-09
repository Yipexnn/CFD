# Windows test checklist

Step-by-step manual test of the desktop application on Windows. Tick each item; note
anything unexpected (screenshot + `Log` dock text) and send it back.

## 1. Prerequisites (once)
- [ ] Visual Studio 2022 Build Tools, workload "Desktop development with C++"
- [ ] Git, CLion
- [ ] vcpkg in `C:\dev\vcpkg`, `VCPKG_ROOT` set (README → Windows setup)
- [ ] MS-MPI runtime **10.1.12498.52** (`msmpisetup.exe`, link in README → MPI)
- [ ] Gmsh: unzip the Windows SDK/zip from gmsh.info, set `CFD_GMSH=C:\path\to\gmsh.exe`
      (or add its folder to `PATH`); restart CLion afterwards
- [ ] ~25 GB free disk space (first GUI build compiles Qt + VTK, 1–2 h)

## 2. Build
- [ ] `git pull` on `main`
- [ ] CLion → Settings → Build → CMake: enable preset `windows-msvc-gui-release`
      (or `windows-msvc-gui-dev` for debugging); toolchain = Visual Studio, amd64
- [ ] Reload CMake project (first time: long vcpkg build) → no errors
- [ ] Build target `cfd_gui` (also builds `cfd`)
- [ ] Run "All CTest" → all tests pass (MPI tests need MS-MPI)

## 3. Command line (Developer PowerShell in the repo)
```
build\windows-msvc-gui-release\src\app\cfd.exe --info
build\windows-msvc-gui-release\src\app\cfd.exe run examples\cavity.json
build\windows-msvc-gui-release\src\app\cfd.exe run examples\cylinder_2d.json
mpiexec -n 2 build\windows-msvc-gui-release\src\app\cfd.exe run examples\heated_cavity.json
```
- [ ] `--info` shows `MPI: yes`
- [ ] cavity converges; `examples\results\cavity\cavity.vtu` exists
- [ ] cylinder_2d: Gmsh runs, `forces.csv` last line Cx ≈ 5.59
- [ ] heated cavity on 2 processes: `heat.csv` xmin_Nu ≈ 4.56

## 4. GUI
Start `build\windows-msvc-gui-release\src\gui\cfd_gui.exe`.
- [ ] Window opens; New case shows a box mesh in the 3D view (rotate/zoom with the mouse)
- [ ] File → Open `examples\cavity.json` → Run → residual plot updates live, field |U| shown
- [ ] Post-processing: slice (axis/position), contours, vectors, streamlines, colour range,
      edges, opacity all react
- [ ] Stop button stops a running case; Run again restarts
- [ ] Open `examples\cylinder_2d.json` → Generate mesh (Gmsh) → mesh with boundary layer
      visible (enable Edges) → Run → field p
- [ ] Mesh tab "Geometry": add a primitive, change sizes, Generate mesh → updates
- [ ] Open `examples\heated_cavity.json` → Run → field T; Physics tab shows energy settings
- [ ] Open `examples\conjugate_plate.json` → Run → field T shows plate + fluid
- [ ] Processes = 2 → Run `examples\heated_cavity.json` → residuals update, final T shown
- [ ] File → Save case as… → reopen → same settings
- [ ] File → Save screenshot works

## 5. GPU (NVIDIA, optional)
- [ ] Install the CUDA Toolkit 12.x (with Visual Studio integration); restart CLion
- [ ] Enable preset `windows-msvc-gui-cuda-release`, reload CMake, build `cfd_gui`
- [ ] `cfd.exe --info` shows `GPU: cuda (device: <your GPU>)`
- [ ] Run "All CTest" (Gpu.* tests now run on the GPU)
- [ ] `cfd.exe run examples\cavity.json` prints `Linear solvers: GPU, cuda (device: ...)`
      (default `"device": "auto"`); converges like with `"device": "cpu"`; note both run times
- [ ] Same for `examples\sphere_3d.json` (needs Gmsh): Cd ≈ 1.07–1.09 with GPU and CPU
- [ ] GUI: Solver page → "Linear solvers": Auto shows "GPU found"; Run cavity with Auto / CPU /
      GPU → same result

## 6. Report
For each failed item: what you did, what happened, Log text, screenshot.
