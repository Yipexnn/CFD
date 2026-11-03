# Phase 5 – Desktop GUI (Qt + VTK)

Goal: a desktop application to set up, run and post-process cases without editing JSON by hand.

## Architecture
- **`cfd_gui`** executable (`src/gui`), Qt Widgets plus a VTK render view (`QVTKOpenGLNativeWidget`)
  - builds against **Qt 6** (Windows/vcpkg) and **Qt 5.15** (Linux system packages)
  - option `CFD_BUILD_GUI` (default OFF); vcpkg manifest feature `gui` (qtbase, vtk[qt])
- **Main window**
  - left dock **Case editor**:
    - mesh: box (cells, extent, grading, patch types) or Gmsh file
    - physics: ν, body force, turbulence
    - boundary table: type, velocity, pressure, turbulence inlet values
    - solver and output settings
  - centre: **3D view** of the mesh / results
  - right dock **Post-processing**:
    - field (\|U\|, Ux/Uy/Uz, p, k, ω, ν_t, y+)
    - display: boundary surface, slice plane (x/y/z + position), contours, vector glyphs, streamlines
    - mesh edges, colour bar, range
  - bottom dock: **live residual plot** (log scale) + log
  - File: New / Open / Save case (JSON), export VTU, screenshot
  - Run / Stop: solver in a worker thread (`CaseRunner` + progress callback). The view refreshes with live field snapshots during the run.
- **Non-GUI pieces** (unit tested):
  - `save_case` / `to_json` in `cfd::cases`, round-trip with `parse_case`
  - mesh + fields → `vtkUnstructuredGrid` conversion (polyhedral cells)

## Build / CI
- Windows GUI dependencies: release-only triplet `x64-windows-static-md-release` (host = target), VTK overlay port without Qt Quick (`vcpkg/ports/vtk`). The first build takes ~2.5 h on CI and is cached afterwards (later builds ~5 min).
- presets:
  - `windows-msvc-gui-{dev,release}` (vcpkg feature `gui`; dev = RelWithDebInfo)
  - `linux-system-gui` (system Qt5/VTK/Eigen/json/gtest, no vcpkg)
- CI jobs:
  - `linux-gui`: apt packages, fast
  - `windows-gui` (own workflow, never cancelled mid-build): vcpkg Qt6 + VTK, cached.
- Local checks run headless under Xvfb with screenshots.

## Definition of Done
- [x] `cfd_gui` builds on Windows (Qt6, vcpkg) and Linux (Qt5, system packages)
- [x] Open/edit/save case files; box mesh preview; Gmsh import
- [x] Run/stop with live residuals and field updates
- [x] Post-processing: surface, slice, contours, vectors, streamlines, colour bar
- [x] Unit tests for case serialization and VTK conversion
- [ ] User tested on Windows (manual)
