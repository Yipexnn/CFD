# CFD

3D CFD desktop application in C++ (work in progress).

See [PLAN.md](PLAN.md) and [Phase 0 details](docs/phase0.md).

## Status

Phase 0 – project setup (core library, CLI app `cfd`, unit tests, CI).

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

## Usage

```sh
cfd --version
cfd --info
cfd --help
```

## Project layout

- `src/core` – core library (`cfd::core`)
- `src/app` – CLI application (`cfd`)
- `tests` – unit tests
- `cmake` – CMake helper modules
- `docs` – documentation
