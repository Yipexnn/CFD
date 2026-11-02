# Phase 0 – Project Setup (detailed plan)

Goal: a buildable, tested, CI-checked C++20 skeleton for the CFD project.
**Scope:** core library + CLI app + tests + CI. **No GUI** (Qt/VTK come in Phase 5).

---

## 1. Decisions

| Topic | Choice |
|---|---|
| Language | C++20, no compiler extensions |
| Build | CMake ≥ 3.25, Ninja, `CMakePresets.json` |
| Packages | vcpkg, manifest mode (`vcpkg.json`, `builtin-baseline` pinned) |
| Windows toolchain | CLion with **Visual Studio (MSVC) toolchain** (not bundled MinGW) |
| Windows triplet | `x64-windows-static-md` (static libs, dynamic CRT → no DLL copying) |
| Phase 0 deps | `eigen3`, `nlohmann-json`, `gtest` |
| Parallelism | OpenMP (`find_package(OpenMP)`; MSVC = OpenMP 2.0) |
| Tests | GoogleTest + `gtest_discover_tests` + CTest |
| CI | GitHub Actions: Windows/MSVC (required) + Ubuntu/GCC (fast feedback) + clang-format check |
| Version | `0.1.0` (from `project()` → generated header) |

---

## 2. Target repository layout

```
CFD/
├─ .github/workflows/ci.yml
├─ cmake/CfdTargetDefaults.cmake
├─ docs/phase0.md
├─ src/
│  ├─ core/
│  │  ├─ CMakeLists.txt
│  │  ├─ include/cfd/core/
│  │  │  ├─ version.hpp.in
│  │  │  ├─ types.hpp
│  │  │  └─ build_info.hpp
│  │  └─ src/build_info.cpp
│  └─ app/
│     ├─ CMakeLists.txt
│     └─ main.cpp
├─ tests/
│  ├─ CMakeLists.txt
│  ├─ test_build_info.cpp
│  └─ test_dependencies.cpp
├─ .clang-format
├─ .editorconfig
├─ .gitignore
├─ CMakeLists.txt
├─ CMakePresets.json
├─ vcpkg.json
├─ CLAUDE.md
├─ PLAN.md
└─ README.md
```

Targets:
- `cfd_core`: static library, alias `cfd::core`
- `cfd`: CLI executable
- `cfd_tests`: GoogleTest executable

---

## 3. Coding conventions (apply to all phases)

- Namespace `cfd::<module>` (e.g. `cfd::core`). Includes are written as `#include <cfd/core/types.hpp>`.
- Files: `snake_case.hpp/.cpp`. Types: `PascalCase`. Functions/variables: `snake_case`. Private members: trailing `_`. Constants: `k_name`.
- `#pragma once`. No `using namespace` in headers.
- Base types (in `types.hpp`): `Scalar = double`, `Label = std::int32_t`, `Vec3 = Eigen::Vector3d`.
- OpenMP loops: **signed** loop index (`Label`), no `collapse`, simple reductions only (MSVC OpenMP 2.0).
- Every new source file gets a unit test. No warnings allowed (CI uses warnings-as-errors).
- Formatting: `.clang-format` (run before commit).

---

## 4. Tasks

Agent legend (from CLAUDE.md):
- **H**: Haiku 5.5 subagent, effort medium (grunt work, simple single-file tasks)
- **O**: orchestrator itself (complex tasks). Never use Sonnet.

Each task produces **one file** (T-11 is a header/impl pair handled sequentially). The orchestrator reviews every result.

### Wave 1 – independent config files (parallel)

**T-01 `.gitignore`** (H)
- Ignore: `build/`, `cmake-build-*/`, `out/`, `.idea/`, `.vs/`, `.vscode/`, `vcpkg_installed/`, `CMakeUserPresets.json`, `compile_commands.json`, `*.user`, `results/`, `*.vtu`, `*.vtk`, `Testing/`.
- Done when: `git status` after a local build shows no build output.

**T-02 `.clang-format`** (H)
- `BasedOnStyle: LLVM`, `IndentWidth: 4`, `ColumnLimit: 100`, `PointerAlignment: Left`, `AllowShortFunctionsOnASingleLine: Inline`, `IncludeBlocks: Regroup`, `SortIncludes: CaseSensitive`, `NamespaceIndentation: None`, `AccessModifierOffset: -4`.
- Done when: `clang-format --dry-run` on any file parses without error.

**T-03 `.editorconfig`** (H)
- UTF-8, LF, final newline, trim trailing whitespace; 4 spaces for C++/CMake; 2 spaces for JSON/YAML/MD.

**T-04 `vcpkg.json`** (O)
- First get the current vcpkg commit: `git ls-remote https://github.com/microsoft/vcpkg HEAD`.
- Content: `name: "cfd"`, `version: "0.1.0"`, `dependencies: ["eigen3", "nlohmann-json", "gtest"]`, `builtin-baseline: "<sha>"`.

**T-05 `cmake/CfdTargetDefaults.cmake`** (H)
- Defines `function(cfd_target_defaults target)`:
  - MSVC: `/W4 /permissive- /utf-8 /Zc:__cplusplus`; defines `NOMINMAX`, `_USE_MATH_DEFINES`.
  - GCC/Clang: `-Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wunused -Woverloaded-virtual -Wnull-dereference -Wdouble-promotion -Wformat=2`.
  - If `CFD_WARNINGS_AS_ERRORS`: MSVC `/WX`, others `-Werror`.
  - Sets `cxx_std_20` via `target_compile_features(... PUBLIC cxx_std_20)`.
- Use generator expressions or `if(MSVC)` branches; no global `add_compile_options`.

### Wave 2 – build system (after Wave 1)

**T-06 `CMakeLists.txt` (root)** (O)
- `cmake_minimum_required(VERSION 3.25)`, `project(cfd VERSION 0.1.0 LANGUAGES CXX)`.
- `CMAKE_CXX_STANDARD 20`, `CMAKE_CXX_STANDARD_REQUIRED ON`, `CMAKE_CXX_EXTENSIONS OFF`, `CMAKE_EXPORT_COMPILE_COMMANDS ON`.
- Options: `CFD_BUILD_APP` (ON), `CFD_BUILD_TESTS` (ON), `CFD_ENABLE_OPENMP` (ON), `CFD_WARNINGS_AS_ERRORS` (OFF).
- Append `${CMAKE_SOURCE_DIR}/cmake` to `CMAKE_MODULE_PATH`; `include(CfdTargetDefaults)`.
- `find_package(Eigen3 CONFIG REQUIRED)`, `find_package(nlohmann_json CONFIG REQUIRED)`, `find_package(OpenMP REQUIRED COMPONENTS CXX)` if OpenMP is enabled.
- `add_subdirectory(src/core)`; app/tests depending on options; tests: `enable_testing()` before `add_subdirectory(tests)`.
- Prevent in-source builds (fatal error if source dir == binary dir).

**T-07 `CMakePresets.json`** (O)
- Schema `version: 6`.
- Hidden `base`: generator `Ninja`, `binaryDir: ${sourceDir}/build/${presetName}`, `toolchainFile: $env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake`.
- Hidden `windows-base` (inherits `base`): condition `hostSystemName == Windows`, `CMAKE_CXX_COMPILER: cl`, `VCPKG_TARGET_TRIPLET: x64-windows-static-md`, `architecture: {value: x64, strategy: external}`.
- Hidden `linux-base` (inherits `base`): condition `hostSystemName == Linux`, `CMAKE_CXX_COMPILER: g++`.
- Configure presets: `windows-msvc-debug`, `windows-msvc-release`, `linux-gcc-debug`, `linux-gcc-release` (`CMAKE_BUILD_TYPE` Debug/Release).
- Matching `buildPresets` and `testPresets` with the same names (`output.outputOnFailure: true`).
- Done when: `cmake --list-presets` works on Linux and shows only the linux presets.

**T-08 `src/core/include/cfd/core/version.hpp.in`** (H)
- Template for `configure_file`: `namespace cfd::core { inline constexpr int k_version_major = @PROJECT_VERSION_MAJOR@; ... minor, patch; inline constexpr std::string_view k_version_string = "@PROJECT_VERSION@"; }`.

**T-09 `src/core/include/cfd/core/types.hpp`** (H)
- `Scalar = double`, `Label = std::int32_t`, `Vec3 = Eigen::Matrix<Scalar,3,1>`.
- `static_assert`s on sizes/signedness.

**T-10 `src/core/include/cfd/core/build_info.hpp`** (H)
- `struct BuildInfo { std::string version; std::string compiler; std::string build_type; bool openmp_enabled; int openmp_max_threads; std::string eigen_version; };`
- `BuildInfo build_info();` and `std::string build_info_string();` (multi-line, human-readable).

**T-11 `src/core/src/build_info.cpp`** (H, after T-10)
- Compiler detection via `_MSC_VER` / `__clang__` / `__GNUC__` macros (incl. version numbers).
- Build type from compile definition `CFD_BUILD_TYPE` (set by CMake).
- OpenMP: `#ifdef _OPENMP` → `omp_get_max_threads()`, else `false` / `1`.
- Eigen version from `EIGEN_WORLD_VERSION.EIGEN_MAJOR_VERSION.EIGEN_MINOR_VERSION`.

**T-12 `src/core/CMakeLists.txt`** (O)
- `configure_file(include/cfd/core/version.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/generated/include/cfd/core/version.hpp @ONLY)`.
- `add_library(cfd_core STATIC src/build_info.cpp)` + `add_library(cfd::core ALIAS cfd_core)`.
- Public include dirs: `include/` and `generated/include/`.
- Link PUBLIC `Eigen3::Eigen`, `nlohmann_json::nlohmann_json`; PUBLIC `OpenMP::OpenMP_CXX` if enabled.
- PRIVATE definition `CFD_BUILD_TYPE="$<CONFIG>"`. Call `cfd_target_defaults(cfd_core)`.

**T-13 `src/app/main.cpp`** (H)
- No CLI library: parse `argv` by hand.
- `cfd --version` prints `cfd 0.1.0`; `cfd --info` prints `build_info_string()`; `--help` / no arguments print usage. Unknown argument → message to stderr, exit code 1.

**T-14 `src/app/CMakeLists.txt`** (H)
- `add_executable(cfd main.cpp)`, link `cfd::core`, `cfd_target_defaults(cfd)`.

**T-15 `tests/CMakeLists.txt`** (H)
- `find_package(GTest CONFIG REQUIRED)`; `add_executable(cfd_tests test_build_info.cpp test_dependencies.cpp)`.
- Link `cfd::core`, `GTest::gtest_main`; `cfd_target_defaults(cfd_tests)`.
- `include(GoogleTest)` + `gtest_discover_tests(cfd_tests DISCOVERY_MODE PRE_TEST)`.

**T-16 `tests/test_build_info.cpp`** (H)
- The version string equals `"<major>.<minor>.<patch>"` built from the version constants.
- `build_info()` returns a non-empty compiler, version and eigen_version, and `openmp_max_threads >= 1`.
- `build_info_string()` contains the version.

**T-17 `tests/test_dependencies.cpp`** (H)
- Eigen: `Vec3` cross product of the x and y unit vectors equals the z unit vector.
- nlohmann_json: an object with a double/int/string round-trips through `dump()`/`parse()`.
- OpenMP: a parallel `reduction(+:sum)` over `Label i = 0..999'999` equals the serial sum (guard with `#ifdef _OPENMP`, otherwise `GTEST_SKIP()`).

### Wave 3 – CI & docs (after Wave 2 builds locally)

**T-18 `.github/workflows/ci.yml`** (O)
- Triggers: `push` to `main` and `claude/**`, `pull_request`, `workflow_dispatch`.
- Job `build-test`, matrix:
  - `windows-latest` + preset `windows-msvc-release`
  - `ubuntu-latest` + preset `linux-gcc-release`
- Steps:
  1. `actions/checkout@v4`
  2. Windows only: `ilammy/msvc-dev-cmd@v1` (arch x64)
  3. `lukka/get-cmake@latest` (CMake + Ninja)
  4. vcpkg (`shell: bash`): read the baseline with `jq -r '."builtin-baseline"' vcpkg.json`; `git clone https://github.com/microsoft/vcpkg "$RUNNER_TEMP/vcpkg"`; `git -C ... checkout <sha>`; run bootstrap (`.bat` on Windows, `.sh` on Linux); export `VCPKG_ROOT` via `$GITHUB_ENV`.
  5. Binary cache: `VCPKG_DEFAULT_BINARY_CACHE=${{ github.workspace }}/.vcpkg-cache` (create the dir); `actions/cache@v4` keyed on `runner.os` + `hashFiles('vcpkg.json')`.
  6. `cmake --preset <p> -DCFD_WARNINGS_AS_ERRORS=ON`
  7. `cmake --build --preset <p>`
  8. `ctest --preset <p>`
  9. Smoke test: `build/<p>/src/app/cfd --info`
- Job `format` (ubuntu): `clang-format --dry-run --Werror` on all `*.hpp *.cpp *.hpp.in` under `src/` and `tests/`.

**T-19 `README.md`** (H)
- Short project description + link to `PLAN.md`.
- **Windows setup (CLion):**
  1. Install *Visual Studio 2022 Build Tools* with the workload "Desktop development with C++".
  2. Install Git.
  3. vcpkg: `git clone https://github.com/microsoft/vcpkg C:\dev\vcpkg`, run `C:\dev\vcpkg\bootstrap-vcpkg.bat`, set the user environment variable `VCPKG_ROOT=C:\dev\vcpkg`, restart CLion.
  4. CLion → Settings → Build, Execution, Deployment → Toolchains: add **Visual Studio**, architecture `amd64`, move it to the top.
  5. CLion → Settings → CMake: enable preset `windows-msvc-debug` (and `-release`), disable the default "Debug" profile, set the preset's toolchain to Visual Studio.
  6. Build target `cfd`; run tests via the `cfd_tests` configuration or *All CTest*.
- Command-line build (Linux and a Windows Developer PowerShell): `cmake --preset ...`, `cmake --build --preset ...`, `ctest --preset ...`.
- Note: the first configure builds the vcpkg packages (a few minutes).

**T-20 `PLAN.md`** (O)
- Link `docs/phase0.md` from Phase 0; mark Phase 0 done once the Definition of Done is met.

---

## 5. Orchestrator workflow

1. Work on branch `claude/loving-ramanujan-qt2i61` (create it from `main`).
2. Run Wave 1 → review → Wave 2 → review → local build → Wave 3 → review.
3. **Subagent prompt template** (one task per agent):
   ```
   Create exactly one file: <path>. Do not touch other files. Do not commit.
   Spec: <copy the task section from docs/phase0.md>
   Conventions: <copy section 3>
   Context: repo layout in docs/phase0.md §2. Report back the file content summary.
   ```
   - H tasks: `model: haiku`, `effort: medium`. O tasks: the orchestrator writes them itself (no Sonnet).
4. **Review per file:** matches spec, conventions, no extra files, clang-format clean.
5. **Local verification (Linux sandbox):**
   - `git clone https://github.com/microsoft/vcpkg <scratchpad>/vcpkg && <scratchpad>/vcpkg/bootstrap-vcpkg.sh -disableMetrics`
   - `export VCPKG_ROOT=<scratchpad>/vcpkg`
   - `cmake --preset linux-gcc-debug -DCFD_WARNINGS_AS_ERRORS=ON && cmake --build --preset linux-gcc-debug && ctest --preset linux-gcc-debug`
   - `./build/linux-gcc-debug/src/app/cfd --info`
   - `clang-format --dry-run --Werror` on all sources.
6. Commit per wave (clear messages), push to the branch, check CI (Windows + Linux + format).
7. When CI is green: **ask the user** before merging into `main`.
8. Stop early if usage or context gets close to 90%.

---

## 6. Definition of Done

- [ ] CI is green: Windows/MSVC, Ubuntu/GCC, format check; warnings-as-errors on.
- [ ] `cfd --version` → `cfd 0.1.0`; `cfd --info` shows compiler, build type, OpenMP threads, Eigen version.
- [ ] All tests pass via `ctest`.
- [ ] The user has built and run it in CLion on Windows with the `windows-msvc-debug` preset (manual check).
- [ ] `PLAN.md` updated, merged into `main`.

## 7. Risks / notes

- CLion's default toolchain is MinGW; it **must** be switched to Visual Studio, or presets with `cl` fail.
- `VCPKG_ROOT` must be set before CLion starts (or set it in the CLion preset environment).
- MSVC supports only OpenMP 2.0: signed loop indices, no `collapse`, no task constructs.
- The first vcpkg build in CI is slow; later runs use the cache.
