#pragma once

#include <string>

namespace cfd::gpu {

// Square sparse matrix in CSR format (host memory, 0-based, sorted columns not required).
struct CsrView {
    int n = 0;
    int nnz = 0;
    const int* row_ptr = nullptr; // n + 1 entries
    const int* col = nullptr;     // nnz entries
    const double* val = nullptr;  // nnz entries
};

// Jacobi-preconditioned Krylov methods: CG (symmetric positive definite) and BiCGStab.
enum class Method { Cg, BiCgStab };

struct SolveControl {
    double abs_tolerance = 1e-10; // stop when |b - A x|_2 <= abs_tolerance
    int max_iterations = 1000;
};

struct SolveStats {
    int iterations = 0;
    double residual = 0.0; // |b - A x|_2 at exit
    bool converged = false;
    bool on_device = false; // ran on a GPU
};

// Solves A x = b in place (x = initial guess) on the accelerator selected at build time
// (CFD_GPU_BACKEND): "cuda" uses CUDA kernels when a CUDA device is present; "openmp" uses
// OpenMP target offload (host threads if no device or no offload compiler support); without a
// backend (or no device) the same algorithm runs on the host with OpenMP threads.
SolveStats solve(const CsrView& a, const double* b, double* x, Method method,
                 const SolveControl& control);

// Same algorithms, always on host threads (independent of the backend).
SolveStats solve_host(const CsrView& a, const double* b, double* x, Method method,
                      const SolveControl& control);

// Backend compiled in: "cuda", "openmp-offload" or "host".
[[nodiscard]] std::string backend_name();
// True if the backend found an accelerator device at run time.
[[nodiscard]] bool device_available();
// One-line description, e.g. "cuda (device: NVIDIA GeForce RTX 3070)".
[[nodiscard]] std::string device_description();

namespace detail {
// Host / OpenMP-offload implementation (always available).
SolveStats solve_openmp(const CsrView& a, const double* b, double* x, Method method,
                        const SolveControl& control);
SolveStats solve_openmp(const CsrView& a, const double* b, double* x, Method method,
                        const SolveControl& control, bool allow_device);
bool openmp_device_available();
#ifdef CFD_GPU_CUDA
SolveStats solve_cuda(const CsrView& a, const double* b, double* x, Method method,
                      const SolveControl& control);
bool cuda_device_available();
std::string cuda_device_name();
#endif
} // namespace detail

} // namespace cfd::gpu
