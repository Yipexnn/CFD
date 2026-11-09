#pragma once

#include <memory>
#include <string>
#include <vector>

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

// Algebraic multigrid hierarchy for an AMG-preconditioned CG (built on the host, e.g. by
// fv::AmgPreconditioner): per level the operator a, and except on the coarsest level the
// restriction r (n_{l+1} x n_l) and prolongation p (n_l x n_{l+1}); coarse_inverse is the dense
// row-major inverse of the coarsest operator. V-cycle: damped Jacobi smoothing (symmetric).
struct AmgLevelView {
    CsrView a;
    CsrView p;
    CsrView r;
};
struct AmgView {
    std::vector<AmgLevelView> levels;
    const double* coarse_inverse = nullptr;
    int smooth_sweeps = 2;
    double jacobi_weight = 0.7;
};

// CG preconditioned by one AMG V-cycle per iteration; A = levels[0].a. Device as for solve().
SolveStats solve_amg_cg(const AmgView& amg, const double* b, double* x,
                        const SolveControl& control);
SolveStats solve_amg_cg_host(const AmgView& amg, const double* b, double* x,
                             const SolveControl& control);

// Fixed sparsity pattern of an LDU matrix in CSR form: positions of the diagonal and of the
// upper / lower face coefficients in the CSR value array.
struct LduPattern {
    int n = 0;
    int n_faces = 0;
    int nnz = 0;
    const int* row_ptr = nullptr;
    const int* col = nullptr;
    const int* diag_pos = nullptr;
    const int* upper_pos = nullptr;
    const int* lower_pos = nullptr;
};

// Persistent linear solver for repeated solves on one pattern (e.g. one equation of a time
// loop). With a CUDA device the pattern, work vectors and AMG coarse levels stay on the device
// and the CSR values are assembled there from the uploaded LDU coefficients; otherwise the
// values are assembled on the host and solved with the OpenMP / host path.
class DeviceSolver {
public:
    explicit DeviceSolver(const LduPattern& pattern);
    ~DeviceSolver();
    DeviceSolver(const DeviceSolver&) = delete;
    DeviceSolver& operator=(const DeviceSolver&) = delete;
    DeviceSolver(DeviceSolver&&) = delete;
    DeviceSolver& operator=(DeviceSolver&&) = delete;

    // AMG hierarchy for use_amg solves: levels[0].a is not used (the assembled matrix is), its
    // p / r and all coarser levels are copied now. Call again after an AMG rebuild.
    void set_amg(const AmgView& amg);
    // Assembles A from diag (n) / upper, lower (n_faces) and solves A x = b (x: initial guess).
    SolveStats solve(const double* diag, const double* upper, const double* lower, const double* b,
                     double* x, Method method, bool use_amg, const SolveControl& control);
    [[nodiscard]] bool on_device() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Finite-volume assembly of the implicit upwind convection + diffusion coefficients of the
// internal faces of an LDU matrix: upper = min(F, 0) - G, lower = -max(F, 0) - G,
// diag[owner] += max(F, 0) + G, diag[neighbour] += -min(F, 0) + G, divergence += +-F, with
// G = gamma * delta_coeff; the implicit convection uses convection * F (0: pure Laplacian, e.g.
// the pressure equation, while divergence still sums F). diag and divergence are accumulated
// (not cleared). On a CUDA device
// or with OpenMP offload the face loop runs on the accelerator (atomic accumulation); the host
// path is the serial loop (deterministic).
void assemble_convection_diffusion(int n_cells, int n_faces, const int* owner, const int* neighbour,
                                   const double* flux, const double* gamma,
                                   const double* delta_coeff, double convection, double* upper,
                                   double* lower, double* diag, double* divergence);

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
SolveStats solve_amg_openmp(const AmgView& amg, const double* b, double* x,
                            const SolveControl& control, bool allow_device);
void assemble_openmp(int n_cells, int n_faces, const int* owner, const int* neighbour,
                     const double* flux, const double* gamma, const double* delta_coeff,
                     double convection, double* upper, double* lower, double* diag,
                     double* divergence);
#ifdef CFD_GPU_CUDA
SolveStats solve_amg_cuda(const AmgView& amg, const double* b, double* x,
                          const SolveControl& control);
void assemble_cuda(int n_cells, int n_faces, const int* owner, const int* neighbour,
                   const double* flux, const double* gamma, const double* delta_coeff,
                   double convection, double* upper, double* lower, double* diag,
                   double* divergence);
void* cuda_system_create(const LduPattern& pattern);
void cuda_system_destroy(void* handle);
void cuda_system_set_amg(void* handle, const AmgView& amg);
SolveStats cuda_system_solve(void* handle, const double* diag, const double* upper,
                             const double* lower, const double* b, double* x, Method method,
                             bool use_amg, const SolveControl& control);
SolveStats solve_cuda(const CsrView& a, const double* b, double* x, Method method,
                      const SolveControl& control);
bool cuda_device_available();
std::string cuda_device_name();
#endif
} // namespace detail

} // namespace cfd::gpu
