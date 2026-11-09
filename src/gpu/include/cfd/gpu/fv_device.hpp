#pragma once

// Finite-volume kernels on the accelerator: cell gradients (Green-Gauss, weighted least
// squares) and the explicit face corrections (linear upwind, limiter, non-orthogonal
// correction) of the scalar and momentum equations. The mesh is copied to the device once;
// each call moves only the fields. All loops gather over the faces of a cell in ascending face
// order, so the results equal the serial CPU face loops (no atomics, deterministic). Without a
// device the same kernels run on host threads.

#include <memory>

namespace cfd::gpu {

// Host arrays of the mesh (copied by DeviceFv's constructor). Faces [0, n_internal_faces) are
// internal; vectors are 3 doubles per entry.
struct FvMeshView {
    int n_cells = 0;
    int n_internal_faces = 0;
    int n_faces = 0;
    const int* owner = nullptr;          // n_faces
    const int* neighbour = nullptr;      // n_internal_faces
    const int* cell_offsets = nullptr;   // n_cells + 1: faces of each cell (CSR) ...
    const int* cell_faces = nullptr;     // ... in ascending face order
    const double* weight = nullptr;      // n_internal_faces: phi_f = w phi_P + (1 - w) phi_N
    const double* sf = nullptr;          // 3 n_faces: face area vectors
    const double* face_centre = nullptr; // 3 n_faces
    const double* cell_centre = nullptr; // 3 n_cells
    const double* volume = nullptr;      // n_cells
    const double* correction = nullptr;  // 3 n_faces: non-orthogonal correction vectors
    const double* lsq_d = nullptr;       // 3 n_faces: d / |d|^2 (d = C_N - C_P or C_f - C_P)
    const double* lsq_inverse = nullptr; // 9 n_cells: (sum d d^T / |d|^2)^-1, column-major
};

class DeviceFv {
public:
    explicit DeviceFv(const FvMeshView& mesh);
    ~DeviceFv();
    DeviceFv(const DeviceFv&) = delete;
    DeviceFv& operator=(const DeviceFv&) = delete;

    // Gradient of a cell field phi (n_cells) with boundary face values bv (n_boundary):
    // grad = 3 n_cells.
    void gradient(const double* phi, const double* bv, bool least_squares, double* grad);
    // Gradient of a vector field u (3 n_cells) with boundary values ub (3 n_boundary):
    // grad = 9 n_cells, column-major 3x3 per cell with grad(j, i) = du_i / dx_j. u, ub and the
    // gradient stay on the device for momentum_corrections.
    void vector_gradient(const double* u, const double* ub, bool least_squares, double* grad);
    // Explicit internal-face fluxes of a scalar convection-diffusion equation, accumulated into
    // source (n_cells): e_f = -F_f grad_up . (x_f - x_up) [linear_upwind]
    //                         + gamma_f k_f . (w grad_P + (1 - w) grad_N),
    // source_P += e_f, source_N -= e_f; the gradient of phi is computed on the device.
    void scalar_corrections(const double* phi, const double* bv, bool least_squares,
                            const double* flux, const double* gamma, bool linear_upwind,
                            double* source);
    // Explicit internal-face fluxes of the momentum equation with the field and gradient of the
    // last vector_gradient call, accumulated into source (3 n_cells):
    //   -F_f lim_up * (grad_up^T (x_f - x_up)) [linear_upwind; lim = Barth-Jespersen limiter
    //   per component if limited] + nu_f (grad_f^T k_f) [+ nu_f grad_f Sf if transpose_term],
    // grad_f = w grad_P + (1 - w) grad_N; nu_face: n_internal_faces.
    void momentum_corrections(const double* flux, const double* nu_face, bool linear_upwind,
                              bool limited, bool transpose_term, double* source);
    // True if the kernels run on an accelerator (not on host threads).
    [[nodiscard]] bool on_device() const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

} // namespace cfd::gpu
