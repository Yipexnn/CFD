// Jacobi-preconditioned CG / BiCGStab with OpenMP: target offload when compiled with
// CFD_GPU_OMP_TARGET (the OpenMP runtime falls back to the host without a device), host
// threads otherwise. All vectors stay on the device for the whole solve.
#include <algorithm>
#include <cfd/gpu/krylov.hpp>
#include <cmath>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#define CFD_STR(x) #x

namespace cfd::gpu::detail {

// Host threads.
namespace host {
// Threads only for large vectors (thread start-up dominates small systems).
#if defined(_OPENMP) && defined(_MSC_VER) && !defined(__clang__)
#define CFD_LOOP __pragma(omp parallel for if (n > 20000))
#define CFD_LOOP_SUM(var) __pragma(omp parallel for reduction(+ : var) if (n > 20000))
#elif defined(_OPENMP)
#define CFD_LOOP _Pragma("omp parallel for if (n > 20000)")
#define CFD_LOOP_SUM(var) _Pragma(CFD_STR(omp parallel for reduction(+ : var) if (n > 20000)))
#else
#define CFD_LOOP
#define CFD_LOOP_SUM(var)
#endif
#include "krylov_kernels.inc"
#undef CFD_LOOP
#undef CFD_LOOP_SUM
} // namespace host

struct HostSpace {
    using Arrays = host::Arrays;
    static SolveStats cg(const Arrays& a, const double* b, double* x, const double* d, double* r,
                         double* z, double* p, double* q, const SolveControl& c) {
        return host::cg(a, b, x, d, r, z, p, q, c);
    }
    static SolveStats bicgstab(const Arrays& a, const double* b, double* x, const double* d,
                               double* r, double* r0, double* p, double* v, double* s, double* t,
                               double* ph, double* sh, const SolveControl& c) {
        return host::bicgstab(a, b, x, d, r, r0, p, v, s, t, ph, sh, c);
    }
};

#if defined(CFD_GPU_OMP_TARGET) && defined(_OPENMP)
// OpenMP target offload (arrays mapped once per solve by the caller).
namespace offload {
#define CFD_LOOP _Pragma("omp target teams distribute parallel for")
#define CFD_LOOP_SUM(var)                                                                          \
    _Pragma(CFD_STR(omp target teams distribute parallel for reduction(+ : var)))
#include "krylov_kernels.inc"
#undef CFD_LOOP
#undef CFD_LOOP_SUM
} // namespace offload

struct OffloadSpace {
    using Arrays = offload::Arrays;
    static SolveStats cg(const Arrays& a, const double* b, double* x, const double* d, double* r,
                         double* z, double* p, double* q, const SolveControl& c) {
        return offload::cg(a, b, x, d, r, z, p, q, c);
    }
    static SolveStats bicgstab(const Arrays& a, const double* b, double* x, const double* d,
                               double* r, double* r0, double* p, double* v, double* s, double* t,
                               double* ph, double* sh, const SolveControl& c) {
        return offload::bicgstab(a, b, x, d, r, r0, p, v, s, t, ph, sh, c);
    }
};
#endif

bool openmp_device_available() {
#if defined(CFD_GPU_OMP_TARGET) && defined(_OPENMP)
    return omp_get_num_devices() > 0;
#else
    return false;
#endif
}

SolveStats solve_openmp(const CsrView& m, const double* b, double* x, Method method,
                        const SolveControl& control, bool allow_device) {
    const int n = m.n;
    const int nnz = m.nnz;
    std::vector<double> dinv_v(static_cast<std::size_t>(n), 1.0);
    for (int i = 0; i < n; ++i) {
        for (int k = m.row_ptr[i]; k < m.row_ptr[i + 1]; ++k) {
            if (m.col[k] == i && m.val[k] != 0.0) {
                dinv_v[static_cast<std::size_t>(i)] = 1.0 / m.val[k];
            }
        }
    }
    const int nvec = method == Method::Cg ? 4 : 8;
    std::vector<double> work_v(static_cast<std::size_t>(nvec) * static_cast<std::size_t>(n), 0.0);
    const int* rp = m.row_ptr;
    const int* ci = m.col;
    const double* va = m.val;
    double* dinv = dinv_v.data();
    double* w = work_v.data();
    const int nw = nvec * n;
    SolveStats st;
    auto run = [&](auto space) {
        using Space = decltype(space);
        const typename Space::Arrays a{n, rp, ci, va};
        if (method == Method::Cg) {
            return Space::cg(a, b, x, dinv, w, w + n, w + 2 * n, w + 3 * n, control);
        }
        return Space::bicgstab(a, b, x, dinv, w, w + n, w + 2 * n, w + 3 * n, w + 4 * n, w + 5 * n,
                               w + 6 * n, w + 7 * n, control);
    };
#if defined(CFD_GPU_OMP_TARGET) && defined(_OPENMP)
    if (allow_device && openmp_device_available()) {
#pragma omp target data map(to : rp[0 : n + 1], ci[0 : nnz], va[0 : nnz], b[0 : n], dinv[0 : n])   \
    map(tofrom : x[0 : n]) map(alloc : w[0 : nw])
        { st = run(OffloadSpace{}); }
        st.on_device = true;
        return st;
    }
#endif
    (void)nnz;
    (void)allow_device;
    (void)nw;
    st = run(HostSpace{});
    st.on_device = false;
    return st;
}

SolveStats solve_openmp(const CsrView& m, const double* b, double* x, Method method,
                        const SolveControl& control) {
    return solve_openmp(m, b, x, method, control, true);
}

} // namespace cfd::gpu::detail

namespace cfd::gpu {

SolveStats solve_host(const CsrView& a, const double* b, double* x, Method method,
                      const SolveControl& control) {
    return detail::solve_openmp(a, b, x, method, control, false);
}

} // namespace cfd::gpu
